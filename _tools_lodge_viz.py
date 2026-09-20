"""LODGE: render the generated 139-dim motion as a 2-D skeleton video WITH music.

Why this exists: the repo's own render.py needs SMPL/SMPLH/SMPLX *mesh* files
(SMPLX_NEUTRAL.npz etc.) which require registration at smpl.is.tue.mpg.de, and it
hardcodes the author's absolute paths. But `dld/data/render_joints/smplfk.py`
ships its own rest-pose joints in `data/smplx_neu_J_1.npy`, so forward kinematics
works offline with no licensed model. This renders joint positions instead of a
mesh -- same idea as Bailando's skeleton videos.

Frame rate: the demo wav is 32.0 s and produced 961 music-feature frames
(see dld/data/utils/audio.py, HOP_LENGTH=512, extract(fps=30)), so the dance is
1:1 with music features at 30 fps. `FPS: 12.5` in the yaml is a render-time
downsample setting, not the data rate.

Usage (from LODGE/):
    ./venv/Scripts/python.exe ../_tools_lodge_viz.py <npy> [--music <wav>] [--no-audio] [--out <mp4>]

`--out` overrides the output path (default: LODGE/_lodge_video/<npy 名>.mp4)；
工作台就是靠它把成品统一收进 _workbench_out/lodge/ 的。
"""
import io
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LODGE")
OUT_DIR = "_lodge_video"
FPS = 30
W, H = 960, 960
WORLD_H = 2.2          # metres of vertical world covered by the frame
WORLD_CY = 0.90        # world y that sits at the frame centre

BG = (18, 20, 26)
BONE = (120, 200, 255)
JOINT = (235, 245, 255)
TOUCH = (255, 92, 92)      # foot in contact with the floor
FREE = (110, 220, 140)     # foot off the floor
FLOOR = (70, 78, 92)

# smplx_parents limited to the 22 body joints (indices 0..21)
BONES = [(1, 0), (2, 0), (3, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 5),
         (9, 6), (10, 7), (11, 8), (12, 9), (13, 9), (14, 9), (15, 12),
         (16, 13), (17, 14), (18, 16), (19, 17), (20, 18), (21, 19)]
# contact[0..3] belongs to these joints (see smplfk.plot_single_pose)
CONTACT_JOINTS = [7, 8, 10, 11]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    import torch
    from PIL import Image, ImageDraw
    from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk

    argv = sys.argv[1:]
    no_audio = "--no-audio" in argv
    music = None
    out = None
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--music" and i + 1 < len(argv):
            music = argv[i + 1]
            i += 2
            continue
        if a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
            continue
        if not a.startswith("--"):
            positional.append(a)
        i += 1
    if not positional:
        print(__doc__)
        return 1
    npy = positional[0]
    if not os.path.exists(npy):
        print("no such npy:", npy)
        return 1

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SMPLX_Skeleton(device=dev, Jpath="data/smplx_neu_J_1.npy")
    data = torch.from_numpy(np.load(npy).astype(np.float32))
    joints = do_smplxfk(data, model).detach().cpu().numpy()   # (T, 55, 3)
    T = joints.shape[0]
    contact = np.load(npy)[:, :4]
    print("npy     :", npy)
    print("joints  :", joints.shape)

    # trim to the music length so we don't show the zero-padded tail
    if music and os.path.exists(music):
        import soundfile as sf
        dur = sf.info(music).duration
        n = min(T, int(dur * FPS))
        if n < T:
            print("trim    : %d -> %d frames (music is %.2fs)" % (T, n, dur))
        T = n

    scale = H / WORLD_H
    cx = W / 2.0
    cy = H / 2.0
    floor_y = cy + WORLD_CY * scale

    def to_px(p):
        return (cx + p[0] * scale, cy - (p[1] - WORLD_CY) * scale)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(npy))[0]
    out_mp4 = out or os.path.join(OUT_DIR, stem + ".mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "image2pipe", "-vcodec", "png", "-r", str(FPS), "-i", "-"]
    if music and not no_audio and os.path.exists(music):
        cmd += ["-i", music, "-map", "0:v", "-map", "1:a", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"]
    if music and not no_audio and os.path.exists(music):
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += [out_mp4]

    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(T):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        # floor
        d.line([(0, floor_y), (W, floor_y)], fill=FLOOR, width=3)
        pose = joints[i]
        # bones, depth-shaded by mean z
        for a, b in BONES:
            za = pose[a, 2]
            shade = float(np.clip(1.0 - (za + 0.7) * 0.55, 0.45, 1.0))
            col = tuple(int(c * shade) for c in BONE)
            d.line([to_px(pose[a]), to_px(pose[b])], fill=col, width=6)
        # joints
        for j in range(22):
            x, y = to_px(pose[j])
            r = 7
            col = JOINT
            if j in CONTACT_JOINTS:
                k = CONTACT_JOINTS.index(j)
                col = TOUCH if contact[i, k] > 0.95 else FREE
                r = 10
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        try:
            ff.stdin.write(buf.getvalue())
        except BrokenPipeError:
            print("ffmpeg closed stdin at frame", i)
            break
        if (i + 1) % 200 == 0:
            print("  frame %d/%d" % (i + 1, T), flush=True)

    try:
        ff.stdin.close()
    except Exception:
        pass
    ff.wait()

    size = os.path.getsize(out_mp4) if os.path.exists(out_mp4) else 0
    print("wrote   :", out_mp4, size, "bytes")
    return 0 if size else 2


if __name__ == "__main__":
    sys.exit(main())
