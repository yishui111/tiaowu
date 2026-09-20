"""Render Bailando sequences straight from the pkl files, bypassing json entirely.

Why: `write2json` writes ~3248 single-frame json files per sequence and the sandbox
rejects file writes past a quota, so the full eval died at sequence 17/40 and only
4 of 31 json dirs came out complete. The pkl files (written one per sequence, before
the json step) are all 40 intact, and `read_keypoints` only ever consumes a 25x3
pose array -- so the json round-trip is pure overhead here.

Resolution MUST be 960x540 -- that is what `configs/*.yaml` sets under `testing:`,
and `write2json` bakes it into the pixel coordinates it writes. Rendering at any
other size silently scales/offsets the skeleton. Verified: at 960x540 the pkl path
is pixel-identical to the json path (`--check`, max abs diff 0).

Pipeline: pkl -> 25 keypoints (same mapping as visualizeAndWritefromPKL) -> pixel
coords -> connect_keypoints -> PNG in memory -> ffmpeg stdin. Zero temp files.

Usage (from Bailando/):
    ./venv/Scripts/python.exe ../_tools_render_from_pkl.py <dance>          # one mp4
    ./venv/Scripts/python.exe ../_tools_render_from_pkl.py <dance> --check  # diff vs json path
    ./venv/Scripts/python.exe ../_tools_render_from_pkl.py --all --jobs 6   # every pkl, 6 at a time
    ./venv/Scripts/python.exe ../_tools_render_from_pkl.py <dance> --out-dir D:/somewhere

`--out-dir` 默认是 Bailando/_bailando_video；工作台用它把成品统一收进 _workbench_out/bailando/。
注意：多进程走的是 spawn，子进程会重新 import 本模块，所以输出目录必须**当参数传进去**，
不能靠改模块级全局变量（子进程里看不到）。
"""
import io
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Bailando")
PKL_DIR = os.path.join("experiments", "cc_motion_gpt", "eval", "pkl", "ep000400")
JSON_DIR = os.path.join("experiments", "cc_motion_gpt", "eval", "jsons", "ep000400")
OUT_DIR = "_bailando_video"
W, H = 960, 540          # configs/*.yaml -> testing: {height: 540, width: 960}
FPS = 60                 # utils/functional.py img2video: ffmpeg -r 60

# same joint mapping as utils.functional.visualizeAndWritefromPKL
JOINT_MAP = {
    0: 12, 1: 9, 2: 16, 3: 18, 4: 20, 5: 17, 6: 19, 7: 21,
    8: 0, 9: 1, 10: 4, 11: 7, 12: 2, 13: 5, 14: 8,
    15: 15, 16: 15, 17: 15, 18: 15,
    19: 11, 20: 11, 21: 8, 22: 10, 23: 10, 24: 7,
}


def pkl_to_25pts(pkl_path):
    """pkl -> (T, 25, 2) normalised 2-D keypoints, mirroring visualizeAndWritefromPKL."""
    result = np.load(pkl_path, allow_pickle=True).item()["pred_position"]
    d = np.array(result, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :24]
    b = d.shape[0]
    d = d.reshape(b, 24, 3)
    d = d - d[:1, :1, :]
    d2 = d[:, :, :2] / 1.5
    d2[:, :, 0] /= 2.2
    t = np.zeros((b, 25, 2), dtype=np.float32)
    for k, v in JOINT_MAP.items():
        t[:, k] = d2[:, v]
    return t


def _env():
    """Make ROOT importable / cwd-correct inside a spawned worker."""
    if os.getcwd() != ROOT:
        os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import cv2
    from PIL import Image
    from utils.keypoint2img import (
        define_edge_lists, extract_valid_keypoints, connect_keypoints,
    )
    return cv2, Image, define_edge_lists, extract_valid_keypoints, connect_keypoints


def _render_one(dance, out_dir=None):
    """Render one whole sequence to mp4. Module-level so Windows spawn can pickle it."""
    cv2, Image, define_edge_lists, extract_valid_keypoints, connect_keypoints = _env()
    out_dir = out_dir or OUT_DIR

    pkl_path = os.path.join(PKL_DIR, dance + ".pkl.npy")
    pts25 = pkl_to_25pts(pkl_path)
    T = pts25.shape[0]
    edge_lists = define_edge_lists(False)
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, dance.split(".")[0] + ".mp4")

    t0 = time.time()
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "image2pipe", "-vcodec", "png", "-r", str(FPS), "-i", "-",
         "-vb", "20M", "-vcodec", "mpeg4", out_mp4],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    err = ""
    try:
        for i in range(T):
            xs = (pts25[i, :, 0] + 1.0) * 0.5 * W
            ys = (pts25[i, :, 1] + 1.0) * 0.5 * H
            p = np.stack([xs, ys, np.full_like(xs, 0.8)], axis=-1)
            parts = [extract_valid_keypoints(p, edge_lists),
                     np.zeros((70, 3)), np.zeros((21, 3)), np.zeros((21, 3))]
            img3 = connect_keypoints(parts, edge_lists, (W, H), 0, False, False)
            buf = io.BytesIO()
            Image.fromarray(img3).save(buf, format="PNG")
            ff.stdin.write(buf.getvalue())
    except BrokenPipeError:
        err = "ffmpeg closed stdin"
    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.wait()

    size = os.path.getsize(out_mp4) if os.path.exists(out_mp4) else 0
    return dance, T, size, time.time() - t0, err


def _render_one_ctx(pair):
    """Pool 里跑的壳：把 (dance, out_dir) 拆开。spawn 下必须靠参数传，不能用全局。"""
    return _render_one(pair[0], pair[1])


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    argv = sys.argv[1:]
    check_only = "--check" in argv
    do_all = "--all" in argv
    jobs = 1
    out_dir = OUT_DIR
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--jobs" and i + 1 < len(argv):
            jobs = int(argv[i + 1]); i += 2; continue
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1]); i += 1; continue
        if a == "--out-dir" and i + 1 < len(argv):
            out_dir = argv[i + 1]; i += 2; continue
        if a.startswith("--out-dir="):
            out_dir = a.split("=", 1)[1]; i += 1; continue
        if not a.startswith("--"):
            args.append(a)
        i += 1

    if do_all:
        names = sorted(f[:-8] for f in os.listdir(PKL_DIR) if f.endswith(".pkl.npy"))
    elif args:
        names = [args[0]]
    else:
        print(__doc__)
        return 1

    # ---------------------------------------------------------------- check ---
    if check_only:
        cv2, Image, define_edge_lists, extract_valid_keypoints, connect_keypoints = _env()
        from utils.keypoint2img import read_keypoints

        dance = names[0]
        pts25 = pkl_to_25pts(os.path.join(PKL_DIR, dance + ".pkl.npy"))
        edge_lists = define_edge_lists(False)

        def post(img3):
            im = Image.fromarray(img3).transpose(Image.FLIP_TOP_BOTTOM)
            im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2BGRA)
            return np.asarray(Image.fromarray(np.uint8(im)))

        xs = (pts25[0, :, 0] + 1.0) * 0.5 * W
        ys = (pts25[0, :, 1] + 1.0) * 0.5 * H
        p = np.stack([xs, ys, np.full_like(xs, 0.8)], axis=-1)
        parts = [extract_valid_keypoints(p, edge_lists),
                 np.zeros((70, 3)), np.zeros((21, 3)), np.zeros((21, 3))]
        a = post(connect_keypoints(parts, edge_lists, (W, H), 0, False, False))

        jdir = os.path.join(JSON_DIR, dance)
        f0 = sorted(f for f in os.listdir(jdir) if "frame000000" in f)[0]
        b = post(read_keypoints(os.path.join(jdir, f0), (W, H),
                                remove_face_labels=False, basic_point_only=False))
        print("dance      :", dance)
        print("pkl frames :", pts25.shape[0])
        print("json frame :", f0)
        print("shape      : pkl=%s json=%s" % (a.shape, b.shape))
        if a.shape == b.shape:
            print("IDENTICAL  :", np.array_equal(a, b))
            print("max abs diff:", int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
            print("non-black  : pkl=%d json=%d" % (
                int((a[:, :, :3].sum(-1) > 0).sum()),
                int((b[:, :, :3].sum(-1) > 0).sum())))
            os.makedirs("_pkl_check", exist_ok=True)
            Image.fromarray(a).save(os.path.join("_pkl_check", "pkl_frame000000.png"))
            Image.fromarray(b).save(os.path.join("_pkl_check", "json_frame000000.png"))
            print("saved both to _pkl_check/")
        return 0

    # ---------------------------------------------------------------- batch ---
    os.makedirs(out_dir, exist_ok=True)
    print("sequences: %d   jobs: %d   %dx%d @ %dfps   -> %s"
          % (len(names), jobs, W, H, FPS, out_dir), flush=True)
    t_all = time.time()
    rc = 0
    if jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs) as pool:
            for dance, T, size, dt, err in pool.imap_unordered(
                    _render_one_ctx, [(n, out_dir) for n in names]):
                ok = size > 0 and not err
                print("  %-38s %5d frames  %7.1fs  %9d bytes  %s"
                      % (dance, T, dt, size, "OK" if ok else "FAIL " + err), flush=True)
                if not ok:
                    rc = 2
    else:
        for dance in names:
            dance, T, size, dt, err = _render_one(dance, out_dir)
            ok = size > 0 and not err
            print("  %-38s %5d frames  %7.1fs  %9d bytes  %s"
                  % (dance, T, dt, size, "OK" if ok else "FAIL " + err), flush=True)
            if not ok:
                rc = 2
    print("ALL_DONE rc=%d  total %.1f min" % (rc, (time.time() - t_all) / 60.0))
    return rc


if __name__ == "__main__":
    sys.exit(main())
