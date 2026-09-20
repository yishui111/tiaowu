"""Minimal source/config patches needed to run these 2022-2024 dance repos on the
modern stack (torch 2.5.1 / numpy 2.2 / PyYAML 6) that this machine has, plus
fixing paths that were hardcoded to the authors' training boxes.

Idempotent: re-running prints "SKIP (pattern not found)" for already-applied
patches. Handles CRLF files (these repos are CRLF) by normalising to LF, patching,
then restoring CRLF.

Usage: python _tools_patch.py
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
AUTHOR_BOX = "/data2/lrh/project/dance/Lodge/lodge302/"


def patch(relpath, old, new):
    p = os.path.join(ROOT, relpath)
    if not os.path.exists(p):
        print("  MISSING FILE: %s" % relpath)
        return
    with open(p, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")
    if old not in text:
        print("  SKIP (pattern not found): %s" % relpath)
        return
    n = text.count(old)
    text = text.replace(old, new)
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(p, "wb") as f:
        f.write(text.encode("utf-8"))
    print("  PATCHED %-58s x%d%s" % (relpath, n, "  [crlf]" if crlf else ""))


print("== numpy 2.x: np.float removed ==")
patch("LODGE/dld/data/render_joints/utils/quaternion.py",
      "np.finfo(np.float).eps", "np.finfo(float).eps")

print("== PyYAML 6: yaml.load needs an explicit Loader ==")
for f in [
    "Bailando/main.py",
    "Bailando/main_actor_critic.py",
    "Bailando/main_gpt_all.py",
    "Bailando/tpami_bailandopp/main_actor_critic.py",
    "Bailando/tpami_bailandopp/main_gpt_all.py",
    "Bailando/tpami_bailandopp/mix.py".replace("mix.py", "main_mix.py"),
]:
    patch(f, "yaml.load(f)", "yaml.load(f, Loader=yaml.FullLoader)")

print("== LODGE: normalizer path hardcoded to the author's training box ==")
for f in [
    "LODGE/exp/Global_Module/FineDance_Global/global_train.yaml",
    "LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml",
]:
    patch(f, AUTHOR_BOX + "data/Normalizer.pth", "data/Normalizer.pth")

print("== LODGE: music shorter than one global window (cfg.length1) ==")
# infer_lodge.py slices music_fea_full[-cfg.length1:] and assumes it yields
# exactly cfg.length1 frames. If the wav is shorter than length1/fps seconds the
# slice comes up short, cond_tokens get a shorter sequence dim than the model's
# null_cond_embed (built with seq_len=length1), and torch.where blows up with
# "The size of tensor a (768) must match the size of tensor b (1024)".
# Zero-pad the tail so the window is always full.
patch("LODGE/infer_lodge.py",
      '        print("music_fea_full", music_fea_full.shape)\n'
      '        local_num = music_fea_full.shape[0] // cfg.length2\n',
      '        print("music_fea_full", music_fea_full.shape)\n'
      '        # 音乐短于一个 global 窗口(cfg.length1 帧)时，末尾补零。\n'
      '        # 否则下面的 music_fea_full[-cfg.length1:] 取不满，cond_tokens 的序列长度\n'
      '        # 会小于模型 null_cond_embed 的 seq_len，torch.where 直接报 shape 错。\n'
      '        if music_fea_full.shape[0] < cfg.length1:\n'
      '            _pad_len = cfg.length1 - music_fea_full.shape[0]\n'
      '            _pad = np.zeros((_pad_len, music_fea_full.shape[1]), dtype=music_fea_full.dtype)\n'
      '            music_fea_full = np.concatenate([music_fea_full, _pad], axis=0)\n'
      '            print("music shorter than one window -> zero-padded to", music_fea_full.shape)\n'
      '        local_num = music_fea_full.shape[0] // cfg.length2\n')

print("== pytorch3d stub: matrix_to_quaternion was wrong twice over ==")
# 1) the 0.5/sqrt(k) scalar lacked unsqueeze(-1), so for any batched
#    (..., J, 3, 3) input it was broadcast against the wrong axis and raised
#    "The size of tensor a (22) must match the size of tensor b (4)".
# 2) the stacked components were (x, y, z, w) while the rest of the file
#    (axis_angle_to_quaternion / quaternion_to_matrix / quaternion_to_axis_angle)
#    is wxyz -- so even the unbatched case returned a scrambled quaternion.
_OLD_CASES = (
    '    case_x = 0.5 / _sqrt(kx) * torch.stack([kx, m01 + m10, m02 + m20, m21 - m12], dim=-1)\n'
    '    case_y = 0.5 / _sqrt(ky) * torch.stack([m01 + m10, ky, m12 + m21, m02 - m20], dim=-1)\n'
    '    case_z = 0.5 / _sqrt(kz) * torch.stack([m02 + m20, m12 + m21, kz, m10 - m01], dim=-1)\n'
    '    case_w = 0.5 / _sqrt(kw) * torch.stack([m21 - m12, m02 - m20, m10 - m01, kw], dim=-1)\n'
)
_NEW_CASES = (
    '    # each case must yield (w, x, y, z) to match the wxyz convention used by the\n'
    '    # rest of this file; the scalar is unsqueezed so it broadcasts over the last axis\n'
    '    case_x = (0.5 / _sqrt(kx)).unsqueeze(-1) * torch.stack([m21 - m12, kx, m01 + m10, m02 + m20], dim=-1)\n'
    '    case_y = (0.5 / _sqrt(ky)).unsqueeze(-1) * torch.stack([m02 - m20, m01 + m10, ky, m12 + m21], dim=-1)\n'
    '    case_z = (0.5 / _sqrt(kz)).unsqueeze(-1) * torch.stack([m10 - m01, m02 + m20, m12 + m21, kz], dim=-1)\n'
    '    case_w = (0.5 / _sqrt(kw)).unsqueeze(-1) * torch.stack([kw, m21 - m12, m02 - m20, m10 - m01], dim=-1)\n'
)
for f in [
    "LODGE/pytorch3d/transforms/__init__.py",
    "EDGE/pytorch3d/transforms/__init__.py",
]:
    patch(f, _OLD_CASES, _NEW_CASES)
