# -*- coding: utf-8 -*-
"""三个音乐驱动 3D 舞蹈生成模型的本地工作台（纯标准库，不依赖任何第三方包）。

它做的事情：
    - 用一张网页代替"背命令行"：每个模型能干什么、要什么输入、产出什么，都写在页面上
    - 上传或指定一段音乐，点一下按钮就跑
    - 实时回显子进程日志，跑完直接在页面里播放视频
    - 汇总浏览历史上生成过的全部成品

设计上的几个刻意选择：
    - 本文件**不 import torch**，所有重活都在各自仓库的 venv 子进程里跑，
      工作台本身用哪个 Python 启动都行。
    - 同时只允许一个任务（三个模型都要吃满 16GB 显存，并发必 OOM）。
    - 页面每 2 秒发一次心跳；心跳断了（用户关了页面）且没有任务在跑，服务自动退出。
      这样"双击 bat → 开页面 → 用完关页面"就是完整的开关，不用去任务管理器杀进程。

用法：
    python serve.py [--port 18066] [--no-browser]
"""
import argparse
import glob
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import wave
import webbrowser

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录（tools/ 的上一级）
TOOLS = os.path.join(ROOT, "tools")
UPLOADS = os.path.join(TOOLS, "_uploads")
ASSETS = os.path.join(TOOLS, "_assets")
STASH = os.path.join(TOOLS, "_stash")
OUTROOT = os.path.join(ROOT, "_workbench_out")

EDGE = os.path.join(ROOT, "EDGE")
LODGE = os.path.join(ROOT, "LODGE")
BAILANDO = os.path.join(ROOT, "Bailando")

PY = {
    "edge": os.path.join(EDGE, "venv", "Scripts", "python.exe"),
    "lodge": os.path.join(LODGE, "venv", "Scripts", "python.exe"),
    "bailando": os.path.join(BAILANDO, "venv", "Scripts", "python.exe"),
}

LODGE_MUSIC_DIR = os.path.join(LODGE, "data", "finedance", "music")
LODGE_LABEL_DIR = os.path.join(LODGE, "data", "finedance", "label_json")
LODGE_SONG = "063"          # infer_lodge.py 的 test_list 里必须有这个 id
LODGE_CP1 = "exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt"
LODGE_CP2 = "exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt"

BAILANDO_PKL = os.path.join(BAILANDO, "experiments", "cc_motion_gpt", "eval", "pkl", "ep000400")
BAILANDO_CFG = "configs/cc_motion_gpt.yaml"

# LODGE 的风格标签，取自 LODGE/dld/data/FineDance_dataset.py 的 Genres_fd
LODGE_STYLES = [
    ("Hiphop", "嘻哈"),
    ("Breaking", "霹雳舞"),
    ("Popping", "机械舞"),
    ("Locking", "锁舞"),
    ("Urban", "都市编舞"),
    ("Jazz", "爵士"),
    ("Krump", "狂派"),
    ("House", "浩室"),
]

# 已生成成品的扫描位置： (模型, 目录, 是否算"工作台产出")
OUTPUT_SOURCES = [
    ("EDGE", os.path.join(EDGE, "renders")),
    ("LODGE", os.path.join(LODGE, "_lodge_video")),
    ("Bailando", os.path.join(BAILANDO, "_bailando_video")),
    ("EDGE", os.path.join(OUTROOT, "edge")),
    ("LODGE", os.path.join(OUTROOT, "lodge")),
    ("Bailando", os.path.join(OUTROOT, "bailando")),
]

PROGRESS_RE = re.compile(r"^\s*\d+%\|")

# --------------------------------------------------------------------------- #
# 全局状态
# --------------------------------------------------------------------------- #
LOCK = threading.RLock()
JOBS = {}                 # id -> job dict
JOB_ORDER = []            # 按创建顺序
CURRENT = {"id": None}    # 正在跑的任务 id
STATE = {"pinged": False, "last_ping": 0.0, "stop": False}
START_TS = time.time()
SEQ = [0]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _safe_stem(name):
    """把用户文件名变成安全的英文/数字 stem（模型会拿它当输出名）。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^0-9A-Za-z_\-]+", "_", stem).strip("_")
    stem = re.sub(r"_+", "_", stem)
    return stem or "music"


def wav_duration(path):
    """只支持 wav（上传时已经统一转成 wav）。返回秒数，失败返回 None。"""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def ffmpeg_to_wav(raw_bytes, dst):
    """把上传的原始字节转成模型要的 wav。

    刻意用 `-i pipe:0` 从 stdin 读：这样**不产生中间临时文件**，也就没有"用完还要删"
    这一步（删文件在受限环境里可能被拦，而且同扩展名时还会踩到 Windows 的
    WinError 32「自己拷自己」）。
    """
    exe = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [exe, "-y", "-loglevel", "error", "-i", "pipe:0",
           "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", dst]
    p = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("转成 wav 失败：%s" % p.stdout.decode("utf-8", "replace")[-400:])


def ensure_demo(which):
    """把仓库自带的示例音乐复制一份到 tools/_assets，之后就一直用它。"""
    os.makedirs(ASSETS, exist_ok=True)
    dst = os.path.join(ASSETS, "edge_demo_beat.wav" if which == "edge" else "lodge_demo_063.wav")
    if os.path.exists(dst):
        return dst
    src = (os.path.join(EDGE, "demo_music", "demo_beat.wav") if which == "edge"
           else os.path.join(LODGE_MUSIC_DIR, LODGE_SONG + ".wav"))
    if not os.path.exists(src):
        return None
    shutil.copy2(src, dst)
    return dst


def _log(job, text):
    """往任务日志里追加一行。tqdm 那种进度条会原地刷新，不刷屏。"""
    with LOCK:
        lines = job["lines"]
        if PROGRESS_RE.match(text) and lines and PROGRESS_RE.match(lines[-1]):
            lines[-1] = text
        else:
            lines.append(text)
        if len(lines) > 5000:
            del lines[:len(lines) - 3500]


def _run_streaming(job, cmd, cwd):
    """跑一条命令，stdout/stderr 合并后按行喂给任务日志。非 0 退出抛异常。"""
    _log(job, "")
    _log(job, "> " + " ".join('"%s"' % c if " " in c else c for c in cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, bufsize=0, env=env,
    )
    job["pid"] = proc.pid
    buf = b""
    while True:
        chunk = proc.stdout.read(8192)
        if not chunk:
            break
        buf += chunk
        while True:
            m = re.search(rb"[\r\n]", buf)
            if not m:
                break
            seg, buf = buf[:m.start()], buf[m.end():]
            s = seg.decode("utf-8", "replace").rstrip()
            if s:
                _log(job, s)
    if buf:
        s = buf.decode("utf-8", "replace").rstrip()
        if s:
            _log(job, s)
    rc = proc.wait()
    job["pid"] = None
    if rc != 0:
        raise RuntimeError("命令退出码 %d" % rc)


def _worker(job, plan):
    """任务线程：pre -> 逐条命令 -> post 收成品 -> final 收尾 -> cleanup 扫尾。

    三个收尾阶段的区别（这个划分很关键）：
        final    —— **必须完成**才允许任务算结束。典型是"把临时顶替的仓库文件还原回去"，
                    没做完就放锁，下一个任务会读到脏数据。同步执行。
        cleanup  —— **尽力而为**。典型是删临时目录。删不掉只影响磁盘，不该让用户干等，
                    所以放**后台守护线程**里跑：即使清理卡住（杀软锁文件、安全策略拦截），
                    任务状态也已经正常落成 done、锁也已经释放。
        status   —— 和释放任务锁写在同一个 with 里，保证前端不会看到
                    "产物已出来但顶部还显示运行中"这种自相矛盾的状态。
    """
    failed = None
    try:
        if plan.get("pre"):
            plan["pre"](job)
        for step in plan["steps"]:
            label, cmd, cwd = step(job) if callable(step) else step
            _log(job, "")
            _log(job, "===== %s =====" % label)
            _run_streaming(job, cmd, cwd)
        if plan.get("post"):
            job["videos"] = plan["post"](job) or []
    except Exception as exc:                                   # noqa: BLE001
        failed = exc
        _log(job, "")
        _log(job, "[失败] %s" % exc)

    if plan.get("final"):
        try:
            plan["final"](job)
        except Exception as exc:                               # noqa: BLE001
            _log(job, "[清理异常] %s" % exc)
    if not failed:
        _log(job, "")
        _log(job, "[完成] 任务结束")

    with LOCK:
        job["error"] = str(failed) if failed else ""
        job["status"] = "failed" if failed else "done"
        job["ended"] = time.time()
        CURRENT["id"] = None

    if plan.get("cleanup"):
        def _sweep():
            try:
                plan["cleanup"](job)
            except Exception as exc:                           # noqa: BLE001
                _log(job, "[清理] %s" % exc)
        threading.Thread(target=_sweep, daemon=True).start()


def start_job(model, title, plan):
    """占一个任务号并起线程（供内部/扩展使用）。"""
    with LOCK:
        if CURRENT["id"]:
            run = JOBS[CURRENT["id"]]
            return None, "已经有一个任务在跑了（%s）。" % run["title"]
        SEQ[0] += 1
        jid = "j%d" % SEQ[0]
    return start_job_with_id(model, jid, title, plan)


# --------------------------------------------------------------------------- #
# 三种任务的装配
# --------------------------------------------------------------------------- #
def _outdir_for(model):
    d = os.path.join(OUTROOT, model)
    os.makedirs(d, exist_ok=True)
    return d


def plan_edge(params, jid):
    music = params["music"]
    stem = _safe_stem(music)
    out_length = params["out_length"]
    cache = params["cache"]

    work = os.path.join(EDGE, "_workbench", jid)
    indir = os.path.join(work, "music")
    tmp_render = os.path.join(work, "renders")
    final_dir = _outdir_for("edge")

    def pre(job):
        os.makedirs(indir, exist_ok=True)
        os.makedirs(tmp_render, exist_ok=True)
        shutil.copy2(music, os.path.join(indir, stem + ".wav"))
        _log(job, "音乐：%s" % music)
        _log(job, "时长：%.1f 秒，生成长度：%d 秒" % (wav_duration(music) or 0, out_length))

    cmd = [PY["edge"], "-u", "test.py",
           "--music_dir", indir,
           "--render_dir", tmp_render,
           "--checkpoint", "checkpoint.pt",
           "--out_length", str(out_length)]
    if cache:
        cmd += ["--cache_features", "--feature_cache_dir", os.path.join(EDGE, "cached_features")]

    def post(job):
        vids = []
        for f in sorted(glob.glob(os.path.join(tmp_render, "*.mp4"))):
            dst = os.path.join(final_dir, "%s_%s.mp4" % (stem, jid))
            shutil.copy2(f, dst)
            vids.append(dst)
        return vids

    def cleanup(job):
        # 切片出来的音频和临时视频已经拷走了，留着只是占地方。
        # 放 cleanup（后台线程）而不是 final：删不掉不该让用户干等。
        try:
            if os.path.isdir(work):
                shutil.rmtree(work, ignore_errors=True)
                _log(job, "[清理] 已删除临时目录 %s" % work)
        except Exception as exc:                               # noqa: BLE001
            _log(job, "[清理] 临时目录没删掉（不影响结果）：%s" % exc)

    return {
        "pre": pre,
        "steps": [("EDGE 生成舞蹈（Jukebox 特征 -> 扩散采样 -> 渲染 + 合音乐）", cmd, EDGE)],
        "post": post,
        "cleanup": cleanup,
    }


def plan_lodge(params, jid):
    music = params["music"]
    stem = _safe_stem(music)
    style = params["style"]
    do_render = params["render"]

    work = os.path.join(LODGE, "_workbench", jid)
    staged_wav = os.path.join(LODGE_MUSIC_DIR, LODGE_SONG + ".wav")
    staged_json = os.path.join(LODGE_LABEL_DIR, LODGE_SONG + ".json")
    stash_dir = os.path.join(STASH, "lodge")
    bak_wav = os.path.join(stash_dir, LODGE_SONG + ".wav")
    bak_json = os.path.join(stash_dir, LODGE_SONG + ".json")
    final_dir = _outdir_for("lodge")
    box = {}

    def pre(job):
        os.makedirs(work, exist_ok=True)
        os.makedirs(stash_dir, exist_ok=True)
        os.makedirs(LODGE_MUSIC_DIR, exist_ok=True)
        os.makedirs(LODGE_LABEL_DIR, exist_ok=True)
        # 只备份一次：备份里存的永远是仓库自带的那份 demo 数据
        if not os.path.exists(bak_wav) and os.path.exists(staged_wav):
            shutil.copy2(staged_wav, bak_wav)
        if not os.path.exists(bak_json) and os.path.exists(staged_json):
            shutil.copy2(staged_json, bak_json)
        box["had_wav"] = os.path.exists(staged_wav)
        box["had_json"] = os.path.exists(staged_json)

        shutil.copy2(music, staged_wav)
        with open(staged_json, "w", encoding="utf-8") as f:
            json.dump({"style2": style}, f, ensure_ascii=False)
        _log(job, "音乐：%s" % music)
        _log(job, "风格标签：%s（写入 %s.json）" % (style, LODGE_SONG))
        _log(job, "时长：%.1f 秒" % (wav_duration(music) or 0))

    def step_infer(job):
        return ("LODGE 生成动作（Global 1024 帧 + Local 256 帧精修）",
                [PY["lodge"], "-u", "infer_lodge.py",
                 "--cfg", "exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml",
                 "--cfg_assets", "configs/data/assets.yaml",
                 "--soft", "1.0", "--device", "0"], LODGE)

    def step_render(job):
        # infer_lodge 每次都会新建一个带时间戳的 samples_* 目录，取最新那个
        pat = os.path.join(LODGE, "experiments", "Local_Module", "FineDance_FineTuneV2_Local",
                           "samples_*", "concat", "npy", LODGE_SONG + ".npy")
        cands = [p for p in glob.glob(pat) if os.path.getmtime(p) >= job["started"] - 5]
        if not cands:
            cands = glob.glob(pat)
        if not cands:
            raise RuntimeError("没找到生成出来的动作文件（%s.npy）" % LODGE_SONG)
        npy = max(cands, key=os.path.getmtime)
        _log(job, "动作文件：%s" % npy)
        out_mp4 = os.path.join(final_dir, "%s_%s_%s.mp4" % (style, stem, jid))
        box["out_mp4"] = out_mp4
        return ("渲染骨架视频 + 合音乐",
                [PY["lodge"], "-u", os.path.join(ROOT, "_tools_lodge_viz.py"), npy,
                 "--music", music, "--out", out_mp4], LODGE)

    steps = [step_infer]
    if do_render:
        steps.append(step_render)

    def post(job):
        if box.get("out_mp4") and os.path.exists(box["out_mp4"]):
            return [box["out_mp4"]]
        return []

    def final(job):
        # 仓库数据目录恢复原状，跑多少次都不留痕。★ 这个必须同步做完才放锁。
        try:
            if box.get("had_wav") and os.path.exists(bak_wav):
                shutil.copy2(bak_wav, staged_wav)
            if box.get("had_json") and os.path.exists(bak_json):
                shutil.copy2(bak_json, staged_json)
            _log(job, "[清理] 已还原 LODGE 数据目录里的 %s.wav / %s.json" % (LODGE_SONG, LODGE_SONG))
        except Exception as exc:                               # noqa: BLE001
            _log(job, "[清理] 还原失败，请手动把 tools/_stash/lodge 里的文件拷回 LODGE/data/finedance/：%s" % exc)

    def cleanup(job):
        try:
            if os.path.isdir(work):
                shutil.rmtree(work, ignore_errors=True)
        except Exception as exc:                               # noqa: BLE001
            _log(job, "[清理] 临时目录没删掉（不影响结果）：%s" % exc)

    return {"pre": pre, "steps": steps, "post": post, "final": final, "cleanup": cleanup}


def plan_bailando(params, jid):
    dance = params["dance"]
    do_all = dance == "__ALL__"
    final_dir = _outdir_for("bailando")

    if do_all:
        names = sorted(f[:-8] for f in os.listdir(BAILANDO_PKL) if f.endswith(".pkl.npy"))
        title = "Bailando 渲染全部 %d 条内置测试序列" % len(names)
    else:
        names = [dance]
        title = "Bailando 渲染 %s" % dance

    cmd = [PY["bailando"], "-u", os.path.join(ROOT, "_tools_render_from_pkl.py")]
    if do_all:
        cmd += ["--all", "--jobs", "6"]
    else:
        cmd += [dance]
    cmd += ["--out-dir", final_dir]

    def post(job):
        # 只报这次跑出来的（--all 时是 40 个，页面会截断显示并指向成品库）
        files = sorted(glob.glob(os.path.join(final_dir, "*.mp4")))
        fresh = [f for f in files if os.path.getmtime(f) >= job["started"] - 5]
        return fresh or files

    return {
        "steps": [("从已生成的动作数据渲染骨架视频（%d 条）" % len(names), cmd, BAILANDO)],
        "post": post,
    }


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def resolve_music(params):
    """把前端传来的 music 字段解析成一个真实的 wav 绝对路径。"""
    kind = params.get("music_kind")
    if kind == "demo":
        p = ensure_demo(params["model"])
        if not p:
            raise ValueError("仓库里没找到自带示例音乐，请改用上传或本地路径。")
        return p
    if kind == "path":
        p = (params.get("music_path") or "").strip().strip('"')
        if not p:
            raise ValueError("请填写本地音乐文件的完整路径。")
        if not os.path.isabs(p):
            p = os.path.join(ROOT, p)
        if not os.path.isfile(p):
            raise ValueError("找不到这个文件：%s" % p)
        return p
    if kind == "upload":
        p = params.get("music_upload") or ""
        if not os.path.isfile(p):
            raise ValueError("上传的音乐不见了，请重新上传。")
        return p
    raise ValueError("请先选择一段音乐。")


def validate(model, params):
    """返回 (规整后的 params, 提示列表)。有问题直接抛 ValueError。"""
    notes = []
    params = dict(params)
    params["model"] = model

    if model == "bailando":
        # Bailando 不吃音乐，只认内置的 40 条序列
        dance = params.get("dance") or ""
        if dance != "__ALL__":
            if not os.path.isfile(os.path.join(BAILANDO_PKL, dance + ".pkl.npy")):
                raise ValueError("没有这条内置序列：%s" % (dance or "(空)"))
        params["dance"] = dance
        return params, notes

    params["music"] = resolve_music(params)
    dur = wav_duration(params["music"])
    if dur is None:
        raise ValueError("读不出这段音频的时长，可能不是标准 wav。")
    if dur < 8:
        raise ValueError("音乐只有 %.1f 秒，太短了，至少要 8 秒。" % dur)

    if model == "edge":
        want = int(params.get("out_length") or 30)
        want = max(5, min(60, want))
        # EDGE 的 slice_audio 是「窗口 5s、步长 2.5s」：
        #   片数 n_slices = floor((dur-5)/2.5) + 1
        # test.py 里 sample_size = int(out_length/2.5) - 1，随后
        #   randint(0, n_slices - sample_size)
        # 负数会直接抛 ValueError，所以必须 n_slices >= sample_size，
        # 反解出 out_length <= 2.5 * (floor((dur-5)/2.5) + 2)。
        n_win = int((dur - 5.0) / 2.5 + 1e-6)
        limit = 2.5 * (n_win + 2)
        limit = int(math.floor(limit / 2.5 + 1e-6)) * 2.5
        limit = int(max(5, min(60, limit)) // 2.5 * 2.5)
        if want > limit:
            notes.append("音乐只有 %.1f 秒，生成长度已从 %d 秒下调到 %g 秒"
                         "（EDGE 要凑够一个完整时间窗，否则会直接报错）。" % (dur, want, limit))
            want = limit
        params["out_length"] = want
        params["cache"] = bool(params.get("cache"))
    elif model == "lodge":
        style = params.get("style") or "Hiphop"
        valid = [s for s, _ in LODGE_STYLES]
        if style not in valid:
            style = "Hiphop"
        params["style"] = style
        params["render"] = params.get("render", True) is not False
        if dur > 45:
            notes.append("音乐 %.1f 秒，LODGE 会按窗口逐段生成，可能要跑十几分钟。" % dur)
    return params, notes


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def scan_outputs():
    out = []
    seen = set()
    for model, d in OUTPUT_SOURCES:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            rp = os.path.realpath(f)
            if rp in seen:
                continue
            seen.add(rp)
            st = os.stat(f)
            out.append({
                "model": model,
                "name": os.path.basename(f),
                "size": st.st_size,
                "mtime": st.st_mtime,
                "url": "/media?path=" + urllib.parse.quote(f),
                "is_workbench": os.path.realpath(d).startswith(os.path.realpath(OUTROOT)),
            })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def list_dances():
    items = []
    if os.path.isdir(BAILANDO_PKL):
        vids = set()
        vd = os.path.join(BAILANDO, "_bailando_video")
        if os.path.isdir(vd):
            vids = {os.path.splitext(f)[0] for f in os.listdir(vd) if f.endswith(".mp4")}
        for f in sorted(os.listdir(BAILANDO_PKL)):
            if f.endswith(".pkl.npy"):
                n = f[:-8]
                items.append({"name": n, "video": n in vids})
    return items


def state_payload():
    with LOCK:
        busy = CURRENT["id"]
        jobs = [{
            "id": j["id"], "model": j["model"], "title": j["title"],
            "status": j["status"], "started": j["started"], "ended": j["ended"],
            "videos": j["videos"], "error": j["error"],
        } for j in (JOBS[i] for i in reversed(JOB_ORDER))]
    return {
        "busy": busy,
        "jobs": jobs[:20],
        "outputs": scan_outputs(),
        "styles": [{"value": s, "label": "%s（%s）" % (s, c)} for s, c in LODGE_STYLES],
        "dances": list_dances(),
        "demos": {
            "edge": ensure_demo("edge"),
            "lodge": ensure_demo("lodge"),
        },
        "has_gpu_job": busy is not None,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DanceWorkbench/1.0"

    # 别把每个请求都打到控制台
    def log_message(self, fmt, *args):
        pass

    # ---------------------------------------------------------------- 基础 ---
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def _json_body(self):
        raw = self._body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _inside_root(self, path):
        rp = os.path.realpath(path)
        return rp.startswith(os.path.realpath(ROOT) + os.sep)

    # ------------------------------------------------------------------ GET ---
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            p = os.path.join(TOOLS, "index.html")
            if not os.path.exists(p):
                return self._send(500, "缺 tools/index.html", "text/plain; charset=utf-8")
            with open(p, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")

        if u.path == "/api/state":
            return self._json(state_payload())

        if u.path == "/api/job":
            jid = (q.get("id") or [""])[0]
            frm = int((q.get("from") or ["0"])[0])
            with LOCK:
                job = JOBS.get(jid)
                if not job:
                    return self._json({"error": "没有这个任务"}, 404)
                lines = job["lines"][frm:]
                nxt = len(job["lines"])
                return self._json({
                    "id": jid, "status": job["status"], "lines": lines, "next": nxt,
                    "videos": [{"name": os.path.basename(v), "path": v,
                                "url": "/media?path=" + urllib.parse.quote(v)}
                               for v in job["videos"]],
                    "error": job["error"],
                    "title": job["title"], "model": job["model"],
                })

        if u.path == "/media":
            p = (q.get("path") or [""])[0]
            if not p or not os.path.isfile(p) or not self._inside_root(p):
                return self._send(404, "not found", "text/plain")
            return self._media(p)

        return self._send(404, "not found", "text/plain")

    def _media(self, path):
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = int(m.group(2))
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                status = 206
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ----------------------------------------------------------------- POST ---
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)

        if u.path == "/api/ping":
            STATE["pinged"] = True
            STATE["last_ping"] = time.time()
            return self._json({"ok": True})

        if u.path == "/api/quit":
            self._json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.3), _shutdown("收到关闭指令")),
                             daemon=True).start()
            return

        if u.path == "/api/upload":
            raw = self._body()
            name = urllib.parse.unquote(self.headers.get("X-Filename") or "upload")
            stem = _safe_stem(name)
            ext = os.path.splitext(name)[1].lower()
            os.makedirs(UPLOADS, exist_ok=True)
            tag = time.strftime("%H%M%S") + "%03d" % (int(time.time() * 1000) % 1000)
            dst = os.path.join(UPLOADS, "%s_%s.wav" % (tag, stem))
            try:
                if ext == ".wav":
                    with open(dst, "wb") as f:
                        f.write(raw)
                else:
                    ffmpeg_to_wav(raw, dst)     # 从 stdin 读，不落中间文件
            except Exception as exc:                           # noqa: BLE001
                return self._json({"ok": False, "error": "处理音频失败：%s" % exc}, 400)
            return self._json({"ok": True, "path": dst, "name": os.path.basename(dst),
                               "duration": wav_duration(dst)})

        if u.path == "/api/run":
            try:
                params = self._json_body()
            except Exception as exc:                           # noqa: BLE001
                return self._json({"ok": False, "error": "请求格式不对：%s" % exc}, 400)
            model = params.get("model")
            if model not in ("edge", "lodge", "bailando"):
                return self._json({"ok": False, "error": "未知模型"}, 400)
            try:
                clean, notes = validate(model, params)
            except ValueError as exc:
                return self._json({"ok": False, "error": str(exc)}, 400)

            # 先占住任务号（同时保证同一时刻只有一个任务），再按号装配计划
            with LOCK:
                if CURRENT["id"]:
                    run = JOBS[CURRENT["id"]]
                    return self._json({"ok": False,
                                       "error": "已经有一个任务在跑了（%s），请等它结束。" % run["title"]}, 409)
                SEQ[0] += 1
                jid = "j%d" % SEQ[0]

            if model == "edge":
                title = "EDGE · %s（%d 秒）" % (_safe_stem(clean["music"]), clean["out_length"])
                plan = plan_edge(clean, jid)
            elif model == "lodge":
                title = "LODGE · %s（%s）" % (_safe_stem(clean["music"]), clean["style"])
                plan = plan_lodge(clean, jid)
            else:
                title = ("Bailando · 全部 %d 条" % len(list_dances())
                         if clean["dance"] == "__ALL__" else "Bailando · " + clean["dance"])
                plan = plan_bailando(clean, jid)

            job, err = start_job_with_id(model, jid, title, plan)
            if err:
                return self._json({"ok": False, "error": err}, 409)
            return self._json({"ok": True, "job": job["id"], "title": title, "notes": notes})

        return self._send(404, "not found", "text/plain")


def start_job_with_id(model, jid, title, plan):
    with LOCK:
        if CURRENT["id"]:
            run = JOBS[CURRENT["id"]]
            return None, "已经有一个任务在跑了（%s）。" % run["title"]
        job = {
            "id": jid, "model": model, "title": title, "status": "running",
            "lines": [], "videos": [], "error": "", "pid": None,
            "started": time.time(), "ended": None,
        }
        JOBS[jid] = job
        JOB_ORDER.append(jid)
        CURRENT["id"] = jid
    threading.Thread(target=_worker, args=(job, plan), daemon=True).start()
    return job, None


# --------------------------------------------------------------------------- #
# 心跳看门狗：页面关了就把自己关掉
# --------------------------------------------------------------------------- #
def _shutdown(reason):
    print("\n[工作台] %s，正在退出……" % reason)
    sys.stdout.flush()
    STATE["stop"] = True
    os._exit(0)


def _port_alive(port, timeout=0.6):
    """端口上是否已经有服务在应答。见 main() 里关于 SO_REUSEADDR 的说明。"""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def watchdog():
    while not STATE["stop"]:
        time.sleep(1.0)
        if not STATE["pinged"]:
            if time.time() - START_TS > 180:
                _shutdown("页面一直没有打开")
            continue
        with LOCK:
            busy = CURRENT["id"] is not None
        if busy:
            continue          # 任务跑完再说，别把跑了一半的活儿掐了
        if time.time() - STATE["last_ping"] > 12:
            _shutdown("页面已关闭")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18066)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    for p in (UPLOADS, ASSETS, STASH, OUTROOT):
        os.makedirs(p, exist_ok=True)

    # ★ Windows 上 HTTPServer 默认带 SO_REUSEADDR，它允许**两个进程同时绑定同一个端口**
    #   （不像 Linux 只对 TIME_WAIT 生效）。这样双击两次 bat 会起两个服务，
    #   请求被随机分给其中一个，症状是"日志时有时无、quit 关不掉"。
    #   所以先探一下端口：有人应答就认定已经在运行，直接把浏览器叫起来然后退出。
    if _port_alive(args.port):
        print("[工作台] 端口 %d 上已经有一个工作台在运行了。" % args.port)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:%d/" % args.port)
        return 0

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if getattr(exc, "errno", None) in (48, 98, 10048):
            print("[工作台] 端口 %d 已被占用，说明工作台可能已经在运行了。" % args.port)
            if not args.no_browser:
                webbrowser.open("http://127.0.0.1:%d/" % args.port)
            return 0
        raise

    url = "http://127.0.0.1:%d/" % args.port
    print("=" * 62)
    print("  音乐驱动 3D 舞蹈生成 · 本地工作台")
    print("=" * 62)
    print("  地址：%s" % url)
    print("  工作区：%s" % ROOT)
    print("  关掉浏览器页面，工作台会自动退出。")
    print("=" * 62)
    print()
    sys.stdout.flush()

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print("\n[工作台] 已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
