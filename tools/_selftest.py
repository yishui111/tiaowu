# -*- coding: utf-8 -*-
"""工作台自测：起服务 -> 查状态 -> 上传音频 -> 参数校验 -> 真跑任务 -> 校验产物 -> 关服务。

用法：
    python tools/_selftest.py quick     # Bailando 单条渲染（约 30 秒）
    python tools/_selftest.py full      # 再加一个真实的 EDGE 任务（约 5~8 分钟）

注意：本环境里子进程会随工具调用结束被回收，所以「起服务 -> 测完 -> 关服务」
必须全部塞在同一次调用里，别分两次跑。
"""
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 18066
BASE = "http://127.0.0.1:%d" % PORT
LOGF = os.path.join(ROOT, "tools", "_selftest_server.log")

OK = []
BAD = []


def chk(cond, label, detail=""):
    (OK if cond else BAD).append(label)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   <- " + str(detail)[:180]) if detail else ""))
    return bool(cond)


def req(path, data=None, headers=None, raw=None, timeout=120):
    h = dict(headers or {})
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        h["Content-Type"] = "application/json"
    if raw is not None:
        body = raw
    r = urllib.request.Request(BASE + path, data=body, headers=h,
                               method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def wait_port(t=30):
    """★ 本机对「已关闭的回环端口」是超时而不是拒绝连接（实测每次 2 秒），
    所以要用墙钟时间封顶，别只数次数，否则失败路径要等 4.5 分钟才报错。"""
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            with urllib.request.urlopen(BASE + "/api/state", timeout=1):
                return True
        except Exception:
            time.sleep(0.2)
    return False


def make_wav(path, seconds=20.0, sr=44100):
    """写一段带拍点的正弦测试 wav。"""
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        beat = 1.0 if (t % 0.5) < 0.06 else 0.0
        v = 0.35 * math.sin(2 * math.pi * 220 * t) + 0.25 * beat * math.sin(2 * math.pi * 80 * t)
        s = int(max(-1.0, min(1.0, v)) * 32000)
        frames += struct.pack("<hh", s, s)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return path


def poll(job_id, limit):
    frm, t0, shown = 0, time.time(), set()
    while time.time() - t0 < limit:
        try:
            req("/api/ping", data={})
        except Exception:
            pass
        st, j = req("/api/job?id=%s&from=%d" % (job_id, frm))
        if st != 200:
            time.sleep(2)
            continue
        frm = j.get("next", frm)
        for ln in (j.get("lines") or []):
            if ln and ln not in shown and not ln.startswith("  ") and "it/s" not in ln:
                if len(shown) < 12:
                    print("      | %s" % ln[:140])
                shown.add(ln)
        if j.get("status") != "running":
            print("      -> %.1f 秒，状态=%s" % (time.time() - t0, j.get("status")))
            return j
        time.sleep(2)
    return {"status": "timeout", "videos": [], "error": "本地超时"}


def ffprobe(path):
    exe = "ffprobe"
    p = subprocess.run([exe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,nb_frames",
                        "-show_entries", "format=duration",
                        "-of", "json", path],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        d = json.loads(p.stdout.decode("utf-8", "replace"))
        s = (d.get("streams") or [{}])[0]
        return int(s.get("width") or 0), int(s.get("height") or 0), \
            int(s.get("nb_frames") or 0), float((d.get("format") or {}).get("duration") or 0)
    except Exception:
        return 0, 0, 0, 0.0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    py = sys.executable
    print("=" * 66)
    print("工作台自测（%s）" % mode)
    print("解释器：%s" % py)
    print("=" * 66)

    log = open(LOGF, "wb")
    srv = subprocess.Popen([py, os.path.join(ROOT, "tools", "serve.py"),
                            "--port", str(PORT), "--no-browser"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    rc = 0
    try:
        # ------------------------------------------------------------------ 1
        print("\n[1] 启动服务")
        if not chk(wait_port(30), "服务在 30 秒内起来并响应 /api/state"):
            log.flush()
            print("\n服务没起来，日志尾部：")
            print(open(LOGF, "rb").read().decode("utf-8", "replace")[-2500:])
            return 1
        # ★ 必须确认「应答的就是我们刚起的那个进程」：Windows 的 SO_REUSEADDR
        #   允许两个进程绑同一端口，否则会误测到残留的旧服务，症状是 quit 关不掉。
        chk(srv.poll() is None,
            "刚起的服务进程还活着（没被「端口已占用」挡掉）",
            "退出码 %s，日志：%s" % (srv.poll(),
                open(LOGF, "rb").read().decode("utf-8", "replace")[-200:]))

        # ------------------------------------------------------------------ 2
        print("\n[2] GET /api/state")
        st, s = req("/api/state")
        chk(st == 200, "HTTP 200")
        chk(len(s.get("styles") or []) >= 6, "返回 LODGE 风格列表",
            "%d 个" % len(s.get("styles") or []))
        chk(len(s.get("dances") or []) == 40, "返回 40 条 Bailando 序列",
            "%d 条" % len(s.get("dances") or []))
        chk(len(s.get("outputs") or []) >= 42, "扫到已有成品视频",
            "%d 个" % len(s.get("outputs") or []))
        chk(bool((s.get("demos") or {}).get("edge")), "EDGE 示例音乐就位")
        chk(bool((s.get("demos") or {}).get("lodge")), "LODGE 示例音乐就位")
        chk(s.get("busy") is None, "当前空闲")
        print("      示例音乐：edge=%s" % os.path.basename(str((s.get("demos") or {}).get("edge"))))
        print("                lodge=%s" % os.path.basename(str((s.get("demos") or {}).get("lodge"))))

        # ------------------------------------------------------------------ 3
        print("\n[3] 上传音频")
        tw = os.path.join(ROOT, "tools", "_selftest_20s.wav")
        make_wav(tw, 20.0)
        raw = open(tw, "rb").read()
        st, u = req("/api/upload", raw=raw,
                    headers={"X-Filename": urllib.parse.quote("测试_音乐.wav")})
        chk(st == 200 and u.get("ok"), "上传 wav 成功", u.get("error") or u.get("name"))
        chk(abs((u.get("duration") or 0) - 20.0) < 0.2, "服务端读出的时长正确",
            "%.2f 秒" % (u.get("duration") or -1))
        chk(u.get("name", "").startswith("0") or "_" in u.get("name", ""),
            "中文文件名被转成安全名", u.get("name"))
        st, u2 = req("/api/upload", raw=raw,
                     headers={"X-Filename": urllib.parse.quote("测试音乐.mp3")})
        chk(st == 200 and u2.get("ok") and str(u2.get("path", "")).endswith(".wav"),
            "非 wav 会被 ffmpeg 转成 wav", u2.get("error") or os.path.basename(str(u2.get("path"))))

        # ------------------------------------------------------------------ 4
        print("\n[4] 参数校验（这些都不该真启动任务）")
        st, r = req("/api/run", data={"model": "edge", "music_kind": "path",
                                      "music_path": r"G:\不存在\no.wav"})
        chk(st == 400 and not r.get("ok"), "不存在的路径被拒绝", r.get("error"))
        st, r = req("/api/run", data={"model": "bailando", "dance": "不存在的序列"})
        chk(st == 400 and not r.get("ok"), "不存在的序列被拒绝", r.get("error"))
        st, r = req("/api/run", data={"model": "nope"})
        chk(st == 400 and not r.get("ok"), "未知模型被拒绝", r.get("error"))

        # ------------------------------------------------------------------ 5
        plan = {"quick": ["bailando"], "edge": ["edge"], "lodge": ["lodge"],
                "full": ["bailando", "edge", "lodge"]}.get(mode, ["bailando"])
        print("\n[5] 端到端任务：%s" % "、".join(plan))

        def verify(label, j, expect_audio, limit_note=""):
            chk(j.get("status") == "done", "%s 跑完且状态为 done" % label, j.get("error"))
            vids = j.get("videos") or []
            chk(len(vids) >= 1, "%s 拿到产物视频" % label, "%d 个" % len(vids))
            if not vids:
                return
            p = vids[0]["path"]
            chk(os.path.exists(p), "%s 视频文件在磁盘上" % label, p)
            w, h, n, d = ffprobe(p)
            print("      ffprobe: %dx%d, %d 帧, %.1f 秒" % (w, h, n, d))
            chk(w > 0 and n > 100, "%s 视频有效" % label, "%dx%d %d 帧" % (w, h, n))
            if expect_audio:
                pr = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0", p],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                ac = pr.stdout.decode("utf-8", "replace").strip()
                chk(bool(ac), "%s 视频带音轨" % label, ac or "无音轨")
            rq = urllib.request.Request(BASE + vids[0]["url"],
                                        headers={"Range": "bytes=0-1023"})
            with urllib.request.urlopen(rq, timeout=20) as resp:
                code, ln = resp.status, len(resp.read())
            chk(code == 206 and ln == 1024, "%s 媒体接口支持 Range" % label,
                "HTTP %s, %d 字节" % (code, ln))

        def submit(payload, label):
            st, r = req("/api/run", data=payload)
            chk(st == 200 and r.get("ok"), "%s 任务已受理" % label,
                r.get("error") or r.get("title"))
            if not r.get("ok"):
                return None
            st2, r2 = req("/api/run", data=payload)
            chk(st2 == 409, "%s：任务进行中时再提交会被挡（409）" % label, r2.get("error"))
            return r

        if "bailando" in plan:
            dance = (s.get("dances") or [{}])[0].get("name")
            r = submit({"model": "bailando", "dance": dance}, "Bailando")
            if r:
                verify("Bailando", poll(r["job"], 500), expect_audio=False)

        if "edge" in plan:
            r = submit({"model": "edge", "music_kind": "demo",
                        "out_length": 30, "cache": False}, "EDGE")
            if r:
                chk(not (r.get("notes") or []), "30 秒示例音乐没有被降长度", r.get("notes"))
                verify("EDGE", poll(r["job"], 1500), expect_audio=True)

        if "lodge" in plan:
            mw = os.path.join(ROOT, "LODGE", "data", "finedance", "music", "063.wav")
            mj = os.path.join(ROOT, "LODGE", "data", "finedance", "label_json", "063.json")
            bw = os.path.join(ROOT, "tools", "_stash", "lodge", "063.wav")
            bj = os.path.join(ROOT, "tools", "_stash", "lodge", "063.json")
            before = (os.path.getsize(mw) if os.path.exists(mw) else 0,
                      open(mj, "rb").read() if os.path.exists(mj) else b"")
            r = submit({"model": "lodge", "music_kind": "demo", "style": "Breaking",
                        "render": True}, "LODGE")
            if r:
                verify("LODGE", poll(r["job"], 1800), expect_audio=True)
            after = (os.path.getsize(mw) if os.path.exists(mw) else 0,
                     open(mj, "rb").read() if os.path.exists(mj) else b"")
            chk(before == after, "跑完把 LODGE 数据目录还原成原样",
                "before=%s after=%s" % (before[0], after[0]))
            chk(os.path.exists(bw) and os.path.exists(bj), "备份文件在 tools/_stash/lodge 里",
                str(os.path.exists(bw)) + "/" + str(os.path.exists(bj)))
            print("      还原后 063.json = %s" % after[1].decode("utf-8", "replace")[:80])

        # ------------------------------------------------------------------ 6
        print("\n[6] 任务列表")
        st, s2 = req("/api/state")
        chk(len(s2.get("jobs") or []) >= 1, "任务历史可查", "%d 条" % len(s2.get("jobs") or []))
        chk(s2.get("busy") is None, "所有任务结束后回到空闲")

        # ------------------------------------------------------------------ 7
        print("\n[7] 关闭工作台")
        st, _ = req("/api/quit", data={})
        chk(st == 200, "quit 接口返回 200")
        gone = False
        for _ in range(40):
            time.sleep(0.5)
            if srv.poll() is not None:
                gone = True
                break
        chk(gone, "服务进程已退出（页面关了就不会留后台进程）")
        chk(srv.returncode == 0, "退出码为 0", srv.returncode)

    finally:
        if srv.poll() is None:
            srv.terminate()
        log.close()

    print("\n" + "=" * 66)
    print("通过 %d 项，失败 %d 项" % (len(OK), len(BAD)))
    if BAD:
        print("失败项：")
        for b in BAD:
            print("  - %s" % b)
        rc = 1
    else:
        print("全部通过。")
    print("=" * 66)
    return rc


if __name__ == "__main__":
    sys.exit(main())
