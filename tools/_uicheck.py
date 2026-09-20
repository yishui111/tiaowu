# -*- coding: utf-8 -*-
"""页面自检：起服务 -> 无头 Chrome 抓每个标签页的真实 DOM -> 检查 JS 是否把数据填进去了。

只看 DOM，不看截图（本会话读不了图）。能证明：
    - index.html 没有 JS 语法错误（否则下拉框不会被填充）
    - 前端确实从 /api/state 拿到了数据并渲染
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 18067
BASE = "http://127.0.0.1:%d" % PORT
def _find_browser():
    exe = os.environ.get("CHROME") or shutil.which("chrome") or shutil.which("msedge")
    if exe:
        return exe
    pw = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright")
    hits = sorted(glob.glob(os.path.join(pw, "chromium-*", "chrome-win*", "chrome.exe")))
    return hits[-1] if hits else "chrome"

CHROME = _find_browser()
PROFILE = os.path.join(ROOT, "tools", "_chrome_profile")

OK, BAD = [], []


def chk(cond, label, detail=""):
    (OK if cond else BAD).append(label)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   <- " + str(detail)[:200]) if detail else ""))


def wait_port(t=30):
    """★ 本机对「已关闭的回环端口」是超时而不是拒绝连接（实测每次 2 秒），
    所以要用墙钟时间封顶，别只数次数，否则失败路径要等 4.5 分钟才报错。"""
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            urllib.request.urlopen(BASE + "/api/state", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def dump(hash_, budget=9000):
    url = BASE + "/" + hash_
    p = subprocess.run(
        [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--no-proxy-server", "--no-first-run", "--disable-extensions",
         "--user-data-dir=" + PROFILE,
         "--virtual-time-budget=%d" % budget, "--dump-dom", url],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
    return p.stdout.decode("utf-8", "replace")


def main():
    py = sys.executable
    log = open(os.path.join(ROOT, "tools", "_uicheck_server.log"), "wb")
    srv = subprocess.Popen([py, os.path.join(ROOT, "tools", "serve.py"),
                            "--port", str(PORT), "--no-browser"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        print("=" * 66)
        print("页面自检（无头 Chrome 抓 DOM）")
        print("=" * 66)
        if not chk(wait_port(30), "服务起来"):
            return 1
        chk(os.path.exists(CHROME), "找到无头 Chrome", CHROME)

        print("\n[概览页]")
        d = dump("#overview")
        chk(len(d) > 2000, "DOM 正常返回", "%d 字节" % len(d))
        chk("这三个东西是什么" in d, "概览文案在")
        chk("Bailando (CVPR2022)" in d, "标题栏在")

        print("\n[LODGE 页]")
        d = dump("#lodge")
        # 下拉框是 JS 从 /api/state 填的 —— 填上了就说明 JS 没报错、接口通
        chk('value="Hiphop"' in d and 'value="Breaking"' in d,
            "舞种下拉框被 JS 填充", re.findall(r'<option[^>]*>([^<]{0,20})', d)[:6])
        chk("临时“顶替”" in d or "顶替" in d, "LODGE 的数据顶替说明在页面上")

        print("\n[Bailando 页]")
        d = dump("#bailando")
        opts = re.findall(r'<option value="([^"]+)"', d)
        chk(len(opts) == 40, "40 条序列被 JS 填进下拉框", "%d 条" % len(opts))
        chk(any("已有视频" in o for o in re.findall(r"<option[^>]*>([^<]+)</option>", d)),
            "已渲染过的序列被标记")

        print("\n[成品库]")
        d = dump("#library")
        n = d.count('class="vid"')
        chk(n >= 42, "成品库渲染出视频卡片", "%d 个" % n)
        chk("/media?path=" in d, "视频用的是带 Range 的媒体接口")
        chk("preload=\"none\"" in d, "成品库用 preload=none 避免卡顿")

        print("\n[EDGE 页]")
        d = dump("#edge")
        chk("Jukebox" in d and "随机" in d, "EDGE 的机制说明在页面上")
        chk('data-drop="edge"' in d, "拖放上传区在")

    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(
                BASE + "/api/quit", data=b"{}", method="POST"), timeout=5)
        except Exception:
            pass
        time.sleep(1)
        if srv.poll() is None:
            srv.terminate()
        log.close()

    print("\n" + "=" * 66)
    print("通过 %d 项，失败 %d 项" % (len(OK), len(BAD)))
    for b in BAD:
        print("  - %s" % b)
    print("=" * 66)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
