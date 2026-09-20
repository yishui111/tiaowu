# -*- coding: utf-8 -*-
"""把 UTF-8 的 .bat 转成 GBK + CRLF（cmd.exe 的硬要求）。
注意顺序：先 encode（失败会抛错、文件还没动），再落盘。"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
targets = [os.path.join(ROOT, "启动工作台.bat")]

for p in targets:
    raw = open(p, "rb").read()
    t = raw.decode("utf-8")
    t = t.replace("\r\n", "\n").replace("\n", "\r\n")   # 统一 CRLF
    out = t.encode("gbk")                               # ★ 先编码
    open(p, "wb").write(out)                            # ★ 再落盘
    b = open(p, "rb").read()
    print("OK %s  bytes=%d  crlf=%d  lf_only=%d"
          % (os.path.basename(p), len(b), b.count(b"\r\n"),
             b.count(b"\n") - b.count(b"\r\n")))
    try:
        b.decode("gbk")
        print("   gbk decode: OK")
    except Exception as e:
        print("   gbk decode FAIL:", e)
