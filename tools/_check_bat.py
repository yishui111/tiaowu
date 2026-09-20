# -*- coding: utf-8 -*-
"""静态检查 .bat：括号配平、goto 目标是否存在。"""
import re
import sys

p = sys.argv[1]
t = open(p, "rb").read().decode("gbk")
lines = t.split("\r\n")
depth = 0
labels = set()
gotos = []
bad = []
for i, l in enumerate(lines, 1):
    s = l.strip()
    if s.startswith("rem ") or s.startswith("::"):
        continue
    if re.match(r"^:[A-Za-z_]", s):
        labels.add(s[1:].strip())
    for m in re.finditer(r"goto\s+:?([A-Za-z_]\w*)", s):
        gotos.append((i, m.group(1)))
    if s.lower().startswith(("set /p", "echo")):
        continue
    depth += s.count("(") - s.count(")")
    if depth < 0:
        bad.append((i, s))

print("标签:", sorted(labels))
print("goto 指向不存在的标签:", [g for _, g in gotos if g not in labels] or "无")
print("最终括号深度:", depth, "（应为 0）")
print("提前闭合的行:", bad or "无")

# 块内 %VAR% 提前展开风险
inblock = False
for i, l in enumerate(lines, 1):
    if l.strip().startswith("if ") and l.rstrip().endswith("("):
        inblock = True
        continue
    if inblock and l.strip() == ")":
        inblock = False
        continue
    if inblock and re.search(r"%[A-Za-z_]\w*%", l):
        print("  第%d行 在 if 块内用了 %%VAR%%: %s" % (i, l.strip()))

# 检查非 GBK 字符
try:
    t.encode("gbk")
    print("GBK 可编码: OK")
except UnicodeEncodeError as e:
    print("GBK 不可编码:", e)
