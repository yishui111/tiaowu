# 音乐驱动 3D 舞蹈生成工作台

给一段音乐，自动生成人物跳舞的视频。三个学术界开源模型（音乐 → 舞蹈动作 → SMPL 渲染）
装在同一个目录里，我用一个**纯标准库写的网页工作台**（`tools/serve.py` + `tools/index.html`）
把它们统一管起来：浏览器里选音乐、选模型、点开始，视频出来直接看，不用背任何命令行。

| 目录/文件 | 是什么 | 上游项目 |
|---|---|---|
| `Bailando/` | Bailando 2（CVPR2022），音乐 → 舞蹈，权重最全，实测最稳 | [andyt-ai/Bailando](https://github.com/andyt-ai/Bailando) |
| `EDGE/` | EDGE（CVPR2023），Jukebox 特征 + 扩散模型，同一首歌每次生成的舞都不一样 | [Stanford-TML/EDGE](https://github.com/Stanford-TML/EDGE) |
| `LODGE/` | LODGE（CVPR2024），全局+局部两级生成，FineDance 数据集。我加的 `infer_lodge.py` 推理入口随仓库走 | [meiyuma/LODGE](https://github.com/meiyuma/LODGE) |
| `jukebox/` `jukemirlib/` | OpenAI Jukebox（EDGE 的音乐特征提取依赖） | [rodrigo-castellon/jukebox](https://github.com/rodrigo-castellon/jukebox) |
| `tools/` `启动工作台.bat` | **我写的工作台**（本仓库的核心） | — |
| `_tools_*.py` | 我写的补丁/渲染/可视化工具（见「四、我的工具脚本」） | — |

**这个仓库只提交我自己写的部分**。三个引擎的源码、venv（各一套 Python 3.10 + GPU torch）、
`wheels/` 里 4.8 GB 的 CUDA wheel 缓存、`bin/ffmpeg.exe`、数据集和模型权重都不进仓库。
换电脑时：clone 本仓库拿到我的工作台和工具 → 按本文档把引擎装回来 → 双击 `启动工作台.bat`。

本机环境：Windows 10 / RTX 4080 16GB / Python 3.10.11。

---

## 一、仓库里有什么

```
启动工作台.bat             双击启动 → 自动开浏览器 http://127.0.0.1:18066/
tools/
├── serve.py               工作台后端（纯 Python 标准库，不装任何包）
├── index.html             工作台页面（无外部 CDN，离线可用）
├── _assets/               页面用的示例音乐（EDGE/LODGE 演示曲）
├── _selftest.py           自测：三个引擎全链路
├── _uicheck.py            自测：无头浏览器检查页面真的渲染出数据
├── _check_bat.py          自测：启动脚本内容检查
└── _to_gbk.py             工具：把 UTF-8 的 .bat 转成 GBK+CRLF（cmd 的硬要求）
_tools_patch.py            一键给三个引擎打兼容补丁（torch 2.x / numpy 2.x / 路径）
_tools_render_from_pkl.py  把 Bailando 输出的动作 pkl 渲染成 mp4
_tools_lodge_viz.py        把 LODGE 输出渲染成 mp4
make_demo_music.py         生成演示音乐
folder.html                LODGE FineDance 特征文件的 Google Drive 下载清单（数据来源记录）
```

## 二、从零部署（换电脑照着做）

### 0. 基础环境

- Windows + NVIDIA 显卡（这三个模型 CPU 跑不动，必须 GPU torch）
- Python 3.10（三个引擎各建自己的 `venv\`，基座同一个 Python 3.10.11 即可）
- ffmpeg：下载后放到 `bin\ffmpeg.exe`（工作台优先用它），或加入 PATH
- GitHub 走 `https://ghfast.top/` 前缀代理；pip 走国内镜像

### 1. 拉引擎源码 + 建 venv

```bat
git clone https://github.com/andyt-ai/Bailando.git Bailando
git clone https://github.com/Stanford-TML/EDGE.git EDGE
git clone https://github.com/meiyuma/LODGE.git LODGE
git clone https://github.com/rodrigo-castellon/jukebox.git jukebox
git clone https://github.com/rodrigo-castellon/jukemirlib.git jukemirlib

cd Bailando && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ..\EDGE    && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ..\LODGE   && python -m venv venv && venv\Scripts\pip install -r requirements.txt
```

### 2. GPU torch（版本不能错）

| 引擎 | torch 版本 | 原因 |
|---|---|---|
| LODGE | `torch 2.0.1+cu118` | 依赖 pytorch_lightning 1.9.5，官方支持上限约 torch 2.0 |
| EDGE / Bailando | `torch 2.5.1+cu121` | 实测可用 |

三个 venv 里如果装成 CPU 版 torch（`torch.version.cuda is None`），推理会慢到没法用——
用 `python -c "import torch; print(torch.version.cuda)"` 自查。

### 3. 引擎依赖的坑（都是实测踩过的）

- **EDGE 的 jukebox 千万别从 PyPI 装**：PyPI 上的 `jukebox 0.4.1` 是同名的 Django 音乐播放器，
  装了 `import jukemirlib` 必挂。必须装 GitHub 版：
  `pip install git+https://github.com/rodrigo-castellon/jukebox.git`，
  另外还需要 `p_tqdm`。OpenAI Jukebox 官方要 Python 3.7 + torch 1.7，是三个引擎里最难装的。
- **LODGE**：`pip install "setuptools<81"`（新版 setuptools 移除了 pkg_resources，
  lightning_fabric 1.9.5 会 import 崩）；pytorch_lightning 1.9.5 + torchmetrics + omegaconf + einops + librosa。
- **numpy 全部 `<2`**：2022~2024 的老仓库顶不住 `np.float_` 之类的移除。
- **pytorch3d 不用编译**：Windows 上编译 pytorch3d 是出了名的坑，我写了纯 PyTorch 复刻版
  （axis_angle_to_quaternion、RotateAxisAngle 等本仓库用到的变换），放在
  `EDGE/pytorch3d/` 和 `LODGE/pytorch3d/`——**这两个目录随本仓库走**，clone 下来就有，
  放回引擎目录对应位置即可。

### 4. 数据与权重

- **Bailando**：4 个权重（actor_critic/epoch_10.pt、cc_motion_gpt/epoch_400.pt、
  sep_vqvae/epoch_500.pt、sep_vqvae_root/epoch_500.pt）+ AIST++ 数据（约 9 GB），
  按上游 README 的 Google Drive 链接下载解压。
- **LODGE**：两个 checkpoint（exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt、
  exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt）；
  FineDance 特征数据的下载清单我存在 `folder.html`（mofea319.rar / music_npynew.rar /
  Normalizer.pth / smplx_neu_J_1.npy）。
- **EDGE**：checkpoint.pt（约 1.1 GB），按上游 README 下载；
  Jukebox 特征模型首次跑会自动下载（很大，需要耐心/代理）。

### 5. 打补丁 + 启动

```bat
python _tools_patch.py        :: 给三个引擎打兼容补丁（幂等，可重复跑）
启动工作台.bat                 :: 双击，浏览器自动打开 http://127.0.0.1:18066/
```

验收：`python tools\_selftest.py`（三引擎全链路）；`python tools\_uicheck.py`（页面渲染检查）。

---

## 三、工作台是怎么工作的

- `serve.py` 只用 Python 标准库，本身不加载任何模型：选音乐上传 → 按模型把输入摆到
  引擎期待的位置（LODGE 需要 063 编号的 wav+json 成对放入 data/finedance/）→
  用各引擎 `venv\Scripts\python.exe` 起子进程跑推理 → 轮询日志 → 视频出来在页面上直接播。
- 关掉浏览器页面工作台会自动退出（页面心跳机制），不用去杀进程。
- EDGE 从音乐里随机挑段生成，同一首歌每次结果都不同；音乐至少 8 秒。

## 四、我的工具脚本

| 脚本 | 用途 |
|---|---|
| `_tools_patch.py` | 三个引擎跑在新版 torch/numpy 上需要的最小补丁，一键打、幂等 |
| `_tools_render_from_pkl.py` | Bailando eval 产出的动作 pkl → mp4（不需要进 Blender） |
| `_tools_lodge_viz.py` | LODGE 输出 → mp4 |
| `make_demo_music.py` | 合成测试用音乐片段 |
| `tools/_to_gbk.py` | `.bat` 必须 GBK+CRLF 才能被 cmd 正确解析，此脚本负责转换 |

产物目录（`EDGE/_workbench/`、`LODGE/_workbench/`、`_workbench_out/`、各 `_*_video/`）
都是运行时生成的，不入库。
