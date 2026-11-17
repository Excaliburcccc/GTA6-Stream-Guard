# GTA6 Stream Guard

Windows 本地直播画面助手。支持采集卡与本地视频预览，对识别到的警车、裸露、内衣和大片红色血迹加局部马赛克。本地视频的原声按实际播放时钟与画面同步。F8 打开全画面 95% 不透明度的马赛克，F9 恢复。

## 安装和启动

1. 准备 Windows 64 位 Python 3.11、3.12 或 3.13（含 Tcl/Tk），将 `ffmpeg.exe` 和 `ffprobe.exe` 所在目录加入 PATH。
2. 解压或克隆本仓库，运行 `setup.cmd`。安装脚本会安装 Python 依赖，并下载和校验下面的模型。
3. 把本地 MP4 或 WebM 拖到 `start.cmd`，按 F9 播放。双击 `start.cmd` 可打开默认采集设备。

OBS 的画面源选择 `GTA6 Stream Guard - PROGRAM`，启动虚拟摄像机。抖音直播伴侣的画面来源选 `OBS Virtual Camera`，本地视频原声通过系统声音采集接入。完整步骤、热键和命令见 [使用说明](使用说明.md)。

## 模型与下载

| 模型 | 用途 | 下载方式 |
| --- | --- | --- |
| **Ultralytics YOLOv5n v7.0 ONNX** | 检测车辆与人物候选区域 | `download_police_model.py` 下载并校验 SHA-256，保存到 `models/yolov5n-v7.onnx`。[上游发布页](https://github.com/ultralytics/yolov5/releases/tag/v7.0) |
| **OpenAI CLIP ViT-B/32**，`openai/clip-vit-base-patch32`，固定修订 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | 核验警车、衣物及人体区域，辅助场景判断 | `download_scene_model.py` 下载所需文件并逐个校验 SHA-256，保存到 `models/clip-vit-base-patch32/`，合计约 610 MB。[固定版本文件](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268) |
| **NudeNet 3.4.2（`320n.onnx`）** | 提出人体敏感部位候选 | 由 `requirements.txt` 安装 `nudenet==3.4.2`，模型随 Python 包安装。[PyPI 固定版本](https://pypi.org/project/nudenet/3.4.2/) |

已运行过安装脚本的环境，可以在仓库根目录单独补装或校验模型。

```powershell
.\.venv\Scripts\python.exe -m pip install "nudenet==3.4.2"
.\.venv\Scripts\python.exe download_police_model.py
.\.venv\Scripts\python.exe download_scene_model.py
```

模型来源与许可证说明见 [third-party](third-party/README.md)。
