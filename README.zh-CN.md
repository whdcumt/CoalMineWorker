<div align="center">
  <p>
    <a href="https://github.com/ultralytics/assets/releases/tag/v8.2.0" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es/) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [हिन्दी](https://docs.ultralytics.com/hi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
</div>
<br>

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) 是一款前沿、最先进（SOTA）的模型，基于先前 YOLO 版本的成功，引入了新功能和改进，进一步提升性能和灵活性。YOLOv8 设计快速、准确且易于使用，使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

我们希望这里的资源能帮助您充分利用 YOLOv8。请浏览 YOLOv8 <a href="https://docs.ultralytics.com/">文档</a> 了解详细信息，在 <a href="https://github.com/ultralytics/ultralytics/issues/new/choose">GitHub</a> 上提交问题以获得支持，并加入我们的 <a href="https://ultralytics.com/discord">Discord</a> 社区进行问题和讨论！

如需申请企业许可，请在 [Ultralytics Licensing](https://ultralytics.com/license) 处填写表格

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png" alt="YOLOv8 performance plots"></a>

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>

以下是提供的内容的中文翻译：

## <div align="center">文档</div>

请参阅下面的快速安装和使用示例，以及 [YOLOv8 文档](https://docs.ultralytics.com) 上有关训练、验证、预测和部署的完整文档。

<details open>
<summary>安装</summary>

使用Pip在一个[**Python>=3.8**](https://www.python.org/)环境中安装`ultralytics`包，此环境还需包含[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。这也会安装所有必要的[依赖项](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)。

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

如需使用包括[Conda](https://anaconda.org/conda-forge/ultralytics)，[Docker](https://hub.docker.com/r/ultralytics/ultralytics)和Git在内的其他安装方法，请参考[快速入门指南](https://docs.ultralytics.com/quickstart)。

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>Usage</summary>

### CLI

YOLOv8 可以在命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 可用于各种任务和模式，并接受其他参数，例如 `imgsz=640`。查看 YOLOv8 [CLI 文档](https://docs.ultralytics.com/usage/cli)以获取示例。

### Python

YOLOv8 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="coco8.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

查看 YOLOv8 [Python 文档](https://docs.ultralytics.com/usage/python)以获取更多示例。

</details>

### 笔记本

Ultralytics 提供了 YOLOv8 的交互式笔记本，涵盖训练、验证、跟踪等内容。每个笔记本都配有 [YouTube](https://youtube.com/ultralytics?sub_confirmation=1) 教程，使学习和实现高级 YOLOv8 功能变得简单。

| 文档                                                                                                              | 笔记本                                                                                                                                                                                                                       |                                                                                                     YouTube                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://docs.ultralytics.com/modes/">YOLOv8 训练、验证、预测和导出模式</a>                               | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>                  | <a href="https://youtu.be/j8uQc0qB91s"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube 视频"></center></a>  |
| <a href="https://docs.ultralytics.com/hub/quickstart/">Ultralytics HUB 快速开始</a>                               | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/hub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>                       | <a href="https://youtu.be/lveF9iCMIzc"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube 视频"></center></a>  |
| <a href="https://docs.ultralytics.com/modes/track/">YOLOv8 视频中的多对象跟踪</a>                                 | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_tracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>           | <a href="https://youtu.be/hHyHmOtmEgs"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube 视频"></center></a>  |
| <a href="https://docs.ultralytics.com/guides/object-counting/">YOLOv8 视频中的对象计数</a>                        | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_counting.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>           | <a href="https://youtu.be/Ag2e-5_NpS0"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube 视频"></center></a>  |
| <a href="https://docs.ultralytics.com/guides/heatmaps/">YOLOv8 视频中的热图</a>                                   | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/heatmaps.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>                  | <a href="https://youtu.be/4ezde5-nZZw"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube 视频"></center></a>  |
| <a href="https://docs.ultralytics.com/datasets/explorer/">Ultralytics 数据集浏览器，集成 SQL 和 OpenAI 🚀 New</a> | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> | <a href="https://youtu.be/3VryynorQeo"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |

## <div align="center">模型</div>

在[COCO](https://docs.ultralytics.com/datasets/detect/coco)数据集上预训练的YOLOv8 [检测](https://docs.ultralytics.com/tasks/detect)，[分割](https://docs.ultralytics.com/tasks/segment)和[姿态](https://docs.ultralytics.com/tasks/pose)模型可以在这里找到，以及在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet)数据集上预训练的YOLOv8 [分类](https://docs.ultralytics.com/tasks/classify)模型。所有的检测，分割和姿态模型都支持[追踪](https://docs.ultralytics.com/modes/track)模式。

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO supported tasks">

所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

<details open><summary>检测 (COCO)</summary>

查看[检测文档](https://docs.ultralytics.com/tasks/detect/)以获取这些在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的模型的使用示例，其中包括80个预训练类别。

| 模型                                                                                 | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | ------------------- | -------------------- | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 640                 | 37.3                 | 80.4                          | 0.99                               | 3.2              | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | 640                 | 44.9                 | 128.4                         | 1.20                               | 11.2             | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 640                 | 50.2                 | 234.7                         | 1.83                               | 25.9             | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | 640                 | 52.9                 | 375.2                         | 2.39                               | 43.7             | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | 640                 | 53.9                 | 479.1                         | 3.53                               | 68.2             | 257.8