# KPolaris

KPolaris 是面向 CPU 和 GPU 的偏振广义相对论辐射转移程序，可从解析
等离子体模型或 GRMHD 数据生成 Stokes I/Q/U/V 图像，并分析发射位置、传播效应
及物理参数对观测量的影响。

[English](README.md) · [中文手册](docs/zh/README.md) · [Wiki](https://github.com/zelinzh/KPolaris/wiki/ZH-Home) · [画图说明（英文）](docs/plotting.md)

<img src="assets/riaf-preview.png" alt="M87* RIAF 的 Stokes I、Q、U、V" width="820">

默认示例是采用 M87* 质量和距离的 230 GHz 解析吸积流图像。亮环、中央暗区及
亮度不对称均由辐射转移计算产生。横排展示 I/Q/U/V；各面板使用线性色标，
Q/U/V 保留正负号，各自的色标范围见图中刻度。

## 第一次使用

先运行不需要任何外部模拟数据的 RIAF 示例。需要 Python 3.11+、CMake 3.25+、
支持 C++20 的编译器和 HDF5 C++。先安装或激活编译依赖；Ubuntu 24.04 可使用：

```bash
sudo apt install build-essential cmake ninja-build libhdf5-dev python3-venv
```

没有管理员权限时，可按[安装指南](docs/quickstart.md)使用用户目录中的 Spack
环境。仅安装 Python 的 h5py 不会提供编译所需的 HDF5 C++ 开发库。
随后在源码目录运行：

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/kpolaris.py doctor
python3 scripts/kpolaris.py quickstart --output-dir=outputs/first-image
```

工具按需下载 Kokkos，构建 RIAF 程序，生成 256² 图像，检查光线终止状态及 Stokes
有限性，再画出图像。输出目录包含 HDF5、有效参数、PNG、日志和 `run.json`。
只有完成全部检查才报告 `ok: true`。新一次运行请使用新的输出目录。

首次运行采用 256² 分辨率，`image_freq0.png` 展示总强度，坐标单位为微角秒。
若只需快速检查安装，可加 `--resolution=64`。质量取 65 亿太阳质量、距离
16.8 Mpc，视场约 76 微角秒；等离子体和自旋参数用于演示，并非观测拟合。
科学分析仍需对关注的观测量检查收敛。模型、参数及物理含义见
[示例说明](docs/zh/first_image.md)。
详细依赖安装及离线构建见[安装指南](docs/quickstart.md)。

## 继续尝试

```bash
python3 scripts/kpolaris.py demo --example=multifrequency --output-dir=outputs/frequencies
python3 scripts/kpolaris.py demo --example=diagnostics --output-dir=outputs/diagnostics
python3 scripts/kpolaris.py demo --example=response --output-dir=outputs/responses
```

三个示例分别展示多频率图像、发射形成位置与传播深度、以及密度／温度／磁场
标度对观测的响应。它们复用第一次构建的程序。
如果先前使用 CUDA 或 OpenMP quickstart，请在每条 `demo` 命令中也指定
相同的 `--backend=cuda` 或 `--backend=openmp`，以复用对应构建。

CPU 多线程使用 `--backend=openmp --threads=4`。GPU 使用 `--backend=cuda`，
工具会自动识别单一受支持的显卡架构。也可用 `--cuda-arch` 明确指定实际架构，
如 A100 的 `AMPERE80`。工具不会把失败的 GPU 构建偷偷改成 CPU。

**首次编译请预留时间。** CUDA 的 GRMHD 构建可能需要数十分钟，具体取决于
主机 CPU、编译器及启用的功能。建议只编译所需的模型、坐标和计算后端，
保留构建目录，后续成图复用程序。减少图像分辨率不会减少编译工作量。
仅需单频快光时，可按[按需构建说明](docs/zh/user_guide.md#构建)关闭不使用的功能。

```bash
python3 scripts/kpolaris.py quickstart --backend=cuda --output-dir=outputs/cuda
python3 scripts/kpolaris.py build --models=kharma --coordinates=fmks --build-dir=build/kharma
```

GRMHD 支持 iHARM/HARM、KHARMA、AthenaK、BHAC。请自行提供数据，
核对质量单位、电子温度模型及坐标，不能直接把参数模板当作任意数据的物理设定。
`params/` 已按功能分组注释；各参数的含义、单位和适用条件见
[参数说明](docs/zh/parameters.md)及[完整参数参考（英文）](docs/parameters.md)。
标准构建使用热电子。非热选项的系数覆盖范围和验证程度不同，使用前请阅读
[电子分布说明](docs/electron_distributions.md)。
慢光时间序列与批处理见[慢光说明](docs/zh/slow_light.md)。DDC 解码器单独安装。

## 检查与画图

```bash
python3 scripts/kpolaris.py inspect outputs/first-image/image.h5 --json
python3 scripts/plot_kpolaris_pol.py outputs/first-image/image.h5 --layout=stokes --output=outputs/first-image/stokes.pdf
```

`--frame`、`--freq-index` 指定所画的图像。默认 `evpa_0=N`：图像上方为北、
左方为东，EVPA 从北向东增加。`evpa_0=W` 保留原有水平零点；两者 Q/U 同时
取负，I/V 和物理偏振线方向不变。绘图自动读取零点。模型在真实天球上的
位置角仍需另行指定。灰色区域表示信号过弱或像素无效，详见[绘图说明](docs/plotting.md)。
相机方位、图像轴向、Stokes 符号及跨代码换基见[方位约定](docs/zh/polarization_conventions.md)。
缺少距离或物理尺度时，工具不会猜测通量。

## 通过 Agent 部署和运行

KPolaris 支持由 agent 协助安装并生成第一张图像。将下面这段话发送给能够
访问本机终端的编程 agent：

```text
请在这台机器上部署 https://github.com/zelinzh/KPolaris，按照仓库中的
AGENTS.md 和 skills/kpolaris/SKILL.md 检查并配置依赖。优先使用可用且兼容的
GPU 后端，否则使用 CPU 后端进行构建。运行默认的 256×256 M87* 示例，
检查计算是否成功，并把 Stokes I/Q/U/V 图像展示给我。请说明实际使用的
计算后端和结果保存位置。
```

Agent 可以依据仓库内的[操作指南](skills/kpolaris/SKILL.md)完成环境检查、
依赖配置、编译、示例运行和结果验证。首次示例不需要下载模拟数据。
你也可以在指令中指定计算后端或安装目录；处理自己的 GRMHD 数据时，
还应提供[输入指南](docs/zh/user_guide.md)要求的数据格式、路径及物理参数。

## 开发与文档

使用方法和物理定义见[中文手册](docs/zh/README.md)。
手册与代码一起版本管理，也可[生成为 GitHub Wiki](docs/wiki.md)。
开发与贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 联系方式

维护者：Zelin Zhang — [zhangzelin1@nbu.edu.cn](mailto:zhangzelin1@nbu.edu.cn)。

## 许可证

[BSD-3-Clause](LICENSE)。依赖库和模拟数据保留各自的许可证。

## 引用

如果在研究中使用 KPolaris，请引用本软件：

v0.1.0 版本 DOI：[10.5281/zenodo.22879728](https://doi.org/10.5281/zenodo.22879728)。
[所有版本的总 DOI](https://doi.org/10.5281/zenodo.22879727)用于指代持续维护的软件；复现本次发布时请使用版本 DOI。

```bibtex
@software{zhang_kpolaris,
  author  = {Zhang, Zelin and Chen, Bin},
  title   = {{KPolaris}: {GPU}-accelerated Polarized Radiative Transfer in General Relativity},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.22879728},
  url     = {https://github.com/zelinzh/KPolaris}
}
```

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。相关论文发表后，请同时引用论文。
