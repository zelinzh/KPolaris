# 安装、输入和成像

KPolaris 计算相对论偏振辐射转移，输出 Stokes I/Q/U/V 图像、逐光线结果和物理诊断。
本手册的命令从源码目录运行。解析 RIAF 和磁化环模型不需要模拟数据；GRMHD 模型
需要使用者提供快照和物理归一化。

模板的分组注释和常用设置见[参数说明](parameters.md)，各可调参数的完整定义见
[英文参数参考](../parameters.md)。

## 首次使用

安装 Python 3.11+ 和下面列出的构建依赖后，在源码目录运行：

```bash
python3 -m pip install -r requirements.txt
python3 scripts/kpolaris.py doctor
python3 scripts/kpolaris.py quickstart
```

这会编译 RIAF，生成 256² 的 M87* 尺度示例，检查光线终止状态和 Stokes 有限性，并画出总强度图。
结果位于 `outputs/quickstart/`，包括 HDF5、解析后的参数、图片、日志和 `run.json`。
坐标用微角秒、线性色标用亮温表示；模型说明见 [M87* 示例](../first_image.md)。
`--resolution=64` 仅用于快速检查安装；科学计算仍应对目标观测量做收敛检查。
新一次运行请指定不同的 `--output-dir`。

`--backend=openmp` 和 `--backend=cuda` 分别选择多核和 GPU；CUDA 架构自动识别，
也可以显式指定 `--cuda-arch=AMPERE80`。`--dry-run --json` 输出命令计划但不执行。
详见 [部署和排错](../quickstart.md)、[绘图说明](../plotting.md)。

先安装或激活 CMake、C++ 编译器和 HDF5 C++，再执行首次示例。
没有 sudo 权限时，可以在用户目录使用已有的 Spack，并安装或加载
`hdf5+cxx~mpi`；`~cxx` 的 HDF5 和 Python 的 h5py 都不能替代 HDF5 C++ 开发库。
GPU 驱动可用并不表示 CUDA 编译器已经激活，还需要让 `nvcc` 出现在 PATH 中。
每次打开新终端时，应重新激活依赖环境和项目的 Python 虚拟环境。

## 构建

需要 CMake 3.25 或以上、C++20 编译器，以及带 C++ 接口的 HDF5。
默认由 CMake 下载 Kokkos 5.1.1；离线或已有依赖时指定
`KPOLARIS_FETCH_KOKKOS=OFF` 和 `Kokkos_DIR`。绘图和完整 Python 测试需要
NumPy、h5py、Matplotlib。Spack 环境文件可用来安装依赖，并非唯一安装方式。

**首次 CUDA 编译可能需要数十分钟，尤其是启用慢光和诊断的 GRMHD 程序。**
编译主要消耗主机 CPU 和内存，GPU 此时空闲通常是正常的。建议先编译一个模型及
与数据相符的坐标，不要为了首次测试选择所有模型或 `--coordinates=all`。
仅在需要回归测试或轨迹输出时添加 `--tests` 或 `--trace`。

例如，KHARMA 的 FMKS 数据只需：

```bash
python3 scripts/kpolaris.py build --models=kharma --coordinates=fmks \
  --build-dir=build/kharma
```

若只需单频快光 Stokes 图像，可进一步使用独立的精简构建：

```bash
python3 scripts/kpolaris.py build --backend=cuda --models=kharma --coordinates=fmks \
  --build-dir=build/kharma-fast-cuda --jobs=2 \
  --cmake-arg=-DKPOLARIS_ENABLE_ANALYSIS_MODE=OFF \
  --cmake-arg=-DKPOLARIS_ENABLE_SLOW_LIGHT=OFF \
  --cmake-arg=-DKPOLARIS_MAX_FREQUENCIES=1
```

这仍计算完整 I/Q/U/V，但不包含慢光、可选物理诊断／参数响应或共享多频内核；
每次运行使用一个 `freq`。需要这些功能时应重新启用对应编译选项。
缩短编译时间的幅度取决于模型和工具链，并非固定加速比。
普通 helper 新构建默认包含诊断、慢光及两个共享频率的容量。

`--jobs=2` 是默认编译并行度；单个 CUDA 编译进程可能占用数 GiB 主机内存，
内存较小时使用 `--jobs=1`，不要只按 CPU 核数提高并行度。降低图像分辨率、
放宽积分容差或在参数文件中关闭诊断不会减少编译内容。
保留构建目录并复用程序，换快照、相机、分辨率或编译能力范围内的频率无需重新编译；
更换编译器、GPU 架构或功能组合时使用独立构建目录。

直接使用 CMake 时，同样按需选择模型和功能。例如只构建 RIAF：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DKPOLARIS_IMAGE_MODELS=riaf -DKPOLARIS_BUILD_TESTS=OFF
cmake --build build --parallel 2
```

默认编译 RIAF。`KPOLARIS_IMAGE_MODELS` 可取 `riaf`、`binary_riaf`、`torus`、
`iharm`、`kharma`、`athenak`、`bhac` 或 `all`；多个模型用分号连接并加引号。
`KPOLARIS_BUILD_TRACE_TOOL=ON` 增加 trace 工具。
`KPOLARIS_GRMHD_COORDINATES=all` 编译所有受支持的 GRMHD 坐标后端。

OpenMP 构建增加 `KPOLARIS_ENABLE_OPENMP=ON`、`KPOLARIS_ENABLE_SERIAL=OFF`。
运行时同时设置 `OMP_NUM_THREADS` 与 `--kokkos-num-threads`；例如：

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads \
  ./build-openmp/kpolaris_model_image_riaf --kokkos-num-threads=8 \
  --parameter_file=params/riaf_recommended.par --output=riaf.h5
```

CUDA 构建使用 Kokkos 的 `nvcc_wrapper`，并指定实际显卡架构；helper 可以自动识别
单一受支持的显卡架构。A100 对应 `AMPERE80`；不同架构应单独构建。CPU 严格告警和 sanitizer 检查
不能代替目标 GPU 上的数值验证。

安装命令为 `cmake --install build --prefix /path/to/kpolaris`。
程序位于 `bin/`，参数位于 `share/kpolaris/params/`，分析脚本位于
`libexec/kpolaris/`。运行时仍需相容的 Kokkos、HDF5 和 CUDA 依赖环境。

## 第一张图像

```bash
./build/kpolaris_model_image_riaf \
  --parameter_file=params/demo_riaf.par --output=riaf.h5
python3 scripts/plot_kpolaris_pol.py riaf.h5 --layout=intensity \
  --intensity-unit=brightness-temperature --fov-units=muas --output=riaf.png
```

命令行覆盖参数文件。`parameter_output=auto` 保存完整有效配置到
`<output>.params`，可将它作为下一次的 `parameter_file` 使用。
非零退出、无效像素或未完成的光线应先诊断原因，再解释图像。

## 光线方向与步长

相机初始化和两遍积分始终保存指向未来的物理光子波矢量 `state.k=p`。
Pass A 用负仿射步长向过去追踪，Pass B 用正仿射步长向相机传播；端点处
不改变波矢量、事件时间或屏幕基。相机屏幕基 `e1/e2` 连续平行输运，
保持同一个偏振参考。慢光事件时间在 Pass A 中减小，在 Pass B 中增大。
程序的 `step`、`min_step`、`max_step` 均输入正数，Pass A 内部处理方向。

这与旧实现的“过去指向切向量配合正步长、端点恢复物理波矢量”在连续方程上
严格等价。直接调用 C++ 相机初始化接口时，应以负步长进行逆向追踪，
Pass B 前不再对 `k` 取负；命令行参数文件不需要修改。

## GRMHD 输入与单位

| 模型/程序后缀 | 输入参数 | 输入形式 |
| --- | --- | --- |
| `iharm` | `iharm_dump` | iHARM/HARM HDF5，含适用的 MKS/FMKS 元数据 |
| `kharma` | `kharma_dump` | KHARMA/Parthenon PHDF；也可使用 DDC 原生帧名称 |
| `athenak` | `athenak_dump` | AthenaK meshblock 二进制 |
| `bhac` | `bhac_dump` | BHAC AMR `.dat` |

```bash
./build-kharma/kpolaris_model_image_kharma \
  --parameter_file=params/kharma_recommended.par \
  --kharma_dump=/data/snapshot.phdf --nx=256 --ny=256 --output=kharma.h5
```

网格、坐标和电子温度模型必须与输入相符。KHARMA reader 支持均匀全局网格的完整、
不重叠逻辑 meshblock 分解；不支持的多层/不完整 AMR 会报错。AthenaK/BHAC
有各自的布局检查，不能仅通过改扩展名交换格式。

长度和时间使用模型的几何单位，频率以 Hz 指定。各 GRMHD 后端的
`*_M_unit` 与 `*_mbh_solar` 分别控制质量归一化与黑洞质量。
`*_trat_small`、`*_trat_large`、`*_beta_crit` 决定电子温度处方，
`*_sigma_cut` 等决定辐射掩膜。输入密度、质量单位、距离和频率共同决定观测通量，
不应只为了让两幅图同亮而逐张独立归一化。

参数模板包含明确的 `/path/to/snapshot` 占位，必须替换。模板中的质量、自旋、
视角及误差控制是示例设置；科学使用前核对自己的模型。KHARMA 模板可能启用
`kharma_resample=spherical_ks_precomputed`；需要 native MKS/FMKS，尤其 compact
DDC 时，应显式设置 `kharma_resample=none`。

## 相机、频率和输出

`camera=pinhole` 使用观察者处的针孔相机；`parallel_plane` 使用图像平面。
`radius`、`inclination_deg`、视场及像素数共同定义相机。固定相同视场再比较
不同分辨率。坐标参数控制计算后端，不能改变输入数据的物理坐标解释。

单频使用 `freq=230e9`，多频使用 `freq_list=86e9,230e9,345e9`。
编译期 `KPOLARIS_MAX_FREQUENCIES` 限定共享几何内核一次处理的频率数；更大的值
增加每条光线的状态。超出容量时按程序支持的分块方式执行。

`evpa_0=N` 为默认输出：约定图像上方为北、左方为东，EVPA 从北向东增加。
`evpa_0=W` 保留原有水平零点；两者的 Q/U 同时取负，I/V、像素和物理偏振线不变。
图像、通量及观测者基诊断同步转换。Trace 中间样本及其系数保留内部输运基，
只有最终观测 Stokes/EVPA 使用所选零点。绘图自动读取文件记录，不应手动重复翻号。
旧参数文件需增加 `evpa_0=W` 才能复现原来的 Q/U。

Stokes 图像和诊断使用统一的观测者基。与别的软件比较时，要同时对齐像素方向、
偏振基、EVPA 零点、V 符号、频率、单位、视场及发射/掩膜处方。
HDF5 同时包含不变量与有物理单位的输出，读取时依据
[字段及单位定义](../hdf5_schema.md)，不要仅凭数组名称猜测通量缩放。

## Trace 与二元模型

启用 trace 构建后，可用 `kpolaris_model_trace` 和模型参数输出沿光线的几何与辐射
状态。trace 的采样间隔控制存储分辨率，内核积分步长控制数值精度；两者作用不同。
全图轨迹存储量随像素数和保存样本数增长，通常先选择少量代表性光线分析。

`binary_riaf` 使用近似 superposed Kerr–Schild 时空和双源模型。
`params/binary_riaf.par` 给出无需外部轨迹的示例。外部轨迹必须满足
[原生轨迹 schema](../hdf5_schema.md#binary-trajectory-input-schema-kpolarisbinary_trajectoryv1)
定义的时间范围、单位及连续性；该近似不能被当作任意双黑洞时空的精确解。

`scripts/generate_binary_trajectory.py` 可使用使用者提供的 CBwaves 源生成轨迹，
完整参数见 `--help`。例如先执行 `--dry-run` 检查源版本和参数：

```bash
python3 scripts/generate_binary_trajectory.py \
  --cbwaves-source=/path/to/cbwaves-source --output=trajectory.h5 --dry-run
```

工具核对外部源和修正，并在临时目录中构建；第三方源不包含在 KPolaris 包内。
两源辐射模型使用其定义的并合前适用区间，不能把轨迹的延伸范围自动解释为辐射
模型的有效范围。

## 固定模型的连续成图与计时

快光工具支持 `repeat_images=N`。例如：

```sh
kpolaris_model_image_iharm --parameter_file=model.par \
  --repeat_images=9 --output=sequence.h5 --parameter_output=auto --timing=1
```

程序只加载一次流体模型，依次生成 `sequence_repeat0000.h5` 至
`sequence_repeat0008.h5` 及各自的参数文件。每张图都会重新积分光线、计算
偏振转移并保存结果；相机、频率与物理参数保持相同。这一选项用于测量模型
已在内存中时的成图成本，不代表逐张读取不同 GRMHD 快照的图像库吞吐量。

默认 `repeat_images=1` 保持通常的输出文件名。连续成图要求快光模式，以及
`parameter_output=auto` 或 `none`；已有的编号输出会在计算前触发报错。
慢光时间序列应使用慢光批处理接口。

日志中 `image_kernel_single` 是几何与辐射转移的积分时间，
`repeat_image_elapsed` 是模型加载后单张图的完整操作时间；外部计时还包含
进程启动、模型读取和初始化。比较性能时需明确采用哪一种计时口径。

`scripts/benchmark_image_scaling.py` 可自动进行分辨率、CPU/GPU 与内存测试。
它需要 Linux、NumPy 和 h5py；GPU 还需要 `nvidia-smi`。先运行 `--help`
查看参数，使用同一输入及积分设定，每个配置采用独立输出目录。CPU 默认先
使用每个物理核心的一个线程，再使用 SMT 线程，可用 `--cpu-set` 指定亲和性。
测试记录原始图像、有效参数、输入配置、计时、主机 RSS、交换内存及显存采样。
RSS 采用成图程序启动后的高水位计数，显存含 CUDA 上下文；两者分别报告。
每个分辨率先预热，再重复计时；结果对应热文件缓存下的成图成本。
该基准协议要求单张图只有一个频率和一个时刻，检测到多频率或多帧输出会报错，
以免只汇总第一幅图而给出错误的精度或计时解释。
