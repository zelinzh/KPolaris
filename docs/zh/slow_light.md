# 慢光、批处理和 DDC

## 慢光时间

`slow_light=1` 沿光线采样随时间变化的流体。相机处设 `x[0]=0`，采样时间为

```text
t_fluid = slow_light_observation_time + x[0]
```

`slow_light_observation_time` 是光子到达相机的坐标时间。
默认 `slow_light_interpolation=fluid` 先对空间采样后的流体量做时间插值，
从速度与磁场原始变量重建四矢量，再应用磁化度截断并计算系数。
可选的 `slow_light_interpolation=coefficients` 在相邻快照分别计算辐射转移系数，
再按采样时间线性插值系数；复现旧版慢光结果时应显式设置此项。
快光不使用此选项。流体插值保留读取器的空间重建约定：已有的 `ne`、`Theta_e`、`B`、`sigma`、`beta`
派生标量也先插值，不等于从插值后的密度与内能重新推导全部标量。
两种方式都支持慢光图像、批处理与单频 trace；HDF5 属性及有效参数文件记录所选方式。
仅缓存四矢量的预重采样输入不支持 `fluid`；原生网格与直接 meshblock 路径支持。
两种次序在有限时间间隔下不等价，应通过加密快照检查图像、总流量与净偏振的稳定性。
时间表必须严格递增，覆盖所有辐射采样事件，并与快照自身的时间一致。

用 `slow_light_time_probe=1` 可先探测所需的时间支持。设置
`slow_light_dump_list` 与 `slow_light_time_list` 时使用相互对应的快照和时间列表。
观测时刻不是默认等于第一个快照时间；不能用重复的 DDC 帧名称伪造多个物理时间。

快光/慢光比较要定义同一个参考事件。可以明确采用 `t_ref=t_obs-camera_radius`
去掉远处相机的共同传播延迟，但它不是精确 Kerr 时延校正。采用其他对齐也应保存
明确的公式；只比较同一个数字时间不能自动隔离慢光效应。

## 显存驻留与输入预取

```ini
slow_light=1
slow_light_step_mode=decoupled
slow_light_windows_per_block=16
slow_light_pipeline=1
slow_light_prefetch=1
slow_light_prefetch_snapshots=32
```

iHARM、KHARMA、AthenaK、BHAC 的单频、多频、诊断及观测时刻批处理
默认采用 `decoupled + fluid`。
`slow_light_step_mode` 是策略参数；`decoupled` 是取值，`continuous` 是兼容别名，
并非独立开关。读取缓存边界不再截断积分步：缺少采样状态时暂停并恢复提议前步长，
等连续数据块到达后重试，不提交任何部分积分。保留四帧历史状态供缩步重试，
极端情况可整块回退。时间覆盖确实不足则报错，不用端点状态掩盖问题。

`block` 只在缓存块末截步，`snapshot` 在每个快照边界截步，均保留供旧结果复现。
这些成图组合不再自动退回 `snapshot`。时间探测只追踪几何，不使用快照边界步进。
trace 仍逐对推进；快光不受影响。输出记录实际模式，别名统一记录为 `decoupled`。

| 设置 | 含义 |
| --- | --- |
| `slow_light_windows_per_block=N` | 每轮最多 N 个新时间窗口；默认 1，可增大以减少调度开销 |
| `slow_light_prefetch_snapshots=H` | 独立主机预取队列，0 自动采用 N，另有一个进行中的读取 |
| `slow_light_pipeline` | KHARMA 上传/积分流水线开关 |
| `slow_light_snapshot_cache_gib` | KHARMA 设备快照数组预算；0 禁用自动选择 |

设单帧物化设备数据为 D，解耦历史缓存 K=4（旧模式 K=0）。当前块最多占
`(N+1+K)D`，流水线环形池占 `(2N+1+K)D`，还须保留初始模型和光线状态。
自动预算无流水线取 `N=floor(budget/D)-2-K`，有流水线取
`N=floor((floor(budget/D)-3-K)/2)`；解耦模式至少需七/九份快照有效载荷。
显式 N 与预算同时设置时取较小值。H 占主机 RAM，不是显存，也不控制外部 DDC 缓存。
压缩文件大小不等于 D；设备运行时和图像数组不在快照预算内，须另留空间。

缓存大小不改变解耦模式的积分分段，但不代表辐射积分已收敛。仍须分别细化
`adaptive_tolerance`、`max_radiation_step` 并检查 IQUV。旧 `block` 模式改变 N
会改变数值分段，不能将其差异误认为压缩失真。

## KHARMA 同相机图像批次

`slow_light_batch_jobs` 支持全部上述 GRMHD 后端，可同时启用多频和诊断；
KHARMA 支持 PHDF 或 native DDC。每个观测时刻写一个文件，频率分别位于 `/frame_0/freq_i`。
每个时刻独立演化 Stokes，共享相机几何追踪和驻留流体窗口。相机、度规、频率、
单位、电子模型、积分参数和诊断设置应一致。

每行格式为观测时间、首快照索引、末快照索引、输出路径；索引从零起且两端包含：

```text
# observer_time first_index last_index output
1200.0 0 100 /output/frame_000.h5
1200.5 5 105 /output/frame_001.h5
```

这只是文件格式示例；实际时间支持应由探测结果与快照表决定。全局列表先裁到
所有 job 的时间支持并集，最早输入从索引 0 开始。输出路径可加引号；已有输出
或重复输出目的地会被拒绝。

```bash
./build-kharma/kpolaris_model_image_kharma \
  --parameter_file=physics.par --parameter_file=time_support.par \
  --slow_light=1 --slow_light_batch_jobs=jobs.txt \
  --slow_light_windows_per_block=16 --slow_light_prefetch_snapshots=32
```

每个时刻保存单独 HDF5 文件及对应的有效参数文件，记录批次索引和实际窗口设置。
`block` 模式的单帧参数文件保留共享时间表前缀，以复现原批次的块对齐。
同时保存全局 jobs 文件和时间表。可同时设置 `freq_list`，各任务使用相同频率列表；
该接口仍要求相同相机、度规和流体处方。时间探测须单独运行。

## DDC 普通输入

DDC 是单独维护的压缩/解码软件，KPolaris 负责其 KHARMA 客户端、设备驻留和 GRRT。
先启动相容的 DDC native 服务，然后设置：

```ini
kharma_ddc_native=1
kharma_ddc_socket=/tmp/kpolaris-ddc.sock
kharma_ddc_manifest=/data/sequence_manifest.json
kharma_ddc_timeout_seconds=7200
kharma_dump=ddc_frame_00001.phdf
```

原生名称中的整数是 manifest sequence 编号，不是物理时间。普通文件名仍走
PHDF reader；无效协议、时间不符或非有限输入会报错。超时值限制每次阻塞通信，
不是整批作业总时间。

源码包也提供 `scripts/serve_ddc_parallel.py` 兼容服务入口，需额外安装匹配的 DDC
项目并指定 `--codec-project`。它不提供完整 DDC 压缩器，且不替代外部 compact 服务：

```bash
python3 scripts/serve_ddc_parallel.py --codec-project=/path/to/ddc-project \
  --manifest=/data/sequence_manifest.json --socket=/tmp/kpolaris-ddc.sock \
  --cache-dir=/local/ddc-cache
```

## 可选 compact CUDA 输入

默认关闭；需要 CUDA Kokkos 和相容 DDC 头文件：

```text
-DKPOLARIS_ENABLE_DDC_COMPACT_CUDA=ON
-DKPOLARIS_DDC_CODEC_INCLUDE_DIR=/path/to/ddc/src/dense_dump_codec/include
```

头文件目录应提供 `ddc_gpu_reconstruct.cuh`、`ddc_compact_receive.hpp` 和
`ddc_compact_types.hpp`。选择同时支持这些接口和相应包格式的 DDC 版本；运行记录
应固定 DDC 的确切版本。默认核心构建不需要这些依赖。

DDC 服务使用 native compact 传输与匹配的空间 working cache，例如服务参数
`--transport=native --native-transfer-mode=compact --native-radius-max=auto`
和 `--native-working-cache=/local/working-cache`。
客户端进程同时设置 `KPOLARIS_DDC_COMPACT=1`，并使用上述 native 输入参数。
此路径需要保留原始 anchors、有效 working cache，以及未重采样的 native MKS/FMKS。
普通 KHARMA 模板须覆盖 `kharma_resample=none`。

CPU 解压工作块，传输残差、尺度、例外值及 anchor；GPU 重建 float32 primitives，
随后进入常规单位换算和派生量计算。这不等于在 GPU 上解压 bzip2/LZ4。
只需要普通 float32 锁页接收时设置 `KPOLARIS_DDC_PINNED=1`，不设置 COMPACT。
两个开关只有字符串 `1` 表示开启；环境开关和服务参数应与 `.params` 一起保存。

compact SDK 的 include 目录目前不是安装后的自动依赖；直接编译使用 compact
接收头的下游项目，还需显式提供相容的 DDC 头文件路径。

## 精度和计时

接口测试比较同一解码状态的 DDC/PHDF 成像；压缩精度测试比较解码结果与原始流体。
这两种误差要分开测量。CPU 与 GPU 的指数实现可有微小差别，因此重建一致性也要
和有损压缩误差分别报告。

统计初始准备、输入等待、上传/物化、几何追踪和转移积分时，注意流水线重叠。
服务解码时间可能已经包含在客户端等待时间内，不能再次相加。批次总时间除以
图像数描述吞吐量，单张请求到完成的时间描述延迟。为每项计时保存物理参数、
快照时间间隔、内存设置和对应图像误差。

## 多频分组

`freq_list=230e9,345e9,86e9` 指定频率。编译选项 `KPOLARIS_MAX_FREQUENCIES`
限制一个内核共同推进的频率数，默认 1。编译为 2 后，`multifrequency_chunk_size=0`
或 2 可两频共用几何；设为 1 可逐频对照。超出容量会分组，所有组共用快照读取和 Pass A。
诊断也支持共享几何，每个频率独立累积，暂停/拒步不重复计数。组内步长受所有频率约束，
更改分组可能改变有限步长的数值结果，须用收敛比较判断；更改纯缓存大小则不应改变结果。
多频诊断在 `/frame_0/freq_i/analysis`，单频在 `/frame_0/analysis`。
其他模型也可用 `slow_light_windows_per_block` 控制驻留量；GiB 快照预算当前仅估算 KHARMA，
不包括随图像数、频率数和诊断配置增加的光线状态。
