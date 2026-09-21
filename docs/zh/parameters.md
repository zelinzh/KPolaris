# 参数设定说明

`params/` 中的模板已按功能分组添加英文注释，用于快速理解每一部分在调整什么。
[完整参数参考（英文）](../parameters.md)逐项列出图像程序的参数含义、单位、
默认或自动取值、适用模型和兼容别名，并说明 trace 的专用参数。
编译选项与运行参数不同：`--models`、`--coordinates`、`--jobs` 属于
[构建工具](../cli.md)，不应放入成图参数文件。

## 使用参数文件

```bash
cp params/kharma_recommended.par kharma.par
# 先编辑 kharma.par：输入路径、质量单位、电子温度、相机等。
./build/kharma/kpolaris_model_image_kharma --parameter_file=kharma.par \
  --kharma_dump=/data/snapshot.phdf --kharma_resample=none \
  --nx=256 --ny=256 --output=image.h5
```

模板只是起点，不能将其中的物理归一化直接用于任意模拟。首次处理 GRMHD 数据，
先按[输入流程](../user_guide.md)检查数据格式、坐标和单位。
KHARMA 模板包含球面 Kerr–Schild 重采样；上例明确关闭它，使用原生网格。

文件中每行使用 `key=value`，`#` 后面是注释。文件里的值不加引号；含空格路径
也可用 `key=value`。命令行可重复指定 `--parameter_file`，程序按顺序读取文件，
然后应用命令行的显式覆盖。参数文件内部的 `parameter_file=` 不会递归包含文件。
`parameter_output=auto` 保存最终有效参数为 `<output>.params`，应与图像一起保留。

## 优先检查的参数

下表中的 `MODEL` 替换为 `iharm`、`kharma`、`athenak` 或 `bhac`。

| 部分 | 主要参数 | 调整内容 |
| --- | --- | --- |
| 输入与归一化 | `MODEL_dump`, `MODEL_M_unit`, `MODEL_mbh_solar` | 数据路径、以克为单位的模拟质量归一化、以太阳质量为单位的黑洞质量；后两者不是同一个量。 |
| 电子温度 | `MODEL_trat_small`, `MODEL_trat_large`, `MODEL_beta_crit` | 低／高等离子体 beta 下的离子与电子温度比及过渡尺度。 |
| 辐射区域 | `MODEL_sigma_cut`, `MODEL_sigma_cut_high`, `outer_radius` | 高磁化度掩膜及积分域边界。这是物理选择，不能作为单纯的精度设置。 |
| 视角与视场 | `inclination_deg`, `radius`, `dsource`, `fovx_dsource`, `fovy_dsource` | 倾角、有限距离相机位置、物理源距离及角视场。 |
| 像素与频率 | `nx`, `ny`, `freq` 或 `freq_list` | 像素数和观察者频率；改这些值通常不需要重新编译。 |
| 几何精度 | `adaptive_tolerance`, `min_step`, `max_step`, `max_steps` | 测地线与屏幕基积分误差控制和工作量上限。容差不是图像 NMSE。 |
| 辐射采样 | `max_radiation_step`, `max_absorption_depth`, `max_faraday_depth` | 辐射采样间距和每步局部传输深度；应与几何精度分别检查。 |
| 慢光 | `slow_light_*` | 数据时间序列、观察者到达时刻、时间插值和内存驻留；见[慢光说明](slow_light.md)。 |
| 物理诊断 | `analysis_*`, `direct_only`, `equatorial_*` | 发射形成位置、源项与参数响应、发射选择；见[诊断说明](diagnostics.md)。 |

## 单位和容易混淆的设置

- 半径以 `GM/c²` 为单位，时间以 `GM/c³` 为单位；双黑洞使用总参考质量。
  `freq` 单位是 Hz，`dsource` 是 pc，角视场是微角秒。
- `step`、`min_step`、`max_step` 是正的仿射步长大小，不是固定的空间距离或
  快照时间间隔。程序自行处理两遍积分的方向。
- `xspan`/`yspan` 是以几何长度计的半宽；`fovx_dsource`/`fovy_dsource` 是角全宽。
  针孔相机的 `fov` 是屏幕斜率全宽，小角度下近似弧度。不要混用多种视场输入。
- 后写的 `fov` 会清除此前的 `xspan` 等视场设置；因此 RIAF 另一份模板中
  `fov` 在 `xspan` 后面时，以 `fov` 为准。
- `max_radiation_depth` 同时赋值给吸收和 Faraday 限制。需要分别设置时，
  将 `max_absorption_depth`、`max_faraday_depth` 写在它后面。
- `*_warning` 只改变报警阈值，不会提升积分精度。
- `torus_omegac` 是中心焓密度归一化，**不是角速度**；`torus_kappa` 是多方指数，
  **不是电子 κ 分布参数**。
- `interpolate_derived_scalars` 控制空间取样的次序，与慢光 `slow_light_interpolation`
  的时间插值选择相互独立。
- `slow_light_snapshot_cache_gib` 是 KHARMA 快照数组显存预算，不是总显存上限；
  光线、诊断数组和运行时还需要额外空间。

完整参考还说明了自动值 `0`/`-1` 的具体含义、各模型的额外选项、trace 存储控制和
可选功能的编译要求。没有出现在模板里的参数，可以按参考文档添加；非热、慢光、
多频和诊断仍需相应的编译支持。
