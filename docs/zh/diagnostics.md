# 物理参数响应、传播影响与源区分布

本功能扩展 `analysis_mode`，支持 RIAF、iHARM/KHARMA 与 AthenaK 的完整 Stokes 物理参数响应、传播机制的空间分解，以及按几何区域或等离子体属性标记的观测贡献。实现共用于普通转移和慢光窗口转移；每个频率有独立的诊断状态。相关辅助图像、区域分解和灵敏度分析已有文献先例，不应单独作为首创声明。

## 1. 参数变化的物理定义

每次成像选择一个参数族，导数坐标为 `q = ln(scale)`，基准为 `q=0`。所有参数族保持时空、相机、流体速度、磁场方向以及基准辐射区域掩膜不变，并重新计算全部发射、吸收、旋转和转换系数。
随后应用当前的赤道源区选择和旋转开关；direct-only 响应沿所选光路段进行。
因此关闭旋转时的参数响应描述 rho_V 被固定为零的模型，rotation 机制通道也为零。

| `analysis_response` | 参数族 | 可以回答的问题 |
|---|---|---|
| `density_scale` | `ne -> exp(q) ne`，`B -> exp(q/2) B`；电子温度、sigma、beta 不变 | 与固定无量纲 GRMHD 状态下改变质量单位相应的辐射响应；增亮和增加吸收/Faraday 作用如何竞争？ |
| `temperature_scale` | 基准电子温度处方和下限处理之后，`Thetae -> exp(q) Thetae` | 电子升温通过发射还是传播改变观测偏振？这是整体电子温度标度，不等同于 `R_high` 的导数。 |
| `magnetic_scale` | `B -> exp(q) B`，`sigma -> exp(2q) sigma`，`beta -> exp(-2q) beta`；电子温度固定 | 固定磁场形状的场强变化如何影响强度、线偏振和圆偏振？启用依赖 sigma/beta 的可变 kappa 时重新求值。 |
| `coefficients` | 四个系数块各自的对数振幅响应 | 哪个位置的吸收、旋转或转换对最终观测有影响？这些是机制微扰，不能直接称为密度或温度导数。 |
| `none` | 只输出源区贡献 | 指定区域的种子发射最终贡献了多少 Stokes？ |

物理参数族定义在采样后的等离子体上，使用原模型相同的同步辐射系数实现，未另写一套拟合公式。`density_scale` 与重新改变质量单位的比较，在使用 float32 派生量缓存的 AthenaK 中可能有缓存重新量化造成的小差别。已有辐射掩膜视为模型域的一部分；这些导数不包含移动 sigma 排除边界、改变 GRMHD 动力学或磁场方向。需要这些变化时，应定义相应的新模型族并重新验证。

## 2. 导数与机制空间分解

连续方程为

```
dS/dlambda = j - K S
d(dS/dq)/dlambda = -K (dS/dq) + dj/dq - (dK/dq) S.
```

实现对实际使用的半解析 Strang 转移步求局部中心差分，分别改变以下系数块：

- `emission`: jI、jQ、jU、jV；
- `absorption`: aI、aQ、aU、aV，包括差异吸收；
- `rotation`: rhoV；
- `conversion`: rhoQ、rhoU。

对于每步 `S_next = T S + E`，局部扰动产生的响应通过后续完整基准传播矩阵继续演化。这既包含已有辐射对传播改变的响应，也包含本步产生的辐射对传播改变的响应。每个位置按基准分区归属，将局部导数注入对应的机制/分区通道。最后所有通道变换到与图像相同的观测者基。

四个机制、所有分区的导数之和，在差分步长收敛后等于该物理参数的完整导数。中心差分有 `O(h^2)` 误差，不能把有限 h 下的机制响应求和称为机器精度的导数恒等式。

每次响应成像同时沿相同采样光路进行四套完整转移，参数为 `q=+h,-h,+h/2,-h/2`。这些转移从自己的初始零 Stokes 独立演化，所有系数同时改变。后处理同时检查：

1. 源区 Stokes 求和与基准图像闭合；
2. 机制响应之和与完整半步幅中心差分相符；
3. 完整中心差分在 h 与 h/2 下收敛。

这三项验证不能替代光路/辐射采样步长收敛。用于科学结论时，还应减小 `max_radiation_step` 等控制量，检验观测量和导数本身的稳定性。

传播响应是**有符号影响**，不是能量占比或单纯的路径深度。某处 Faraday 深度很大，但在主发射后方时，可以几乎不影响观测；位于主发射前方的同一介质则可以强烈改变偏振。对于物理参数响应，吸收、旋转、转换各项还可能互相抵消。

## 3. 源区与等离子体分布

`analysis_partition` 决定源发射和局部响应的分区。每个有效采样属于恰好一个箱，分区固定在基准模型上。源区贡献保留完整 I、Q、U、V；分量先相加、再计算偏振度或偏振角。

| 分区 | 定义 |
|---|---|
| `radial` | 使用现有 `analysis_radial_bins/min/max` 的对数径向边界，首尾箱包括范围外采样 |
| `region` | 从任一自旋轴量起的极角：默认 `<20°` 为漏斗几何区、`20–60°` 为鞘层几何区、`>=60°` 为盘几何区；各分近侧/远侧，共六箱 |
| `plasma_region` | `sigma>=sigma_boundary`；其余部分按 `beta<beta_boundary` 或 `>=beta_boundary` 划分；另设不可用箱 |
| `near_far` | 球坐标位置方向与相机方向的点积非负/负。相机方位角为零；这是空间半球标记，不是沿弯曲光路的前后次序 |
| `thetae`, `sigma`, `beta`, `ne_cgs`, `b_cgs` | 按指定等离子体标量划箱，输出每个范围的观测 Stokes 贡献及机制响应 |

几何区名表示明确的极角范围，不自动证明其中流体是无束缚喷流。`plasma_region` 则直接区分磁化状态和气压/磁压关系，适合与几何区结果并用。RIAF 没有在本诊断中定义气体 beta/sigma，相关贡献进入不可用箱；不会把缺失值误当作弱磁化气体。

标量分箱支持自定义 `analysis_partition_edges=...`，严格递增，最多 61 个正数边界。包含低于首边界、高于末边界的箱，以及单独的不可用箱。边界值归入上侧箱。默认：

- Thetae：`1,3,10,30,100`；
- sigma、beta：`0.01,0.1,1,10,100`；
- ne [cm^-3]：`100,1000,10000,100000,1000000,10000000`；
- B [G]：`0.1,1,10,100,1000`。

源标记追踪的是种子发射位置。例如某区域的 V 贡献可以由该区域发出的线偏振，在别处经过转换而产生。传播响应通道则标记发生传播系数扰动的位置，两者结合可以区分发射介质与控制偏振的介质。

## 4. 使用

在项目根目录运行；把可执行文件替换为对应模型的 CPU 或 GPU 构建：

```bash
./build/kpolaris_model_image_riaf \
  --parameter_file=params/riaf_recommended.par \
  --analysis_mode=1 \
  --analysis_response=temperature_scale \
  --analysis_response_step=0.0003 \
  --analysis_partition=radial \
  --max_radiation_step=0.25 \
  --output=outputs/riaf_temperature_response.h5

python scripts/analyze_physical_responses.py \
  outputs/riaf_temperature_response.h5 \
  --output=outputs/riaf_temperature_response.json \
  --plot=outputs/riaf_temperature_response.png \
  --maps=outputs/riaf_temperature_observable_maps.h5
```

只做区域贡献时使用 `--analysis_response=none --analysis_partition=region`。只研究传播机制的位置时使用 `--analysis_response=coefficients --analysis_partition=radial`。例如研究磁化度分布，可使用 `--analysis_partition=sigma --analysis_partition_edges=0.01,0.1,0.3,1,3,10`。

几何边界由 `analysis_funnel_angle_deg`、`analysis_disk_angle_deg` 控制；磁化分类由 `analysis_sigma_boundary`、`analysis_beta_boundary` 控制。响应导数的单位是每单位 `ln(scale)`；正的小变化 epsilon 对应 `Delta S ≈ epsilon*dS/dln(scale)`。
`analysis_response_step` 是参数差分步幅，独立于光路积分步长和薄层采样数。

频率选择用后处理的 `--freq-index`，帧选择用 `--frame`。每次运行选择一个参数和一个分区；批量覆盖多个参数/分区时保留每次的有效参数文件。

JSON 包含源区强度与有符号偏振贡献、总参数响应、各机制/分区对通量、线偏振度、圆偏振度和 EVPA 的响应。`--maps` 导出像素级及各分区的这些响应图。近零强度或近零线偏振处的比值以有效性掩膜和 NaN 表示，绝对 Stokes 导数始终保留。

后处理默认要求所有光线返回，并要求源闭合误差低于 `1e-10`、导数分解及扰动步幅收敛的相对 L1 误差低于 `1e-2`。导数校验失败会保存报告并以非零退出；不得把失败的响应图当作已验证结论。报告同时给出以总 I 归一化的绝对误差，便于判断导数接近零时的数值尺度。

源闭合残差逐像素按所有来源的 Stokes 绝对值之和归一化。为避免数值下溢区的单个
浮点舍入差导致误报，归一化分母不低于 float64 最小正规数（约 `2.23e-308`）；
图像和来源数组不因此修改。报告记录这一数值下限及受影响像素数。
这不是用于屏蔽弱信号的亮度阈值；正常数值范围的闭合检查保持相同精度要求。

## 5. 输出、开销与验证

原生 HDF5 的相应 `analysis/physical_response` 保存：

- `source/{I,Q,U,V}_inv`: `[bin,ny,nx]`；
- `derivative/{emission,absorption,rotation,conversion}/d{I,Q,U,V}_inv_dlogp`: `[bin,ny,nx]`；
- `reruns/{plus_h,minus_h,plus_half_h,minus_half_h}/{I,Q,U,V}_inv`: `[ny,nx]`；
- 参数族、分区边界、近远侧定义、固定条件和数值方法的元数据。

单频文件的分析组在 `frame_0/analysis`；多频文件在各 `frame_0/freq_k/analysis`，与现有文件布局一致。

普通成像默认不启用这些功能。开启后复用光路，避免存储完整轨迹；每步仅构建一次齐次传播矩阵，供所有源区和响应通道复用。响应模式会额外计算四套扰动系数及转移，因此有计算开销。附加数组使用双精度，每像素内存为 `4*8*(5*Nbin+4)` 字节；只做源区标记时为 `4*8*Nbin` 字节。例如 256²、16 个箱的响应数组为 168 MiB，不包括已有图像、径向诊断和模型缓存。

验证入口：

```bash
ctest --test-dir BUILD -R physical_response --output-on-failure
python tests/test_physical_response.py \
  --riaf=BUILD/kpolaris_model_image_riaf \
  --iharm=BUILD/kpolaris_model_image_iharm \
  --athenak=BUILD/kpolaris_model_image_athenak \
  --workdir=outputs/physical_response
```

解析测试包括均匀介质层完整密度型响应、纯旋转/转换屏幕、发射与屏幕的次序、非对易混合转移及分箱边界。集成测试包括全部三种物理参数、系数机制、源分区闭合、分区合并不变性、独立质量单位重算、多频率及慢光窗口交接。这些合同检验功能和响应方程；科学使用时还需对自己的输入检查分辨率和转移步长收敛。

相关先例：White (2022), [Blacklight](https://arxiv.org/abs/2203.15963)；Ricarte et al. (2020), [内部 Faraday 分解](https://arxiv.org/abs/2009.02369)；Ricarte et al. (2021), [圆偏振机制](https://doi.org/10.1093/mnras/stab1289)；Motta et al. (2025), [Jipole](https://arxiv.org/abs/2509.07065)；Motta et al. (2026), [GRMHD 图像参数灵敏度](https://arxiv.org/abs/2604.11869)。



## 发射区域选择

### Direct-only 发射

参考 [Bezděková et al., arXiv:2512.09641, §II.1](https://arxiv.org/html/2512.09641#S2.SS1)，图像参数为 `direct_only=1`，默认 `0`。适用于单 Kerr 黑洞的 RIAF、torus 和 GRMHD 图像路径；快光、慢光、独立/共享几何多频率以及物理响应诊断使用相同判据。

#### 定义与物理解释

从观测者向源回溯，令 `z=r cos(theta)`，也就是 Cartesian Kerr–Schild 坐标的 z。在光线第一次从远离赤道面转向返回赤道面时，定位 `dz/ds=0`。只保留观测者与该点之间的发射。没有这类转向时，保留到通常模型边界的全部发射。

这不是赤道面穿越计数，也不是 `dtheta/ds=0` 或方位绕行计数。不同分段约定给出的“直接像”可以不同；输出文件保存此次使用的明确约定。所有光线仍按 Kerr 测地线传播，因此 direct-only 仍有引力弯曲。

KPolaris 的图像计算使用零入射背景。在线性 Stokes 转移方程 `dS/ds=j-KS` 下，把更远各段的 `j` 置零，则这些段的解始终为 `S=0`，不论其中的吸收和法拉第系数如何。因此可以在首个转向点直接以零 Stokes 开始正向积分：它与沿完整光路只关闭远处发射在物理上等价，并减少无效计算。默认设置下，前景段的发射、吸收、二向色性、法拉第旋转和转换照常计算。
若同时设置 `equatorial_h_over_r`，前景段只保留该薄层的发射；
若设置 `faraday_rotation=0`，前景旋转也会关闭，吸收与转换仍保留。
这些选择同样作用于响应的基准和扰动转移，见 本节的物理响应定义。若以后加入非零入射背景，必须重新处理背景的传播，不能直接复用此零边界实现。

慢光仍按保留光路各事件的真实延迟采样流体，未改成快光。已有物理响应和来源标签表示保留段的响应与观测贡献；路径长度、光学深度等沿路径累积量也对应保留段。需要完整光路的诊断时，应使用 `direct_only=0` 的对照产品。

#### 数值实现

- 在 Pass A 接受的几何步上检测同一半空间内的 `z*(dz/ds)` 从正到非正变化，取沿回溯方向的第一个事件。
- 对括住事件的步，用与原积分一致的 RK4 / 两次半步组合细化根，返回观测者一侧的端点。细化同时受浮点分辨率限制。
- BL / spherical KS 使用 `cos(theta) k^r-r sin(theta) k^theta`；FMKS 包含 `dtheta/dx1` 和 `dtheta/dx2` 两项；Cartesian KS 使用 `k^z`。
- 赤道平面内光线的舍入噪声使用相对半径的机器精度阈值排除。步长仍应足够分辨相邻的转向；发布图像需要步长和分辨率收敛检查。
- 截短终点会改变正向积分的离散采样网格，故独立计算的 `all-direct` 残差也受转移积分误差影响。它不能被当作逐像素严格闭合、同一网格上的高阶段标签。
- 原生内部终止码 `9 = reached_direct_turn` 表示 Pass A 分段事件；成功的最终图像仍用 `reason=1` 表示返回相机。
- 双黑洞 `binary_riaf` 没有此单黑洞自旋轴约定，明确拒绝此选项。

#### 使用

```bash
./build/kpolaris_model_image_riaf \
  --parameter_file=params/riaf_recommended.par \
  --direct_only=1 --output=outputs/riaf_direct.h5

# 对已有 AthenaK 参数文件添加同一个选项；慢光窗口选项照常使用。
./build/athenak/kpolaris_model_image_athenak \
  --parameter_file=athenak.par \
  --direct_only=1 --analysis_mode=1 \
  --analysis_response=temperature_scale --analysis_partition=region \
  --output=outputs/athenak_direct_response.h5
```

HDF5 根属性增加 `direct_only`、`emission_segment`；启用时另存 `direct_only_reference`、`direct_only_boundary`、`direct_only_transfer`、`direct_only_no_turn`。有效参数文件可重放 `direct_only` 设置。

#### 验证与可开展的分析

`tests/test_direct_only.cpp` 检查解析制造轨迹的根、上下半空间、赤道穿越排除、坐标 Jacobian、Kerr 光线及独立完整光路上仅关掉远处 `j` 的计算。`tests/test_direct_only.py` 检查原生 RIAF/iHARM/AthenaK 图像、分离/融合路径、多频率、静态慢光极限、诊断闭合、步长细化和参数重放。

配对的 full 与 direct-only 慢光电影可用于检验图像相关信号对更高阶段发射的依赖，并结合来源标签和传播响应，考察哪些源区及哪些等离子体效应影响该信号。单张 full/direct 图像只能说明该分段下的图像组成；确认时延相关峰仍需时间序列及相应统计检验。

配对图像可用以下命令生成 PNG、PDF 和带输入 SHA256 的 JSON：

```bash
python3 scripts/plot_direct_only_compare.py full.h5 direct.h5 \
  --output=outputs/direct_comparison
```

工具检查物理/相机元数据、赤道选择、旋转开关及光线完成状态，保留有符号残差，并检查负强度残差
是否超过数值误差容限，不对图像进行配准或重新归一化。


### 有限厚度赤道发射层

[Chael et al., arXiv:2606.12518v2，§V.1 / 图 5](https://arxiv.org/html/2606.12518#S5.SS1)
将 GRMHD 发射限制在赤道附近 `h/r=0.01` 的薄层，并关闭 Faraday 旋转，
比较内阴影边界的偏振方向与近视界解析结果。KPolaris 提供两个独立的图像开关，
用于构造这类受控的发射与传播实验。

#### 几何和物理定义

`equatorial_h_over_r=η` 保留满足

```
|cos(theta_BL)| = |z_KS| / r_BL <= η
```

的体发射。`h` 为相对黑洞自旋赤道面的半厚度，`r_BL` 为 Kerr 径向坐标；
不是圆柱半径，也不是 proper height。FMKS 使用映射后的物理 θ，
不能直接以原生网格 x2 距中面的距离判断。`η=0.01` 对应约 0.573° 的半张角。
论文没有给出代码级厚度判据，因此这里明确记录所采用的约定。

- `equatorial_h_over_r=0`（默认）关闭选择，恢复全部发射。
- `0<η<1` 只将薄层外的 `jI,jQ,jU,jV` 置零，薄层内保持原始体发射率。
- `η=1` 保留全部发射，且不添加薄层步长限制。
- `faraday_rotation=0` 单独将整个计算域的 `rho_V` 置零；默认 `1` 保留。
  吸收、二色性和 Faraday 转换 `rho_Q,rho_U` 均保留。

测地线及时间延迟不变，前景等离子体仍然传播所选光子。
因此这回答的是“赤道附近产生的辐射，经过原有前景后形成什么观测图像”。
关闭旋转的对照则隔离旋转对这一图像的影响；它仍包含 Faraday 转换，
不能称为“关闭全部 Faraday 效应”。对固定传播算子，full − wedge 表示
薄层以外发射的观测贡献；独立自适应运行之间允许存在数值积分误差。

薄层内发射率不按厚度重新归一化：缩小 η 会减少发射体积和通量。
零厚度解析表面源需要另行指定表面发射率，不能通过 η=0 获得。

#### 使用

在已有 iHARM、RIAF 或其他单 Kerr 图像配置后增加参数：

```bash
build/kpolaris_model_image_iharm --parameter_file=library.par \
  --nx=400 --ny=400 --equatorial_h_over_r=0.01 \
  --equatorial_samples=8 --output=equatorial.h5

build/kpolaris_model_image_iharm --parameter_file=library.par \
  --nx=400 --ny=400 --equatorial_h_over_r=0.01 \
  --equatorial_samples=8 --faraday_rotation=0 \
  --output=equatorial_no_rotation.h5
```

图像接口支持快光、慢光、多频率、分离/融合传输和物理响应诊断。
支持 RIAF、torus，以及现有单 Kerr GRMHD 图像后端（包括 iHARM 和 AthenaK）。
双黑洞模型没有统一的单 Kerr 赤道面，非零薄层参数会被明确拒绝。
这些选项目前属于 image 工具；trace 工具没有相应命令行开关。

可叠加 `direct_only=1`，同时选择赤道薄层与首个垂直转向前的发射段。
二者分别规定空间源区和保留光路，不能互相替代。
若比较论文图 5 的有限厚度实验，应先用全部光路的赤道选择。

#### 数值采样和验证

`equatorial_samples` 是**沿光线的数值采样参数**。`equatorial_h_over_r`
规定物理半厚度；像素数由 `nx,ny` 规定；诊断分箱由 `analysis_radial_bins`
或 `analysis_partition` 规定。增加采样数只收紧积分要求。

令 `q=z_KS/r_BL`，起点、中点、终点的取值为 `q0,qm,q1`。实际检查的是

```
|qm-q0| + |q1-qm| <= 2*equatorial_h_over_r/equatorial_samples
```

（允许浮点判断容差）。一次完整、单调穿过薄层通常至少需要约 N 个接受步；
N=8、h/r=0.01 时右侧为 0.0025。几何与传播精度要求可能产生更多步，
从层内开始或掠过部分薄层的光线不要求恰好 8 步。步长控制在辐射域内
还会检查层外的高度比变化，以便在进入薄层之前及时收紧步长。


`equatorial_samples=8` 控制单个接受步的高度比变化上限 `2η/8`；
检查起点、中点、终点，并在穿过薄层表面时用 RK 轨迹细化边界。
这避免长步跳过薄层和硬边界相位引起的发射柱误差。
同时继续执行原有测地线、辐射步长、吸收和 Faraday 深度控制。
`max_radiation_step` 限制代码归一化的仿射步长，不能解释为固定的径向或固有距离。
极端弯曲/掠射轨迹仍应以通常的几何步长细化确认收敛。

允许 `η=0` 或 `[1e-6,1]`，采样数为 `[2,1024]` 内整数。
单精度构建还要求非零 η 至少为 `1024 × machine epsilon`；更薄的层使用双精度。
若最小步长仍不足以解析薄层，光线标为 `adaptive_step_underflow`，
不会被标为成功到达相机。高精度工作应检查全部终止码，并比较
`equatorial_samples=8,16`，必要时再缩小 `max_radiation_step`。

`tests/test_equatorial.cpp` 用已知解析解的发射薄层加外部吸收屏检验积分，
包含两种传播方向、不同厚度、RK 组合及恰好位于步中点的边界。
`tests/test_equatorial.py` 检查原生成像、默认/全空间等价、来源闭合、
物理参数重算、关闭旋转后保留转换、慢光极限、多频率和参数重放。

#### 输出与分析

仍保存标准全 Stokes 图像、终止诊断和可选的物理分析产品。
HDF5 根和 `/parameters/radiation` 记录三个新参数；根另外保存几何、
传播、采样定义和参考文献。有效参数文件可重放，CSV 注释记录开关。

来源标签和发射加权量只计入选中区域；吸收与传播深度沿保留的整个
光路累计，因此能区分薄层的发射位置与前景的偏振改造。
完整密度/温度/磁场响应会对扰动后的系数再次应用相同选择及旋转设置。
后处理 JSON 也携带这些设置，避免把关闭旋转的响应误读为原始等离子体响应。

```bash
python3 scripts/plot_equatorial_compare.py full.h5 equatorial.h5 \
  --no-rotation=equatorial_no_rotation.h5 --output=equatorial_compare
```

绘图脚本核对相机、流体参数、频率、慢光和路径选择，输出 PNG、PDF、
带输入校验和的 JSON。亮度使用共同标尺，EVPA 遮去弱光/弱线偏振像素。

这类对照可用于测量赤道发射对图像和偏振的贡献、分析前景旋转作用，
以及在厚度序列中判断近视界偏振结构何时接近赤道薄层极限。
要验证论文中的普适偏振结论，还需相应视角/磁流体模型的时间平均、
内阴影边界定位及解析 EVPA 曲线比较；单帧薄层图只是其中的成像工具。


## 积分精度与诊断精度

`adaptive_tolerance` 控制几何 RK 步进的误差估计，`max_step` 控制最大几何步长。
`max_radiation_step`、`max_absorption_depth` 和 `max_faraday_depth` 约束转移采样。
放宽几何容差不能替代辐射步长检查，几何闭合很好也不自动保证强 Faraday 区的 Q/U/V
已收敛。`analysis_response_step` 是参数差分幅度，与这些光路步长不同。

使用随代码提供的参数文件开始，固定物理模型和视场，分别检查分辨率、几何误差、
转移采样和响应差分幅度。逐分量图像误差可用
`NMSE(S) = sum((S-S_ref)^2) / sum(S_ref^2)`；参考分量平方和为零时应标为未定义。
参考图也需要更严格设置下的独立收敛检查。I 的收敛不能代表弱 Q/U/V 或参数导数。

单纯径向深度表示沿途积累；源区 Stokes 表示哪里发出的辐射抵达了观测者；
传播机制响应表示在哪里改变吸收、旋转或转换会影响结果。科学解释应使用对应
的量，不能把一个很大的传播深度自动当作该处主导观测偏振的证据。
