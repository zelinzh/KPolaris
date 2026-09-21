# 相机、图像方向与偏振约定

本文件定义 KPolaris 图像、内部输运基和可选的 N/W 输出零点。约定来自相机几何与偏振基，
不通过与另一幅图像拟合来决定。度规号差为 `(-,+,+,+)`。

## 1. 坐标和观测方位

单黑洞坐标的 `+z` 是坐标北极，倾角 `inclination` 从 `+z` 测量。
针孔相机方位角固定为原生 `phi=0`；平行相机使用下文定义的 Cartesian 平面。
当前接口没有独立的相机绕视线旋转参数。
负自旋表示角动量沿坐标 `-z`；不能把坐标北极始终叫作自旋方向。
模拟数据中的磁场方向保留其符号；`reverse_field` 是改变物理磁场极性的选项，
不是图像方位转换。

针孔相机位于 `(r,theta,phi)=(R,i,0)`。在 Cartesian Kerr–Schild 坐标中，
同一个位置是 `(R sin i, a sin i, R cos i)`。含自旋的第二个分量来自坐标变换，
不是为了对齐图像而平移相机。其局部观察者是 ingoing Kerr–Schild 时间切片的
法向观察者；在 Boyer–Lindquist 坐标中也要变换这个观察者，不能另选一个。

针孔 tetrad 的第三个空间轴向外；中心光线具有零极向和方位角协变动量。
第二轴由增加原生极向坐标的试探向量投影、正交化得到；第一轴由右手性确定。
这是一种明确的有限半径相机定义，与 ipole 的零冲量参数定心相机相容。
轴上观测需要选择方位角的极限，屏幕上的“北”本身不唯一。

平行相机直接定义 Cartesian KS 平面：中心为 `(R sin i,0,R cos i)`，
向观察者的光线方向为 `n=(sin i,0,cos i)`，平面横轴为 `+e_phi=(0,1,0)`，
纵轴为 `-e_theta=(-cos i,0,sin i)`。这是另一种相机投影；有限 R、自旋不为零时，
不能假定它和针孔相机逐像素是同一组物理光线。

## 2. 光子方向与屏幕基

存储的 `k` 始终是未来指向的物理光子波矢，流体测得频率为 `-u_mu k^mu>0`。
Pass A 从相机用负仿射步长追踪至端点，Pass B 用正仿射步长传播至相机。
两遍之间不翻转 `k`，也不重建屏幕基。

每条光线的 `e1,e2` 均与 `k` 正交；在观察者局部静止系中，
`(e1,e2,n)` 按光子传播方向构成右手系。针孔相机对每个像素将 tetrad 的
第一、第二轴投影到该光线的垂直平面，再做 Gram–Schmidt 正交化。
平行相机从平面横、纵轴构造基。此后沿测地线平行输运。

Pass B 最后使用 `R_ij = e_observer,i · e_transported,j` 投影回相机屏幕。
这里的旋转是实际输运基和相机基的重叠，不是加一个人为 EVPA 校准角。
若基发生反射，V 的变换由 `det(R)` 决定；一般不能只翻转 U 或 V 中的一个。

## 3. 像素、数组与图像坐标

像素编号 `pixel=iy*nx+ix`；HDF5 图像数组为 `[iy,ix]`，即 `[y,x]`。
`ix` 增加向图的右侧，`iy` 增加向上，绘图使用 `origin='lower'`。
原生数组不做上下、左右翻转或转置。

在远处 `phi=0` 观察者的极限下，图像 `+x` 指向 `+e_phi`，`+y` 指向
`-e_theta`，即坐标 `+z` 在视平面的投影。对于针孔相机，回溯方向是 `-k`，
因此光线在源侧的位移方向与 outgoing tetrad 的横向分量相反。
中心处针孔屏幕两轴与图像两轴同时反向；同时反向是 180°，不会改变 Q/U、V
或无箭头的 EVPA 线段。离轴像素使用其实际投影后的局部屏幕基。

默认从像素中心采样：`sx=((ix+0.5)/nx-0.5)*fov+x_offset`，y 同理。
平行相机的宽度以 M 表示；针孔的 `sx,sy` 是局部 tetrad 中的方向比，
`k^(a)=(1,sx/L,sy/L,1/L)`，`L=sqrt(1+sx²+sy²)`。

`use_pinhole_pixel_bias=0` 为默认值。显式开启后，
`pinhole_pixel_bias` 按像素宽度叠加到横向采样位置；`-0.01` 保留用于重放
ipole 的历史采样位置。这不是物理方位修正，也不是生成图像后的注册平移。
旧参数文件若明确写了开启，仍保持原采样；此前未写此项的文件应显式补充后再重放。

## 4. Stokes、EVPA 与圆偏振

在上述局部屏幕上，取电场为 `Re[E exp(-i omega t)]`，定义

```text
I = <|E1|² + |E2|²>
Q = <|E1|² - |E2|²>
U =  2 Re<E1 E2*>
V = -2 Im<E1 E2*>
chi = 0.5 atan2(U,Q)  (mod pi)
```

Q>0 表示电场更偏向 e1，U>0 对应从 e1 向 e2 的 +45° 线偏振。
例如复振幅 `(E1,E2)=(1,i)` 给出 V>0，其实电场随局部时间从 e1 转向 e2。
给出复电场、时间相位和屏幕手性可以消除单说“左旋/右旋”时观察方向的歧义。
Q=U=0 时 EVPA 未定义；微弱线偏振处绘图须屏蔽。

纯 Faraday 旋转的本地约定为 `dQ/dl=-rho_V U`、`dU/dl=rho_V Q`，
因此正 rho_V 使 chi 沿 e1→e2 增大，`dchi/dl=rho_V/2`。
一般存在发射、吸收、转换时，累计 rho_V 不能直接当作最终 EVPA 改变量。

局部同步辐射系数使用 e2 沿投影磁场的基；光学薄发射的正 jQ 表示
电矢量垂直投影磁场。设磁场在输运屏幕上的分量为 b1,b2，则旋转因子为
`cos(2chi_B)=(b2²-b1²)/(b1²+b2²)`、`sin(2chi_B)=-2b1b2/(b1²+b2²)`。
同一旋转作用于发射 jQ、吸收 aQ 和转换 rho_Q；不能只为图像好看旋转发射。
RIAF、环面、网格 GRMHD 和直接 meshblock 路径采用同一个磁场基定义。

图像中的 Stokes 已投影到相机屏幕，并按 `evpa_0` 输出。Trace 的中间样本、
系数、`e1/e2` 和 `final_propagated_*` 保持内部平行输运基；只有
`final_observed_*` 和 `final_evpa_wrapped_rad` 使用所选输出零点。不能混用
样本 EVPA 和最终观测 EVPA。内部几何 overlap 和 basis-rotation 诊断不随输出选项改变。

## 5. 天空位置角和跨代码比较

图像按天文图像习惯约定上方为北（N）、左方为东（E）。这里是图像方向的命名，
与 ipole 的 N/W 输出零点用法相同；它不自动指定模拟系统在真实天球上的位置角。
通常远处观察者的小视场图像中，默认 EVPA 从图像北向东增加，模 π。

image 和 trace 的命令行或参数文件均支持：

```text
evpa_0=N
```

- `N`（默认）：以图像竖直方向为零点，相对内部相机基输出
  `(I,-Q,-U,V)`，对应 ipole 的 `evpa_0=N`。
- `W`：以水平方向为零点，保留内部相机基的 `(I,Q,U,V)`，
  对应此前 KPolaris 的 `evpa_0=camera`，以及 ipole 的 W 输出。
- N/W 是偏振参考轴之间的 90° 被动旋转；像素、光线、强度和物理偏振线方向
  不变。V 的符号不变。不允许未知值。

同一设置作用于所有频率、慢光批次帧、IQUV 的有量纲/不变量图、积分通量、
径向贡献、源项分解、响应导数和响应重算图。HDF5 根、header、频率组，CSV
头和可重放参数文件均记录实际选择。旧文件中 `camera` 或缺省的历史 KPolaris
零点按 W 读取；未知或相互冲突的元数据报错。重放旧参数文件时，显式增加
`evpa_0=W` 才能保持旧 Q/U 数值；新文件默认 N，不应再无条件翻号。

该选项只改变输出基，不改变任何传输系数或测地线计算。严格定义仍通过每条光线
的正交屏幕基给出；宽视场针孔图像的偏振线使用实际投影，而不是将局部角度
直接当作整个平面上的位置角。与真实源比较时，再指定模拟系统的天空位置角。

若新基 `e1'=cos psi e1+sin psi e2`、`e2'=-sin psi e1+cos psi e2`，则

```text
Q' = Q cos(2psi) + U sin(2psi)
U' = -Q sin(2psi) + U cos(2psi)
I' = I, V' = V
chi' = chi - psi (mod pi)
```

这是被动换基，psi 必须来自两个基的几何关系，不能通过最小化图像误差来选择。
数组维度换序 `[x,y]→[y,x]` 只是存储转换；物理镜像或旋转还要求变换坐标和
Stokes 基，不能仅调用 `flip`、`rot90` 或加减角度。

ipole 默认 `qu_conv=0` 的 `save_pixel` 在写文件时将其相机 Q/U 同时取负，
并记录 `header/evpa_0=N`；`evpa_0=W` 则保留相机 Q/U。因此 N↔相机的
`(Q,U)→(-Q,-U)` 是明确的 90° 基变换，I、V 不变，并非拟合得到的修正。
这只有在相机方位、roll、投影和采样位置也匹配时才能用于逐像素比较。
ipole `/pol` 的 `[x,y,component]` 与 KPolaris `[component,y,x]` 的维度换序
也不是物理翻转。转换工具要求明确的 N/W 元数据，不猜测缺失的零点。

不能因为另一代码的磁场基 jQ 符号不同，就认定其最终相机 Q/U 应整体翻号。
必须检查完整系数、基底和输出定义。Blacklight 转换工具因此保留原生值；
需要换基时使用显式 `--basis-angle-deg` 并记录从输入基到输出基的几何依据。

绘图默认保留文件的 N/W Stokes 和 EVPA 数值，并标明零点；偏振线自动换回
内部屏幕基后投影，所以切换 N/W 不会旋转物理偏振线。`--qu-conv=camera`
或 `--evpa-conv=camera` 显式将绘图量换到 W。跨代码 CSV 转换器的 `stored`
保留文件值、`camera` 统一到 W、`ipole-native` 统一到 N；读取元数据，避免
二次翻号。针孔偏振线段使用实际 gnomonic 投影的
Jacobian；宽视场离轴像素不能一概用 `(cos chi,sin chi)` 代替。

## 6. 可核查的输出

图像与 trace 根属性写入 `polarization_conventions_version=2`、
`stokes_definition`、`electric_field_phase_convention`、`screen_handedness`、
`evpa_definition`、`camera_azimuth_convention`、`sky_orientation`、
`image_pixel_order`、`image_axis_geometry` 和 `output_stokes_transform`。
图像另外保存 `image_array_order=y,x`、`image_display_origin=lower`；
trace 保存中间样本与最终观测量的基底区别。有效采样偏移仍由相机参数记录。
N/W 不改变 Stokes 数据集的单位或排列，但新默认 N 会使旧 W 输出的 Q/U 取负。旧文件没有这些属性时
须结合生成版本和有效参数解释。
