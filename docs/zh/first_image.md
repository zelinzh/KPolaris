# M87* 首次成图示例

首次成图默认采用 M87* 的质量（65 亿太阳质量）与距离（16.8 Mpc），计算
230 GHz、256² 的解析吸积流图像。打开 `image_freq0.png` 即可看到中央暗区、
亮环及两侧亮度不对称。图像由完整偏振辐射转移计算产生，未施加观测波束，
未翻转或旋转图像；坐标是相机图像平面上的微角秒偏移，并未指定天球位置角。
线性色标表示亮温，不是电子温度。图像、偏振、诊断和多频率示例采用同一组
物理参数。质量与距离取自 EHT 文献，等离子体和自旋参数用于演示，
不代表对观测数据的拟合。快速检查安装可显式指定 `--resolution=64`。

生成首页所示的 I/Q/U/V 横排图：

```bash
python3 scripts/plot_kpolaris_pol.py outputs/first-image/image.h5 \
  --layout=stokes --columns=4 --stokes-scale=independent \
  --intensity-unit=brightness-temperature --fov-units=muas \
  --output=outputs/first-image/stokes.png
```

I/Q/U/V 使用相同物理单位，色标线性显示；各偏振分量使用自己的正负色标范围，
便于辨认结构。比较幅度时应读各自色标；如需以共同的峰值 I 范围显示，
使用 `--stokes-scale=shared`。绘图不改变 HDF5 数据。

完整参数与参考文献见[英文示例说明](../first_image.md)。
