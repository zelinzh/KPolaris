# KPolaris 中文手册

[English manual](../README.md)

中文说明与英文手册分开维护，命令均从源码目录运行；参数名称和文件格式相同。

首次使用可先运行不需要模拟数据的 M87* 示例；已有 GRMHD 数据时，从输入指南
选择对应 reader，再设定物理单位和相机。生成时间序列则继续阅读慢光与批处理。
首次 CUDA 编译可能需要数十分钟，请按需构建并复用生成的程序。

- [安装、输入和成像](user_guide.md)
- [M87* 首次成图示例](first_image.md)
- [参数设定与模板注释](parameters.md)
- [相机、图像方向与偏振约定](polarization_conventions.md)
- [物理诊断、参数响应与发射区域选择](diagnostics.md)
- [慢光、批处理和 DDC](slow_light.md)

以下详细参考使用英文：

- [命令行辅助工具及全部子命令](../cli.md)
- [完整参数参考：含义、单位、默认值和适用范围](../parameters.md)
- [安装与环境配置](../quickstart.md)
- [绘图工具](../plotting.md)
- [电子分布与辐射系数](../electron_distributions.md)
- [HDF5 字段和单位](../hdf5_schema.md)

参数模板不是任意 GRMHD 数据的通用物理设定。运行前应核对坐标、质量单位、
电子温度处方与相机，使用结果前检查光线完成状态及目标观测量的收敛性。
