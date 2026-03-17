## 环境配置
本项目建议在搭载 NVIDIA RTX 4090 的 Linux 环境下运行，模型训练与推理推荐使用 NVIDIA GPU 设备。项目依赖已整理至 env.yaml 文件，可直接通过 Conda 创建运行环境。

```bash
conda env create -f env.yaml
conda activate VP-AdaIR
```

## 测试集推理
```bash
./infer/infer_scene1.sh
./infer/infer_scene2.sh
./infer/infer_scene3.sh
./infer/infer_scene4.sh
./infer/infer_scene5.sh
./infer/infer_scene6.sh
```

