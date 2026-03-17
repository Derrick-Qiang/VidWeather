## Environment Setup
This project is recommended to run on Linux with an NVIDIA RTX 4090. NVIDIA GPUs are recommended for both training and inference. All dependencies are listed in `env.yaml`, and you can create the environment directly with Conda.

```bash
conda env create -f env.yaml
conda activate AdaIR
```

## Test Set Inference
```bash
./infer/infer_scene1.sh
./infer/infer_scene2.sh
./infer/infer_scene3.sh
./infer/infer_scene4.sh
./infer/infer_scene5.sh
./infer/infer_scene6.sh
```

## Checkpoint Download
Please organize the checkpoint files in the following directory structure:

```text
root/
`-- AdaIR/
    |-- XXX_1.ckpt
    `-- XXX_2.ckpt
```
