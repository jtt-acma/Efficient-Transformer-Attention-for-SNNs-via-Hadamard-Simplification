# Efficient Transformer Attention for SNNs via Hadamard Simplification (SSA / USSA)

This repository contains the official implementation of our ICML 2026 paper:

> **Efficient Transformer Attention for SNNs via Hadamard Simplification**  
> Tingting Jiang, Jiangrong Shen, Long Chen, Yaxin Li, Qi Xu

We propose **Simplified Spiking Attention (SSA)** and **Ultra-Simplified Spiking Attention (USSA)**, which replace dense matrix multiplication with Hadamard products, significantly reducing computational and communication complexity for Transformer-based SNNs on neuromorphic hardware.

**Our work is built upon [SpikingResformer](https://github.com/xyshi2000/SpikingResformer) (Shi et al., CVPR 2024) as the baseline.** We inherit its overall architecture and training pipeline, while introducing hardware-friendly attention mechanisms to replace the original MDSSA module.

---

## Installing Dependencies

```bash
pip3 install torch torchvision
pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm
```

## Usage

### Experiments on ImageNet

To reproduce the experiments on ImageNet in the paper, you need to first organize the dataset as follows

```bash
/path/to/your/dataset
|-- train
|   |-- n01440764
|   |-- n01443537
|   `-- ...
`-- val
    |-- n01440764
    |-- n01443537
    `-- ...
```

Then run the following command to reproduce the experiment of SpikingResformer-S

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
main.py \
    -c configs/main/spikingresformer_s.yaml \
    --data-path /path/to/your/dataset \
    --output-dir /path/to/your/output \
    ;
```

Experimental setups of SpikingResformer-Ti, M, L can be found in `configs/main`.

Run the following command to evaluate the pretrained checkpoints

```bash
python main.py \
    --model spikingresformer_s \
    --data-path /path/to/your/dataset \
    --resume /path/to/your/checkpoint \
    --test-only \
    ;
```

Experimental setups on other datasets can befound in `configs`.

### Direct Training

Run the following command to directly train SpikingResformer-Ti* on CIFAR10

```bash
python \
main.py \
    -c configs/direct_training/cifar10.yaml \
    --data-path /path/to/your/dataset \
    --output-dir /path/to/your/output \
    ;
```

Experimental setups on other datasets can be found in `configs/direct_training`.

## Citation

```bibtex
@inproceedings{jiang2026efficient,
    title={Efficient Transformer Attention for SNNs via Hadamard Simplification},
    author={Jiang, Tingting and Shen, Jiangrong and Chen, Long and Li, Yaxin and Xu, Qi},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2026}
}
```
