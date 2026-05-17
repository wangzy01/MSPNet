# Lens Privacy Sealing: A New Benchmark and Method for Physical Privacy-Preserving Action Recognition

This repository contains the official implementation of **MSPNet** for the paper:

**Lens Privacy Sealing: A New Benchmark and Method for Physical Privacy-Preserving Action Recognition**  
Mengyuan Liu, Ziyi Wang<sup>&dagger;</sup>, Peiming Li<sup>&dagger;</sup>, Junsong Yuan  
Accepted as a Regular Paper by **IEEE Transactions on Image Processing (T-IP)**.

<sup>&dagger;</sup> Corresponding authors: Ziyi Wang and Peiming Li.

## Overview

RGB camera-based surveillance systems enable human action recognition for public safety, healthcare supervision, smart homes, and human-robot interaction, but they also raise privacy concerns during data acquisition. Lens Privacy Sealing (LPS) is a low-cost physical privacy solution that covers RGB camera lenses with adjustable laminating film to provide pre-sensor privacy protection.

MSPNet is designed for action recognition from LPS-degraded videos. It uses an Inter-Frame Noise Suppressor (IFNS) to reduce static scattering noise and a Cross-Frame Semantic Aggregator (CFSA) to fuse motion cues across frames. The model further uses contrastive language-image pre-training for robust semantic extraction from degraded video inputs.

![pipeline](assets/pipeline.png)

## Repository

- `models/`: MSPNet, IFNS/CFSA-related modules, and transformer components.
- `datasets/`: dataset loaders and preprocessing pipelines for P3AR-NTU and P3AR-PKU.
- `configs/`: training and testing configurations.
- `txt_file/`: split files and labels used by the experiments.
- `comparison/`: baseline and reconstruction-attack utilities used in the paper.

The P3AR-NTU dataset processing repository is available at:  
https://github.com/wangzy01/P3AR-NTU

## Installation

```bash
conda create -n MSPNet python=3.7
conda activate MSPNet
pip install -r requirements.txt
```

Optional Apex installation:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

The complete environment configuration is also provided in `requirements.yml`.

## Data

Please prepare the P3AR-NTU and P3AR-PKU data paths in the corresponding YAML files under `configs/`. For P3AR-NTU download and preprocessing instructions, refer to the dataset repository:

```text
https://github.com/wangzy01/P3AR-NTU
```

## Training

To train MSPNet on P3AR-NTU with 4 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=25658 ntu_main.py \
    --config configs/NTU/NTU120_XSet.yaml \
    --distributed True \
    --accumulation-steps 2 \
    --output output/ntu_encrypted
```

If your GPU memory is limited, adjust `--accumulation-steps` to maintain the effective batch size.

The pretrained CLIP model is downloaded automatically. You can also specify a local checkpoint:

```bash
--pretrained /PATH/TO/PRETRAINED
```

## Testing

To test MSPNet on P3AR-NTU with 4 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=25658 ntu_main.py \
    --config configs/NTU/NTU120_XSet.yaml \
    --resume /PATH/TO/CKPT \
    --output output/ntu_encrypted \
    --only_test True
```

## Citation

```bibtex
@article{liu2026lens,
  title={Lens Privacy Sealing: A New Benchmark and Method for Physical Privacy-Preserving Action Recognition},
  author={Liu, Mengyuan and Wang, Ziyi and Li, Peiming and Yuan, Junsong},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  note={Accepted}
}
```

## Contact

For questions about this work, please contact the corresponding authors:  
`ziyiwang@stu.pku.edu.cn`, `lipeiming1001@stu.pku.edu.cn`.
