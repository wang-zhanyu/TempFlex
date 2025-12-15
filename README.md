# TempFlex: Advancing MLLMs with Temporal Perception and Natively Scalable Resolution Encoding

### Installation

#### 1. **Clone this repository and navigate to the tempflex folder:**
```bash
git clone https://github.com/wang-zhanyu/TempFlex.git
cd TempFlex
```

#### 2. **Install the package:**
```bash
conda create -n tempflex python=3.10 -y
conda activate tempflex
pip install --upgrade pip
pip install -e ".[train]"
```

### Model Training

#### stage 1
```bash
bash scripts/train_tempflex_qwen3/stage_1.sh
```

#### stage 2
```bash
bash scripts/train_tempflex_qwen3/stage_2.sh
```

#### stage 3
```bash
bash scripts/train_tempflex_qwen3/stage_3.sh
```

#### stage 4
```bash
bash scripts/train_tempflex_qwen3/stage_4.sh
```

## Acknowledgement

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT/): We thank the LLaVA-NeXT team for releasing their codebase. Their contributions greatly facilitated the development and implementation of our approach.
