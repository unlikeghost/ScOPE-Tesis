# 🧪 ScOPE Experiments

Experiments using the **ScOPE** on molecular datasets from [MoleculeNet](https://moleculenet.org/).

## 📊 Datasets

This repository includes experiments on the following molecular property prediction datasets:

| Dataset | Task | Description | Molecules |
|---------|------|-------------|-----------|
| **BACE** | Classification | β-secretase 1 (BACE-1) inhibitors | ~1,500 |
| **BBBP** | Classification | Blood-brain barrier penetration | ~2,000 |
| **ClinTox** | Classification | Clinical toxicity | ~1,500 |
| **HIV** | Classification | HIV replication inhibition | ~41,000 |

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Git LFS](https://git-lfs.github.io/) for dataset files

### Usage

1. **Install dependencies:**
   ```bash
   uv sync --group notebook
   ```

3. **Install Git LFS and download datasets:**
   ```bash
   # Install Git LFS (choose your platform)
   brew install git-lfs          # macOS
   sudo apt install git-lfs      # Ubuntu/Debian
   conda install git-lfs         # Conda

   # Initialize and pull datasets
   git lfs install
   git lfs pull
   ```
