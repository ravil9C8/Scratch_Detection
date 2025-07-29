# Scratch Detection System

This repository contains a scratch detection system using both classification and segmentation approaches.

## Project Overview

The system uses:
- DenseNet121 classifiers (with and without letterboxing) for binary classification
- U-Net for segmentation of scratch regions

## Setup

### Prerequisites

- Python 3.12+
- PyTorch and torchvision
- Other dependencies listed in pyproject.toml

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/scratch-detection.git
   cd scratch-detection

2. Create and activate a virtual environment
   ```bash
   uv add .
   source .venv/bin/activate
## Testing Models

### Configuration 
Edit the paths in test_models.py to match your environment:
IMG_PATH     = Path("/path/to/test/image.png")
MASK_PATH    = Path("/path/to/mask/image.png")         # GT mask (binary)
CKPT_DENSE_LB= Path("/path/to/densenet121_best.pt")
CKPT_DENSE_RS= Path("/path/to/densenet121_without_letterboxing.pt")
CKPT_UNET    = Path("/path/to/unet_best_loss40_60.pt")
OUT_FIG      = Path("viz_unet_output.png")             # saved figure
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

### Runnning the test
```bash
python test_models.py
