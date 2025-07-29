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
   git clone https://github.com/ravil9C8/Scratch_Detection.git
   cd scratch-detection

2. Create and activate a virtual environment
   ```bash
   uv add .
   source .venv/bin/activate
## Testing Models

### Configuration 
Edit the paths in test_models.py to match your environment

### Runnning the test
```bash
python test_models.py
