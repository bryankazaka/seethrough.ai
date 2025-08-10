# SeethroughAI: Finetuning smolvlm256-videoinstruct Model

This repository describes the finetuning process for the `smolvlm256-videoinstruct` model backbone used in SeethroughAI.

## Setup

1. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2. **Install dependencies:**
    ```bash
    pip install -Ur requirements.txt
    ```

## Training Details

- Training was performed on a single NVIDIA RTX 3060 GPU with 12GB VRAM.

## Dataset

- The train/test splits are based on the roots of the GenViideo dataset.
- You can download the dataset using the `data_downloading` notebook.
- The GenViideo dataset repository is available at:  
  [https://github.com/chenhaoxing/DeMamba](https://github.com/chenhaoxing/DeMamba)

