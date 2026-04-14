# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indonesian Sign Language (SIBI) lip reading project — a thesis research project classifying 18 labels (numbers 1,2,3,8,9,10; letters a–f; words: buku, dia, saya, keliling, kelompok, sekarang) from video sequences using deep learning.

## Environment

- Python 3.12.3 in `.venv/` virtual environment
- Activate with: `source .venv/bin/activate`
- GPU: NVIDIA RTX 5050 Laptop GPU (CUDA enabled, PyTorch 2.11.0+cu128)
- Run notebooks with: `jupyter notebook` or `jupyter lab`

## Running Notebooks

The project is entirely notebook-based. Run sequentially from `Lip_Reading_Gabungan/notebooks/`:

```bash
cd Lip_Reading_Gabungan
jupyter notebook notebooks/
```

| Notebook | Status | Purpose |
|----------|--------|---------|
| `01_Environment_Setup_Gabungan.ipynb` | Done | Validates environment, creates directory structure |
| `02_Data_Audit_Gabungan.ipynb` | Done | Analyzes raw video dataset, generates reports |
| `03_Preprocessing_Gabungan.ipynb` | Done | Extracts lip sequences → `.npy` files |
| `04_*` (optical flow) | Not started | Compute Lucas-Kanade optical flow |
| `05_*` (training) | Not started | Model training |

## Architecture & Data Pipeline

```
Raw Videos (3,600 videos @ 1.8GB)
  Dependent/{train,val,test}/{class}/*.mp4
          ↓
Preprocessing (Notebook 03)
  MediaPipe FaceLandmarker (face_landmarker.task, 3.76GB)
  → Extract 40 lip landmarks
  → Select top 30 frames by inter-frame lip motion difference
  → Crop & resize to 96×96 pixels
          ↓
Preprocessed Sequences (3,600 .npy files @ 2.8GB)
  Lip_Reading_Gabungan/preprocessed/dependent/raw_lips/{train,val,test}/{class}/*.npy
  Shape per file: (30, 96, 96, 3)  — (frames, height, width, channels)
          ↓
[Next] Optical Flow → training input shape: (batch, 3, 30, 96, 96)
          ↓
[Next] Model → 18-class classification
```

**Data split:** Train 140/Val 40/Test 20 videos per label × 18 labels = 3,600 total (perfectly balanced).

## Central Configuration

All paths and hyperparameters are in `Lip_Reading_Gabungan/config.json`. Key training params:

```json
{
  "training": {
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 0.001,
    "num_workers": 2,
    "sequence_length": 30,
    "input_size": 64,
    "num_classes": 18
  },
  "optical_flow": { "method": "Lucas-Kanade", "normalization": "global" }
}
```

Note: `input_size` in config is 64 but actual preprocessed output is 96×96 — verify before training.

## Key Paths

- Raw videos: `Dependent/`
- Preprocessed lip sequences: `Lip_Reading_Gabungan/preprocessed/dependent/raw_lips/`
- Optical flow output (next phase): `Lip_Reading_Gabungan/preprocessed/dependent/optical_flows/`
- Model checkpoints: `Lip_Reading_Gabungan/models/`
- Training logs: `Lip_Reading_Gabungan/logs/`
- Analysis results/plots: `Lip_Reading_Gabungan/results/dependent/`
- MediaPipe model: `face_landmarker.task` (project root)

## Current Status

Completed: environment setup, data audit, preprocessing (100% success rate).  
Not started: optical flow computation, model architecture, training, evaluation.  
`models/` directories are empty — no trained models yet.
