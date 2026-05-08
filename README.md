
## Project Overview

This repository implements four progressive models for unsupervised music generation from MIDI data:

| Task | Model | Difficulty | Key Loss |
|------|-------|-----------|----------|
| 1 | LSTM Autoencoder | Easy | Focal |
| 2 | β-VAE | Medium | Focal |
| 3 | Transformer Decoder | Hard | Cross-entropy |
| 4 | RLHF Fine-tuning | Advanced | REINFORCE|

## Setup

```bash

git clone <your-repo-url>
cd music-generation-unsupervised
pip install -r requirements.txt


```

---

## Quick Start

# --- Task 1: Train AE ---
python -m src.training.train_ae

# --- Task 2: Train β-VAE ---
python -m src.training.train_vae

# --- Task 3: Train Transformer Decoder ---
python -m src.training.train_transformer

# --- Generation---
python src/generation/generate_music.py --model ae          --n 5
python src/generation/generate_music.py --model vae         --n 8
python src/generation/generate_music.py --model transformer --n 10 --genre 0
python src/generation/generate_music.py --model markov      --n 5

# --- Evaluation ---
python -m src.evaluation.metrics

## Repository Structure

```
music-generation-unsupervised/
├── requirements.txt
├── data/
│   ├── raw_midi/                  ← Put MAESTRO .mid files here
│   ├── processed/                 ← Auto-generated numpy arrays
│   └── train_test_split/          ← Auto-generated train/val splits
├── src/
│   ├── config.py                  ← All hyperparameters
│   ├── preprocessing/
│   │   ├── midi_parser.py         ← MIDI → note lists
│   │   ├── piano_roll.py          ← Piano-roll (Tasks 1, 2)
│   │   └── tokenizer.py           ← Event tokens  (Tasks 3,4)
│   ├── models/
│   │   ├── autoencoder.py         ← Task 1: LSTM AE
│   │   ├── vae.py                 ← Task 2: β-VAE
│   │   ├── transformer.py         ← Task 3: Transformer decoder
│   │   └── diffusion.py           ← Task 4: RLHF (RewardModel + REINFORCE)
│   ├── training/
│   │   ├── train_ae.py            ← Task 1 training loop
│   │   ├── train_vae.py           ← Task 2 training loop (KL annealing)
│   │   └── train_transformer.py   ← Task 3 training loop
│   ├── evaluation/
│   │   ├── metrics.py             ← All evaluation metrics + plotting helpers
│   │   ├── pitch_histogram.py     ← Pitch-class histogram utilities
│   │   └── rhythm_score.py        ← Rhythm diversity
│   └── generation/
│       ├── generate_music.py      ← CLI entry point for all models
│       ├── midi_export.py         ← Piano-roll / token → MIDI file helpers
│       └── sample_latent.py       ← Latent sampling, interpolation, perturbation
└── outputs/
    ├── checkpoints/               ← Saved model weights
    ├── generated_midis/           ← All MIDI output files
    ├── plots/                     ← Loss curves & evaluation plots
    └── survey_results/            ← Human listening CSV 
```

