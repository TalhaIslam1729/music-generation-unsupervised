import os
from pathlib import Path
import torch
import random
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_MIDI_DIR = DATA_DIR / "raw_midi"

PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_TEST_SPLIT_DIR = DATA_DIR / "train_test_split"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

GENERATED_MIDI_DIR = OUTPUT_DIR / "generated_midis"

PLOTS_DIR = OUTPUT_DIR / "plots"

SURVEY_DIR = OUTPUT_DIR / "survey_results"

REPORT_DIR = PROJECT_ROOT / "report"

ARCHITECTURE_DIR = PROJECT_ROOT / "architecture_diagrams"



steps_per_bar=16
bars_per_segment=4
seq_len=steps_per_bar * bars_per_segment
pitch_low=21
pitch_high=108 
n_pitches=pitch_high - pitch_low + 1
velocity_bins=32
fs=16  
sparsity_threshold = 0.95
   
#tokeniser
vocab_size= 512
pad_token= 0
bos_token= 1
eos_token= 2

#genres
genres = ["2004","2006","2008","2009","2011","2013","2014","2015","2017","2018"]
n_genres = len(genres)

#lstm autoencoder
ae_hidden_dim= 256
ae_latent_dim= 64
ae_num_layers= 2
ae_dropout= 0.3
ae_epochs= 40
ae_batch_size= 64
ae_lr= 0.001

#vae
vae_hidden_dim= 256
vae_latent_dim= 128
vae_num_layers= 2
vae_beta= 1.0           
vae_epochs= 40
vae_batch_size= 64
vae_lr= 0.001

#transformer
tf_d_model= 256
tf_n_heads= 8
tf_num_layers= 6
tf_d_ff= 1024
tf_dropout= 0.1
tf_max_seq_len= 512
tf_epochs= 30
tf_batch_size= 32
tf_lr= 0.001

#rlhf
rl_steps= 200
rl_lr= 0.001
rl_gamma= 0.99
rl_survey_path = os.path.join(SURVEY_DIR, "survey_results.csv")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

