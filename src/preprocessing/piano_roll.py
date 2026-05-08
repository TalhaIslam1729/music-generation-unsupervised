import os
import numpy as np
import pretty_midi

from tqdm import tqdm

from src.config import (
    pitch_low,
    pitch_high,
    n_pitches,
    seq_len,
    fs,
    sparsity_threshold,
    PROCESSED_DIR
)

def midi_to_piano_roll(midi_path,frame_rate=fs):

    midi = pretty_midi.PrettyMIDI(midi_path)

    piano_roll = midi.get_piano_roll(
        fs=frame_rate
    )

    piano_roll = piano_roll[
        pitch_low : pitch_high + 1
    ]

    #Binarize
    piano_roll = (
        piano_roll > 0
    ).astype(np.float32)

    #(88,T) to (T,88)
    piano_roll = piano_roll.T

    return piano_roll

def compute_sparsity(window):

    total = window.size

    active = np.count_nonzero(window)

    sparsity = 1.0 - (active / total)

    return sparsity


def segment_piano_roll(piano_roll,window_size=seq_len,sparse_threshold=sparsity_threshold):


    T = piano_roll.shape[0]

    windows = []

    for start in range(
        0,
        T - window_size,
        window_size
    ):

        end = start + window_size

        window = piano_roll[start:end]

        if window.shape[0] != window_size:
            continue

        sparsity = compute_sparsity(window)

        if sparsity < (1.0 - sparse_threshold):

            windows.append(window)

    if len(windows) == 0:

        return np.empty(
            (0, window_size, n_pitches),
            dtype=np.float32
        )

    return np.stack(windows)

def piano_roll_to_midi(piano_roll,frame_rate=fs,velocity=80,tempo=120.0):

    midi = pretty_midi.PrettyMIDI(
        initial_tempo=tempo
    )

    instrument = pretty_midi.Instrument(
        program=0
    )

    dt = 1.0 / frame_rate

    piano_roll = (
        piano_roll > 0.5
    ).astype(np.int32)

    for pitch_idx in range(n_pitches):

        pitch = pitch_idx + pitch_low

        active = piano_roll[:, pitch_idx]

        changes = np.diff(
            np.concatenate([
                [0],
                active,
                [0]
            ])
        )

        note_starts = np.where(changes == 1)[0]

        note_ends = np.where(changes == -1)[0]

        for start, end in zip(
            note_starts,
            note_ends
        ):

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start * dt,
                end=end * dt
            )

            instrument.notes.append(note)

    midi.instruments.append(instrument)

    return midi


def build_piano_roll_dataset(midi_paths,save_path=None):

    all_windows = []

    for path in tqdm(
        midi_paths,
        desc="Building Piano-Roll Dataset"
    ):

        try:

            piano_roll = midi_to_piano_roll(path)

            windows = segment_piano_roll(
                piano_roll
            )

            if len(windows) > 0:

                all_windows.append(windows)

        except Exception as e:

            print(f"[WARNING] {path}")

            print(e)

    if len(all_windows) == 0:

        dataset = np.empty(
            (0, seq_len, n_pitches),
            dtype=np.float32
        )

    else:

        dataset = np.concatenate(
            all_windows,
            axis=0
        )

    print("\nDataset Shape:")

    print(dataset.shape)

    if save_path is not None:

        np.save(save_path, dataset)

        print(f"\nSaved -> {save_path}")

    return dataset
