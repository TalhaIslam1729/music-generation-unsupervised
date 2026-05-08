import os
import numpy as np
import pretty_midi

from pathlib import Path

from src.config import (
    pitch_low,
    n_pitches,
    fs,
    GENERATED_MIDI_DIR
)


def piano_roll_to_midi(
    piano_roll,
    frame_rate=fs,
    velocity=80,
    tempo=120.0):

    midi       = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)

    dt = 1.0 / frame_rate

    roll = (piano_roll > 0.5).astype(np.int32)

    for pitch_idx in range(n_pitches):

        pitch  = pitch_idx + pitch_low
        active = roll[:, pitch_idx]

        changes = np.diff(
            np.concatenate([[0], active, [0]])
        )

        starts = np.where(changes ==  1)[0]
        ends   = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start * dt,
                end=end   * dt
            )

            instrument.notes.append(note)

    midi.instruments.append(instrument)

    return midi


def export_piano_roll(
    piano_roll,
    output_path,
    frame_rate=fs,
    velocity=80,
    tempo=120.0
):

    midi = piano_roll_to_midi(
        piano_roll,
        frame_rate=frame_rate,
        velocity=velocity,
        tempo=tempo
    )

    output_path = Path(output_path)

    os.makedirs(output_path.parent, exist_ok=True)

    midi.write(str(output_path))

    print(f"[INFO] Saved MIDI: {output_path}")

    return output_path


def export_piano_roll_batch(
    piano_rolls,
    output_dir=GENERATED_MIDI_DIR,
    prefix="sample",
    frame_rate=fs,
    velocity=80,
    tempo=120.0
):

    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for i, roll in enumerate(piano_rolls):

        path = output_dir / f"{prefix}_{i:04d}.mid"

        export_piano_roll(
            roll,
            path,
            frame_rate=frame_rate,
            velocity=velocity,
            tempo=tempo
        )

        saved_paths.append(path)

    return saved_paths

def tokens_to_midi(
    token_ids,
    output_path=None
):

    from src.preprocessing.tokenizer import tokens_to_midi as _tokens_to_midi

    return _tokens_to_midi(token_ids, output_path=output_path)


def export_token_batch(
    token_sequences,
    output_dir=GENERATED_MIDI_DIR,
    prefix="transformer"
):
    """
    Export a list of token sequences to individual MIDI files.

    Returns
    -------
    list of Path
    """

    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for i, tokens in enumerate(token_sequences):

        path = output_dir / f"{prefix}_{i:04d}.mid"

        try:

            tokens_to_midi(tokens, output_path=str(path))

            saved_paths.append(path)

        except Exception as e:

            print(f"[WARNING] Could not export token sequence {i}: {e}")

    return saved_paths

def stitch_piano_rolls(
    segments,
    frame_rate=fs,
    velocity=80,
    tempo=120.0
):

    combined = np.concatenate(segments, axis=0)

    return piano_roll_to_midi(
        combined,
        frame_rate=frame_rate,
        velocity=velocity,
        tempo=tempo
    )

def stitch_and_save(
    segments,
    output_path,
    frame_rate=fs,
    velocity=80,
    tempo=120.0
):

    midi = stitch_piano_rolls(
        segments,
        frame_rate=frame_rate,
        velocity=velocity,
        tempo=tempo
    )

    output_path = Path(output_path)

    os.makedirs(output_path.parent, exist_ok=True)

    midi.write(str(output_path))

    print(f"[INFO] Stitched MIDI saved: {output_path}")

    return output_path


if __name__ == "__main__":

    dummy_roll = (np.random.rand(64, n_pitches) > 0.9).astype(np.float32)

    out = export_piano_roll(
        dummy_roll,
        output_path=GENERATED_MIDI_DIR / "test_export.mid"
    )

    print(f"\nSmoke-test output: {out}")
