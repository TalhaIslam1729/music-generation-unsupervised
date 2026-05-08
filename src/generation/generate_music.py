
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.config import (
    genres,
    GENERATED_MIDI_DIR,
    OUTPUT_DIR,
    device
)

from src.generation.midi_export import (
    export_piano_roll_batch,
    export_token_batch
)


def _load_ae(checkpoint_path=None):

    from src.generation.sample_latent import load_autoencoder

    return load_autoencoder(checkpoint_path)


def _load_vae(checkpoint_path=None):

    from src.generation.sample_latent import load_vae

    return load_vae(checkpoint_path)


def _load_transformer(checkpoint_path=None):

    from src.models.transformer import MusicTransformer

    if checkpoint_path is None:
        checkpoint_path = OUTPUT_DIR / "checkpoints" / "transformer" / "best.pt"

    model = MusicTransformer().to(device)

    ckpt  = torch.load(str(checkpoint_path), map_location=device)

    state = ckpt.get("model_state", ckpt)

    model.load_state_dict(state)

    model.eval()

    print(f"[INFO] Loaded Transformer checkpoint: {checkpoint_path}")

    return model


def generate_ae(
    n_samples=10,
    temperature=1.0,
    threshold=0.35,
    checkpoint_path=None,
    output_dir=GENERATED_MIDI_DIR,
    prefix="ae"
):

    from src.generation.sample_latent import sample_from_ae

    model = _load_ae(checkpoint_path)

    print(f"\nGenerating {n_samples} samples from LSTM Autoencoder...")

    piano_rolls = sample_from_ae(
        model,
        n_samples=n_samples,
        threshold=threshold,
        temperature=temperature
    )

    saved = export_piano_roll_batch(
        piano_rolls,
        output_dir=output_dir,
        prefix=prefix
    )

    print(f"\n[AE] {len(saved)} MIDI files saved to {output_dir}")

    return saved


def generate_vae(
    n_samples=10,
    temperature=1.0,
    threshold=0.35,
    checkpoint_path=None,
    output_dir=GENERATED_MIDI_DIR,
    prefix="vae"
):

    from src.generation.sample_latent import sample_from_vae

    model = _load_vae(checkpoint_path)

    print(f"\nGenerating {n_samples} samples from MusicVAE...")

    piano_rolls = sample_from_vae(
        model,
        n_samples=n_samples,
        threshold=threshold,
        temperature=temperature
    )

    saved = export_piano_roll_batch(
        piano_rolls,
        output_dir=output_dir,
        prefix=prefix
    )

    print(f"\n[VAE] {len(saved)} MIDI files saved to {output_dir}")

    return saved


def generate_transformer(
    n_samples=5,
    genre_id=0,
    temperature=1.0,
    max_length=512,
    checkpoint_path=None,
    output_dir=GENERATED_MIDI_DIR,
    prefix="transformer"
):

    model = _load_transformer(checkpoint_path)

    print(
        f"\nGenerating {n_samples} sequences from MusicTransformer "
        f"(genre={genres[genre_id]}, temperature={temperature})..."
    )

    token_seqs = []

    for i in range(n_samples):

        tokens = model.generate(
            genre_id=genre_id,
            max_length=max_length,
            temperature=temperature,
            device=device
        )

        token_seqs.append(tokens)

        print(f"  [{i+1}/{n_samples}] length={len(tokens)}")

    saved = export_token_batch(
        token_seqs,
        output_dir=output_dir,
        prefix=prefix
    )

    print(f"\n[Transformer] {len(saved)} MIDI files saved to {output_dir}")

    return saved


def generate_markov(
    n_samples=5,
    n_notes=300,
    midi_paths=None,
    output_dir=GENERATED_MIDI_DIR,
    prefix="markov"
):

    from collections import defaultdict, Counter
    import pretty_midi

    from src.config import RAW_MIDI_DIR, pitch_low, pitch_high

    # inline Markov (avoids importing the notebook)
    transitions  = defaultdict(Counter)
    duration_pool = []
    pitch_vocab  = list(range(pitch_low, pitch_high + 1))

    if midi_paths is None:

        import glob

        midi_paths = glob.glob(
            str(RAW_MIDI_DIR / "**/*.mid"),
            recursive=True
        )

    print(f"\nTraining Markov chain on {len(midi_paths)} files...")

    for path in midi_paths:

        try:

            midi = pretty_midi.PrettyMIDI(path)

            for inst in midi.instruments:

                if inst.is_drum:
                    continue

                notes = sorted(inst.notes, key=lambda n: n.start)

                pitches = [n.pitch for n in notes]

                duration_pool.extend(n.end - n.start for n in notes)

                for i in range(len(pitches) - 1):

                    transitions[pitches[i]][pitches[i + 1]] += 1

        except Exception as e:

            print(f"[WARNING] {path}: {e}")

    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    saved = []

    for i in range(n_samples):

        midi_out   = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        current_pitch = random.choice(pitch_vocab)
        current_time  = 0.0

        for _ in range(n_notes):

            dur = random.choice(duration_pool) if duration_pool else 0.25

            instrument.notes.append(
                pretty_midi.Note(
                    velocity=80,
                    pitch=current_pitch,
                    start=current_time,
                    end=current_time + dur
                )
            )

            current_time += dur


            nexts = transitions[current_pitch]

            if nexts:

                ps     = list(nexts.keys())
                counts = np.array(list(nexts.values()), dtype=np.float32)
                probs  = counts / counts.sum()

                current_pitch = np.random.choice(ps, p=probs)

            else:

                current_pitch = random.choice(pitch_vocab)

        midi_out.instruments.append(instrument)

        path = output_dir / f"{prefix}_{i:04d}.mid"

        midi_out.write(str(path))

        print(f"[INFO] Saved: {path}")

        saved.append(path)

    return saved

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate music from a trained model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ae", "vae", "transformer", "markov"],
        help="Which model to use for generation."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of samples / files to generate."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (AE / VAE / Transformer)."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Binarization threshold for piano-roll models (AE / VAE)."
    )

    parser.add_argument(
        "--genre",
        type=int,
        default=0,
        help=f"Genre index for Transformer (0–{len(genres)-1}): {genres}."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token sequence length for Transformer."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Uses default if not specified."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(GENERATED_MIDI_DIR),
        help="Directory for generated MIDI files."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    out_dir = Path(args.output_dir)

    if args.model == "ae":

        generate_ae(
            n_samples=args.n,
            temperature=args.temperature,
            threshold=args.threshold,
            checkpoint_path=args.checkpoint,
            output_dir=out_dir / "ae"
        )

    elif args.model == "vae":

        generate_vae(
            n_samples=args.n,
            temperature=args.temperature,
            threshold=args.threshold,
            checkpoint_path=args.checkpoint,
            output_dir=out_dir / "vae"
        )

    elif args.model == "transformer":

        generate_transformer(
            n_samples=args.n,
            genre_id=args.genre,
            temperature=args.temperature,
            max_length=args.max_length,
            checkpoint_path=args.checkpoint,
            output_dir=out_dir / "transformer"
        )

    elif args.model == "markov":

        generate_markov(
            n_samples=args.n,
            output_dir=out_dir / "markov"
        )
