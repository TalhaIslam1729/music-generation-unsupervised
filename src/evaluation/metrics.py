# src/evaluation/metrics.py

import os
import csv
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import torch

from tqdm import tqdm

from src.config import (
    pitch_low,
    pitch_high,
    PLOTS_DIR,
    GENERATED_MIDI_DIR,
    rl_survey_path
)

from src.preprocessing.midi_parser import (
    parse_dataset,
    load_midi
)

from src.preprocessing.piano_roll import (
    midi_to_piano_roll
)


def plot_duration_histogram(dataset):

    durations = [
        sample["duration"] / 60.0
        for sample in dataset
    ]

    plt.figure(figsize=(10, 5))

    plt.hist(
        durations,
        bins=40
    )

    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Number of Pieces")

    plt.title("Piece Duration Distribution")

    save_path = PLOTS_DIR / "duration_histogram.png"

    plt.savefig(save_path)

    plt.close()

    print(f"[INFO] Saved: {save_path}")


def plot_note_count_distribution(dataset):

    note_counts = [
        len(sample["notes"])
        for sample in dataset
    ]

    plt.figure(figsize=(10, 5))

    plt.hist(
        note_counts,
        bins=40
    )

    plt.xlabel("Number of Notes")
    plt.ylabel("Number of Pieces")

    plt.title("Note Count Distribution")

    save_path = PLOTS_DIR / "note_count_distribution.png"

    plt.savefig(save_path)

    plt.close()

    print(f"[INFO] Saved: {save_path}")


def plot_pitch_distribution(dataset):

    pitch_counts = np.zeros(88)

    for sample in tqdm(
        dataset,
        desc="Counting Pitches"
    ):

        for note in sample["notes"]:

            idx = note["pitch"] - pitch_low

            pitch_counts[idx] += 1

    pitches = np.arange(
        pitch_low,
        pitch_high + 1
    )

    plt.figure(figsize=(14, 5))

    plt.bar(
        pitches,
        pitch_counts
    )

    plt.xlabel("MIDI Pitch")
    plt.ylabel("Frequency")

    plt.title("Pitch Distribution")

    save_path = PLOTS_DIR / "pitch_distribution.png"

    plt.savefig(save_path)

    plt.close()

    print(f"[INFO] Saved: {save_path}")


def plot_velocity_distribution(dataset):

    velocities = []

    for sample in dataset:

        for note in sample["notes"]:

            velocities.append(
                note["velocity"]
            )

    plt.figure(figsize=(10, 5))

    plt.hist(
        velocities,
        bins=40
    )

    plt.xlabel("Velocity")
    plt.ylabel("Frequency")

    plt.title("Velocity Distribution")

    save_path = PLOTS_DIR / "velocity_distribution.png"

    plt.savefig(save_path)

    plt.close()

    print(f"[INFO] Saved: {save_path}")


def piano_roll_sparsity(midi_path):

    midi = load_midi(midi_path)

    piano_roll = midi_to_piano_roll(midi_path)

    total_cells = piano_roll.size

    active_cells = np.count_nonzero(
        piano_roll
    )

    sparsity = 1.0 - (
        active_cells / total_cells
    )

    print("\nPiano Roll Sparsity")

    print("-" * 40)

    print(f"Shape         : {piano_roll.shape}")

    print(f"Total Cells   : {total_cells}")

    print(f"Active Cells  : {active_cells}")

    print(f"Sparsity      : {sparsity:.4f}")

    print("-" * 40)

    return sparsity


def pitch_class_histogram(midi_path):

    midi = pretty_midi.PrettyMIDI(
        midi_path
    )

    counts = np.zeros(12)

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        for note in inst.notes:

            counts[
                note.pitch % 12
            ] += 1

    total = counts.sum()

    if total == 0:
        return counts

    return counts / total


def pitch_histogram_similarity(
    gen_path,
    ref_path
):

    p = pitch_class_histogram(
        ref_path
    )

    q = pitch_class_histogram(
        gen_path
    )

    return float(
        np.abs(p - q).sum()
    )


def batch_pitch_similarity(
    gen_paths,
    ref_paths
):

    scores = []

    for gen_path in gen_paths:

        ref_path = np.random.choice(
            ref_paths
        )

        try:

            score = pitch_histogram_similarity(
                gen_path,
                ref_path
            )

            scores.append(score)

        except Exception as e:

            print(f"[WARNING] {e}")

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "all": scores
    }


def rhythm_diversity_score(
    midi_path,
    quantize_ms=50.0
):

    midi = pretty_midi.PrettyMIDI(
        midi_path
    )

    durations = []

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        for note in inst.notes:

            dur_ms = (
                note.end - note.start
            ) * 1000.0

            quantized = round(
                dur_ms / quantize_ms
            ) * quantize_ms

            durations.append(
                quantized
            )

    if len(durations) == 0:
        return 0.0

    return float(
        len(set(durations)) /
        len(durations)
    )


def repetition_ratio(
    midi_path,
    n=4
):

    midi = pretty_midi.PrettyMIDI(
        midi_path
    )

    pitches = []

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        notes = sorted(
            inst.notes,
            key=lambda x: x.start
        )

        pitches.extend(
            note.pitch
            for note in notes
        )

    if len(pitches) < n + 1:
        return 0.0

    ngrams = [

        tuple(
            pitches[i:i+n]
        )

        for i in range(
            len(pitches) - n + 1
        )
    ]

    counts = Counter(ngrams)

    repeated = sum(

        1

        for count in counts.values()

        if count > 1
    )

    return float(
        repeated / len(ngrams)
    )


def reconstruction_mse(
    x,
    x_hat
):

    return float(
        np.mean(
            (x - x_hat) ** 2
        )
    )


def perplexity_from_loss(loss):

    return float(
        np.exp(loss)
    )


def evaluate_perplexity(
    model,
    dataloader,
    device="cpu"
):

    model.eval()

    total_loss = 0.0

    total_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            target_ids = batch[
                "target_ids"
            ].to(device)

            genre_ids = batch[
                "genre_ids"
            ].to(device)

            logits = model(
                input_ids,
                genre_ids
            )

            loss = model.compute_loss(
                logits,
                target_ids
            )

            total_loss += loss.item()

            total_batches += 1

    avg_loss = (
        total_loss / total_batches
    )

    ppl = perplexity_from_loss(
        avg_loss
    )

    return {
        "cross_entropy": avg_loss,
        "perplexity": ppl
    }


def genre_control_score(
    predicted_genres,
    target_genres
):

    predicted_genres = np.array(
        predicted_genres
    )

    target_genres = np.array(
        target_genres
    )

    accuracy = np.mean(
        predicted_genres ==
        target_genres
    )

    return float(accuracy)


def validate_midi(
    path,
    min_notes=50,
    min_duration=5.0
):

    try:

        midi = pretty_midi.PrettyMIDI(
            path
        )

    except Exception as e:

        print(f"[VALIDATE ERROR] {e}")

        return False

    note_count = sum(

        len(inst.notes)

        for inst in midi.instruments

        if not inst.is_drum
    )

    duration = midi.get_end_time()

    return (
        note_count >= min_notes and
        duration >= min_duration
    )


def load_human_scores(csv_path):

    scores = defaultdict(list)

    with open(csv_path) as f:

        reader = csv.DictReader(f)

        for row in reader:

            scores[
                row["filename"]
            ].append(
                float(row["score"])
            )

    return {

        k: float(np.mean(v))

        for k, v in scores.items()
    }


def mean_human_score(csv_path):

    scores = load_human_scores(
        csv_path
    )

    if len(scores) == 0:

        return float("nan")

    return float(
        np.mean(
            list(scores.values())
        )
    )

def evaluate_model(
    gen_paths,
    ref_paths=None,
    survey_csv=None,
    model_name="model",
    reconstruction_loss=np.nan,
    perplexity=np.nan,
    genre_control=np.nan
):

    results = {

        "model": model_name,

        "reconstruction_loss": reconstruction_loss,

        "perplexity": perplexity,

        "genre_control": genre_control
    }

    # --------------------------------------------------
    # Pitch Similarity
    # --------------------------------------------------

    if ref_paths:

        pitch_scores = batch_pitch_similarity(
            gen_paths,
            ref_paths
        )

        results[
            "pitch_similarity_mean"
        ] = pitch_scores["mean"]

        results[
            "pitch_similarity_std"
        ] = pitch_scores["std"]

    else:

        results[
            "pitch_similarity_mean"
        ] = np.nan

    rhythm_scores = [

        rhythm_diversity_score(path)

        for path in gen_paths

        if validate_midi(path)
    ]

    results[
        "rhythm_diversity_mean"
    ] = float(
        np.mean(rhythm_scores)
    )

    repetition_scores = [

        repetition_ratio(path)

        for path in gen_paths

        if validate_midi(path)
    ]

    results[
        "repetition_ratio_mean"
    ] = float(
        np.mean(repetition_scores)
    )

    if (
        survey_csv and
        os.path.exists(survey_csv)
    ):

        results[
            "human_score"
        ] = mean_human_score(
            survey_csv
        )

    else:

        results[
            "human_score"
        ] = np.nan

    print_results(results)

    return results


def print_results(results):

    print("\n" + "=" * 60)

    print(f"Model: {results['model']}")

    print("-" * 60)

    print(
        f"Reconstruction Loss : "
        f"{results.get('reconstruction_loss', np.nan):.4f}"
    )

    print(
        f"Perplexity          : "
        f"{results.get('perplexity', np.nan):.4f}"
    )

    print(
        f"Pitch Similarity    : "
        f"{results.get('pitch_similarity_mean', np.nan):.4f}"
    )

    print(
        f"Rhythm Diversity    : "
        f"{results.get('rhythm_diversity_mean', np.nan):.4f}"
    )

    print(
        f"Repetition Ratio    : "
        f"{results.get('repetition_ratio_mean', np.nan):.4f}"
    )

    print(
        f"Human Score         : "
        f"{results.get('human_score', np.nan):.4f}"
    )

    print(
        f"Genre Control       : "
        f"{results.get('genre_control', np.nan):.4f}"
    )

    print("=" * 60)


def build_full_results_table(
    results_list,
    save_csv=None
):

    headers = [

        "model",

        "reconstruction_loss",

        "perplexity",

        "pitch_similarity_mean",

        "rhythm_diversity_mean",

        "repetition_ratio_mean",

        "human_score",

        "genre_control"
    ]

    print("\n" + "=" * 120)

    print(

        f"{'Model':20}"
        f"{'Recon':12}"
        f"{'PPL':12}"
        f"{'PitchSim':12}"
        f"{'Rhythm':12}"
        f"{'Repeat':12}"
        f"{'Human':12}"
        f"{'Genre':12}"

    )

    print("=" * 120)

    for r in results_list:

        print(

            f"{r.get('model', ''):20}"

            f"{r.get('reconstruction_loss', np.nan):<12.4f}"

            f"{r.get('perplexity', np.nan):<12.4f}"

            f"{r.get('pitch_similarity_mean', np.nan):<12.4f}"

            f"{r.get('rhythm_diversity_mean', np.nan):<12.4f}"

            f"{r.get('repetition_ratio_mean', np.nan):<12.4f}"

            f"{r.get('human_score', np.nan):<12.4f}"

            f"{r.get('genre_control', np.nan):<12.4f}"
        )

    print("=" * 120)

    if save_csv:

        with open(
            save_csv,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=headers
            )

            writer.writeheader()

            writer.writerows(results_list)

        print(f"\n[INFO] Saved: {save_csv}")


if __name__ == "__main__":

    dataset = parse_dataset()


    plot_duration_histogram(dataset)

    plot_note_count_distribution(dataset)

    plot_pitch_distribution(dataset)

    plot_velocity_distribution(dataset)


    if len(dataset) > 0:

        piano_roll_sparsity(
            dataset[0]["path"]
        )


    generated_paths = []

    for root, _, files in os.walk(
        GENERATED_MIDI_DIR
    ):

        for file in files:

            if (
                file.endswith(".mid") or
                file.endswith(".midi")
            ):

                generated_paths.append(
                    os.path.join(root, file)
                )


    if len(generated_paths) > 0:

        results = evaluate_model(

            gen_paths=generated_paths,

            model_name="test_model",

            survey_csv=rl_survey_path
        )

        build_full_results_table(
            [results]
        )

    else:

        print(
            "\n[INFO] No generated MIDI files found."
        )