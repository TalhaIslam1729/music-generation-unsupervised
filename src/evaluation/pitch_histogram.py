import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

from src.config import (
    pitch_low,
    pitch_high,
    PLOTS_DIR
)


def pitch_class_histogram(midi_path):

    midi   = pretty_midi.PrettyMIDI(midi_path)
    counts = np.zeros(12, dtype=np.float64)

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        for note in inst.notes:

            counts[note.pitch % 12] += 1

    total = counts.sum()

    if total == 0:
        return counts

    return counts / total


def pitch_range_histogram(midi_path, bins=88):

    midi   = pretty_midi.PrettyMIDI(midi_path)
    counts = np.zeros(bins, dtype=np.float64)

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        for note in inst.notes:

            idx = note.pitch - pitch_low

            if 0 <= idx < bins:

                counts[idx] += 1

    total = counts.sum()

    if total == 0:
        return counts

    return counts / total


def l1_distance(p, q):

    return float(np.abs(p - q).mean())


def kl_divergence(p, q, eps=1e-8):


    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def overlap_score(p, q):


    return float(np.minimum(p, q).sum())


def pitch_histogram_similarity(gen_path, ref_path, metric="l1"):


    p = pitch_class_histogram(ref_path)
    q = pitch_class_histogram(gen_path)

    if metric == "l1":
        return l1_distance(p, q)

    if metric == "kl":
        return kl_divergence(p, q)

    if metric == "overlap":
        return overlap_score(p, q)

    raise ValueError(f"Unknown metric: {metric}")


def batch_pitch_similarity(gen_paths, ref_paths, metric="l1"):

    scores = []

    for gen_path in gen_paths:

        ref_path = np.random.choice(ref_paths)

        try:

            score = pitch_histogram_similarity(gen_path, ref_path, metric=metric)

            scores.append(score)

        except Exception as e:

            print(f"[WARNING] {gen_path}: {e}")

    if not scores:
        return {"mean": np.nan, "std": np.nan, "all": []}

    return {
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "all":  scores
    }




PITCH_CLASS_NAMES = [
    "C", "C#", "D", "D#", "E",
    "F", "F#", "G", "G#", "A", "A#", "B"
]


def plot_pitch_class_histogram(
    midi_path,
    label="",
    save_name=None,
    ax=None
):
    """Plot a single pitch-class histogram."""

    hist = pitch_class_histogram(midi_path)

    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        PITCH_CLASS_NAMES,
        hist,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(f"Pitch-Class Histogram  {label}")

    if standalone:

        plt.tight_layout()

        if save_name is not None:

            save_path = PLOTS_DIR / save_name

            plt.savefig(save_path, dpi=150)

            print(f"[INFO] Saved: {save_path}")

        plt.close()


def compare_pitch_histograms(
    gen_path,
    ref_path,
    save_name="pitch_histogram_compare.png"
):
    """Side-by-side pitch-class histogram comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

    plot_pitch_class_histogram(ref_path, label="(Reference)", ax=axes[0])
    plot_pitch_class_histogram(gen_path, label="(Generated)", ax=axes[1])

    plt.suptitle("Pitch-Class Histogram Comparison")
    plt.tight_layout()

    save_path = PLOTS_DIR / save_name

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"[INFO] Saved: {save_path}")


def plot_pitch_range_histogram(
    midi_path,
    label="",
    save_name=None
):


    hist    = pitch_range_histogram(midi_path)
    pitches = np.arange(pitch_low, pitch_high + 1)

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.bar(
        pitches,
        hist,
        color="mediumseagreen",
        width=1.0
    )

    ax.set_xlabel("MIDI Pitch")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(f"Pitch Range Histogram  {label}")

    plt.tight_layout()

    if save_name is not None:

        save_path = PLOTS_DIR / save_name

        plt.savefig(save_path, dpi=150)

        print(f"[INFO] Saved: {save_path}")

    plt.close()


def plot_batch_pitch_distribution(
    midi_paths,
    label="",
    save_name="batch_pitch_distribution.png"
):

    all_hists = []

    for path in midi_paths:

        try:

            h = pitch_class_histogram(path)

            all_hists.append(h)

        except Exception as e:

            print(f"[WARNING] {path}: {e}")

    if not all_hists:
        print("[WARNING] No valid files for batch pitch distribution.")
        return

    all_hists = np.stack(all_hists)     # (N, 12)

    mean = all_hists.mean(axis=0)
    std  = all_hists.std(axis=0)

    x = np.arange(12)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        PITCH_CLASS_NAMES,
        mean,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Mean"
    )

    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.5
    )

    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(f"Batch Pitch-Class Distribution  {label}  (n={len(all_hists)})")

    plt.tight_layout()

    save_path = PLOTS_DIR / save_name

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"[INFO] Saved: {save_path}")


if __name__ == "__main__":

    import glob
    from src.config import RAW_MIDI_DIR, GENERATED_MIDI_DIR

    ref_paths = glob.glob(
        str(RAW_MIDI_DIR / "**/*.mid"),
        recursive=True
    )[:50]

    gen_paths = glob.glob(
        str(GENERATED_MIDI_DIR / "*.mid"),
        recursive=True
    )

    if ref_paths:
        plot_batch_pitch_distribution(
            ref_paths,
            label="(Reference)",
            save_name="reference_pitch_distribution.png"
        )

    if gen_paths and ref_paths:
        compare_pitch_histograms(
            gen_paths[0],
            ref_paths[0]
        )

        results = batch_pitch_similarity(gen_paths, ref_paths, metric="l1")

        print(f"\nPitch Similarity (L1) — mean: {results['mean']:.4f}  std: {results['std']:.4f}")
