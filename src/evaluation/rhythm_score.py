import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

from collections import Counter

from src.config import PLOTS_DIR



def extract_inter_onset_intervals(midi_path):

    midi = pretty_midi.PrettyMIDI(midi_path)

    onsets = []

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        for note in inst.notes:

            onsets.append(note.start)

    if len(onsets) < 2:
        return np.array([])

    onsets = np.sort(onsets)

    return np.diff(onsets)


def quantize_duration(duration_sec, quantize_ms=50.0):

    dur_ms = duration_sec * 1000.0

    return round(dur_ms / quantize_ms) * quantize_ms


def extract_note_durations(midi_path, quantize_ms=50.0):

    midi = pretty_midi.PrettyMIDI(midi_path)

    durations = []

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        for note in inst.notes:

            dur = (note.end - note.start)

            durations.append(
                quantize_duration(dur, quantize_ms)
            )

    return durations


def extract_pitch_sequence(midi_path):

    midi = pretty_midi.PrettyMIDI(midi_path)

    notes = []

    for inst in midi.instruments:

        if inst.is_drum:
            continue

        notes.extend(inst.notes)

    notes.sort(key=lambda n: n.start)

    return [n.pitch for n in notes]


def rhythm_diversity_score(midi_path, quantize_ms=50.0):

    durations = extract_note_durations(midi_path, quantize_ms)

    if not durations:
        return 0.0

    return float(len(set(durations)) / len(durations))


def repetition_ratio(midi_path, n=4):

    pitches = extract_pitch_sequence(midi_path)

    if len(pitches) < n + 1:
        return 0.0

    ngrams = [
        tuple(pitches[i: i + n])
        for i in range(len(pitches) - n + 1)
    ]

    counts  = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)

    return float(repeated / len(ngrams))


def groove_consistency_score(midi_path, quantize_ms=50.0):

    iois = extract_inter_onset_intervals(midi_path)

    if len(iois) == 0:
        return 0.0

    quantized = np.array([
        quantize_duration(ioi, quantize_ms)
        for ioi in iois
    ])

    counts  = Counter(quantized.tolist())
    mode_ioi, mode_count = counts.most_common(1)[0]

    return float(mode_count / len(quantized))


def note_density(midi_path):

    midi = pretty_midi.PrettyMIDI(midi_path)

    total_notes = 0

    for inst in midi.instruments:

        if not inst.is_drum:

            total_notes += len(inst.notes)

    duration = midi.get_end_time()

    if duration == 0:
        return 0.0

    return float(total_notes / duration)


def evaluate_rhythm(midi_path, quantize_ms=50.0, ngram_n=4):

    return {
        "rhythm_diversity":   rhythm_diversity_score(midi_path, quantize_ms),
        "repetition_ratio":   repetition_ratio(midi_path, ngram_n),
        "groove_consistency": groove_consistency_score(midi_path, quantize_ms),
        "note_density":       note_density(midi_path)
    }


def batch_rhythm_scores(midi_paths, quantize_ms=50.0, ngram_n=4):


    keys    = ["rhythm_diversity", "repetition_ratio",
               "groove_consistency", "note_density"]
    buckets = {k: [] for k in keys}

    for path in midi_paths:

        try:

            scores = evaluate_rhythm(path, quantize_ms, ngram_n)

            for k in keys:

                buckets[k].append(scores[k])

        except Exception as e:

            print(f"[WARNING] {path}: {e}")

    results = {}

    for k, values in buckets.items():

        if values:

            results[k] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
                "all":  values
            }

        else:

            results[k] = {"mean": np.nan, "std": np.nan, "all": []}

    return results

def plot_ioi_distribution(
    midi_path,
    label="",
    save_name=None,
    ax=None
):

    iois = extract_inter_onset_intervals(midi_path)

    if len(iois) == 0:
        print("[WARNING] No IOIs found.")
        return

    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        iois,
        bins=50,
        color="salmon",
        edgecolor="black",
        linewidth=0.4
    )

    ax.set_xlabel("Inter-Onset Interval (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"IOI Distribution  {label}")

    if standalone:

        plt.tight_layout()

        if save_name is not None:

            save_path = PLOTS_DIR / save_name

            plt.savefig(save_path, dpi=150)

            print(f"[INFO] Saved: {save_path}")

        plt.close()


def plot_duration_distribution(
    midi_path,
    quantize_ms=50.0,
    label="",
    save_name=None
):

    durations = extract_note_durations(midi_path, quantize_ms)

    if not durations:
        print("[WARNING] No durations found.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    counts = Counter(durations)

    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    ax.bar(
        [str(int(v)) for v in x],
        y,
        color="cornflowerblue",
        edgecolor="black",
        linewidth=0.4
    )

    ax.set_xlabel("Quantized Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Note Duration Distribution  {label}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_name is not None:

        save_path = PLOTS_DIR / save_name

        plt.savefig(save_path, dpi=150)

        print(f"[INFO] Saved: {save_path}")

    plt.close()


def plot_rhythm_comparison(
    gen_paths,
    ref_paths,
    save_name="rhythm_comparison.png"
):
    """Bar-chart comparing mean rhythm scores for generated vs reference."""

    gen_scores = batch_rhythm_scores(gen_paths)
    ref_scores = batch_rhythm_scores(ref_paths)

    metrics = ["rhythm_diversity", "groove_consistency", "note_density"]
    labels  = ["Rhythm Diversity", "Groove Consistency", "Note Density"]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_ref = ax.bar(
        x - width / 2,
        [ref_scores[m]["mean"] for m in metrics],
        width,
        label="Reference",
        color="steelblue"
    )

    bars_gen = ax.bar(
        x + width / 2,
        [gen_scores[m]["mean"] for m in metrics],
        width,
        label="Generated",
        color="tomato"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Rhythm Metrics: Generated vs Reference")
    ax.legend()

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
    )[:30]

    gen_paths = glob.glob(
        str(GENERATED_MIDI_DIR / "*.mid"),
        recursive=True
    )

    if ref_paths:

        print("\nReference rhythm scores:")

        ref_results = batch_rhythm_scores(ref_paths)

        for metric, stats in ref_results.items():

            print(f"  {metric}: mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    if gen_paths:

        print("\nGenerated rhythm scores:")

        gen_results = batch_rhythm_scores(gen_paths)

        for metric, stats in gen_results.items():

            print(f"  {metric}: mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    if ref_paths and gen_paths:

        plot_rhythm_comparison(gen_paths, ref_paths)
