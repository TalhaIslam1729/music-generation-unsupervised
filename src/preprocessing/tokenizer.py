import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tqdm import tqdm

from miditok import REMI, TokenizerConfig
from symusic import Score

from src.config import (
    pitch_low,
    pitch_high,
    velocity_bins,
    pad_token,
    bos_token,
    eos_token,
    tf_max_seq_len,
    vocab_size
)


tokenizer_config = TokenizerConfig(

    pitch_range=(
        pitch_low,
        pitch_high + 1
    ),

    num_velocities=velocity_bins,

    special_tokens=[
        "PAD",
        "BOS",
        "EOS"
    ],

    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,

    beat_res={
        (0, 4): 8,
        (4, 12): 4
    }
)

tokenizer = REMI(tokenizer_config)

def train_bpe(
    midi_paths,
    vocab_target_size=vocab_size
):

    print("\nTraining BPE Vocabulary...")

    tokenizer.train(
        vocab_size=vocab_target_size,
        files_paths=midi_paths
    )

    print("\nBPE Training Complete")


def midi_to_tokens(midi_path,max_len=tf_max_seq_len):

    score = Score(midi_path)

    tokenized = tokenizer(score)

    tokens = tokenized.ids

    tokens = (
        [bos_token] +
        tokens +
        [eos_token]
    )

    if len(tokens) > max_len:

        tokens = tokens[:max_len]

        tokens[-1] = eos_token

    return tokens

def create_attention_mask(tokens):

    return [
        0 if tok == pad_token else 1
        for tok in tokens
    ]


def create_transformer_inputs(tokens):

    x = tokens[:-1]

    y = tokens[1:]

    return x, y


class TokenDataset(Dataset):

    def __init__(
        self,
        midi_paths,
        max_len=tf_max_seq_len
    ):

        self.samples = []

        print("\nTokenizing MIDI Files...\n")

        for path in tqdm(midi_paths):

            try:

                tokens = midi_to_tokens(
                    path,
                    max_len=max_len
                )

                self.samples.append(tokens)

            except Exception as e:

                print(f"[WARNING] {path}")

                print(e)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        tokens = self.samples[idx]

        x, y = create_transformer_inputs(
            tokens
        )

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


def collate_fn(batch):

    inputs = [x for x, _ in batch]

    targets = [y for _, y in batch]

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=pad_token
    )

    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_token
    )

    attention_mask = (
        padded_inputs != pad_token
    ).long()

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets,
        "attention_mask": attention_mask
    }

def build_token_dataset(midi_paths,max_len=tf_max_seq_len):

    dataset = TokenDataset(
        midi_paths,
        max_len=max_len
    )

    print("\nDataset Size:")

    print(len(dataset))

    return dataset


def tokens_to_midi(token_ids,output_path=None):

    token_ids = [

        int(tok)

        for tok in token_ids

        if tok != pad_token
    ]

    token_ids = [

        tok for tok in token_ids

        if tok not in (
            bos_token,
            eos_token
        )
    ]

    midi = tokenizer.decode(
        token_ids
    )

    if output_path is not None:

        midi.dump_midi(output_path)

        print(f"\nSaved MIDI -> {output_path}")

    return midi


if __name__ == "__main__":

    import glob
    from torch.utils.data import DataLoader
    from src.config import RAW_MIDI_DIR

    midi_paths = glob.glob(
        str(RAW_MIDI_DIR / "**/*.midi"),
        recursive=True
    )[:10]

    dataset = build_token_dataset(
        midi_paths
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    print("\nInput Shape:")

    print(batch["input_ids"].shape)

    print("\nTarget Shape:")

    print(batch["target_ids"].shape)

    print("\nAttention Mask Shape:")

    print(batch["attention_mask"].shape)