import os
import glob
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import (
    tf_epochs,
    tf_batch_size,
    tf_lr,
    pad_token,
    genres,
    TRAIN_TEST_SPLIT_DIR,
    RAW_MIDI_DIR,
    OUTPUT_DIR,
    device
)

from src.models.transformer import MusicTransformer

from src.preprocessing.tokenizer import (
    TokenDataset,
    collate_fn,
    build_token_dataset
)


def build_genre_midi_map(root_dir=RAW_MIDI_DIR):

    genre_map = {}

    for genre in genres:

        pattern = str(root_dir / "**" / genre / "**" / "*.mid")

        paths = glob.glob(pattern, recursive=True)

        if not paths:
            pattern = str(root_dir / genre / "**" / "*.mid")
            paths   = glob.glob(pattern, recursive=True)

        genre_map[genre] = paths

    return genre_map


def build_genre_dataset(root_dir=RAW_MIDI_DIR):

    genre_map = build_genre_midi_map(root_dir)

    all_paths  = []
    all_genres = []

    for genre_id, genre in enumerate(genres):

        for path in genre_map.get(genre, []):

            all_paths.append(path)
            all_genres.append(genre_id)

    return all_paths, all_genres


class GenreTokenDataset(TokenDataset):

    def __init__(self, midi_paths, genre_ids, max_len=512):

        super().__init__(midi_paths, max_len=max_len)

        self.genre_ids = []

        surviving = []

        for idx, path in enumerate(midi_paths):

            try:

                from src.preprocessing.tokenizer import midi_to_tokens

                midi_to_tokens(path, max_len=max_len)

                surviving.append(idx)

            except Exception:

                pass

        for idx in surviving:

            self.genre_ids.append(genre_ids[idx])

    def __getitem__(self, idx):

        x, y = super().__getitem__(idx)

        genre = torch.tensor(
            self.genre_ids[idx],
            dtype=torch.long
        )

        return x, y, genre


def genre_collate_fn(batch):

    inputs  = [x        for x, _, _ in batch]
    targets = [y        for _, y, _ in batch]
    gids    = [g        for _, _, g in batch]

    from torch.nn.utils.rnn import pad_sequence

    padded_inputs  = pad_sequence(inputs,  batch_first=True, padding_value=pad_token)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_token)
    genre_ids      = torch.stack(gids)
    attention_mask = (padded_inputs != pad_token).long()

    return {
        "input_ids":      padded_inputs,
        "target_ids":     padded_targets,
        "genre_ids":      genre_ids,
        "attention_mask": attention_mask
    }


def train_one_epoch(model, loader, optimizer, epoch):

    model.train()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc=f"Epoch {epoch}",
        leave=False
    ):

        input_ids  = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        genre_ids  = batch["genre_ids"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, genre_ids)

        loss = model.compute_loss(logits, target_ids)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    for batch in loader:

        input_ids  = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        genre_ids  = batch["genre_ids"].to(device)

        logits = model(input_ids, genre_ids)

        loss = model.compute_loss(logits, target_ids)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    perplexity = float(np.exp(avg_loss))

    return avg_loss, perplexity


def train_transformer(
    epochs=tf_epochs,
    batch_size=tf_batch_size,
    lr=tf_lr,
    val_split=0.1,
    checkpoint_dir=None
):

    if checkpoint_dir is None:
        checkpoint_dir = OUTPUT_DIR / "checkpoints" / "transformer"

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nBuilding genre-aware token dataset...")

    midi_paths, genre_ids = build_genre_dataset()

    if len(midi_paths) == 0:
        raise RuntimeError(
            f"No MIDI files found under {RAW_MIDI_DIR}. "
            "Check your genre folder names in config.genres."
        )

    print(f"  Total files: {len(midi_paths)}")

    dataset = GenreTokenDataset(midi_paths, genre_ids)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=genre_collate_fn,
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=genre_collate_fn,
        pin_memory=(device == "cuda")
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    model     = MusicTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    print(f"\nTraining MusicTransformer on {device} for {epochs} epochs\n")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, epoch
        )

        val_loss, val_ppl = evaluate(model, val_loader)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl
                },
                str(checkpoint_dir / "best.pt")
            )

    torch.save(
        model.state_dict(),
        str(checkpoint_dir / "final.pt")
    )

    print(f"\nCheckpoints saved to {checkpoint_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")

    return model, history


if __name__ == "__main__":

    model, history = train_transformer()
