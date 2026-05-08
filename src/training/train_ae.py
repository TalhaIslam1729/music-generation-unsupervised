import os
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import (
    ae_epochs,
    ae_batch_size,
    ae_lr,
    PROCESSED_DIR,
    TRAIN_TEST_SPLIT_DIR,
    OUTPUT_DIR,
    device
)

from src.models.autoencoder import (
    LSTMAutoencoder,
    FocalLoss
)

from src.preprocessing.piano_roll import (
    piano_roll_to_midi
)

def load_piano_roll_split(split_dir=TRAIN_TEST_SPLIT_DIR):

    train_path = split_dir / "train.npy"
    val_path   = split_dir / "val.npy"

    if train_path.exists() and val_path.exists():

        train_data = np.load(str(train_path))
        val_data   = np.load(str(val_path))

    else:

        fallback = PROCESSED_DIR / "piano_roll.npy"

        if not fallback.exists():
            raise FileNotFoundError(
                f"No dataset found. Expected {train_path} or {fallback}"
            )

        data  = np.load(str(fallback))
        split = int(len(data) * 0.8)

        train_data = data[:split]
        val_data   = data[split:]

    return train_data, val_data


def make_dataloader(data, batch_size, shuffle=True):

    tensor = torch.tensor(
        data,
        dtype=torch.float32
    )

    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device == "cuda")
    )



def train_one_epoch(model, loader, optimizer, criterion, epoch):

    model.train()

    total_loss = 0.0

    for (batch,) in tqdm(
        loader,
        desc=f"Epoch {epoch}",
        leave=False
    ):

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed, _ = model(batch)

        loss = criterion(reconstructed, batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    for (batch,) in loader:

        batch = batch.to(device)

        reconstructed, _ = model(batch)

        loss = criterion(reconstructed, batch)

        total_loss += loss.item()

    return total_loss / len(loader)


def train_autoencoder(
    epochs=ae_epochs,
    batch_size=ae_batch_size,
    lr=ae_lr,
    checkpoint_dir=None
):

    if checkpoint_dir is None:
        checkpoint_dir = OUTPUT_DIR / "checkpoints" / "autoencoder"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("\nLoading piano-roll dataset...")

    train_data, val_data = load_piano_roll_split()

    print(f"  Train : {train_data.shape}")
    print(f"  Val   : {val_data.shape}")

    train_loader = make_dataloader(train_data, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_data,   batch_size, shuffle=False)

    # ----------------------------------------------------------------- model
    model     = LSTMAutoencoder().to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    print(f"\nTraining LSTMAutoencoder on {device} for {epochs} epochs\n")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        val_loss = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss
                },
                str(checkpoint_dir / "best.pt")
            )

    # final checkpoint
    torch.save(
        model.state_dict(),
        str(checkpoint_dir / "final.pt")
    )

    print(f"\nCheckpoints saved to {checkpoint_dir}")
    print(f"Best val loss: {best_val_loss:.5f}")

    return model, history


if __name__ == "__main__":

    model, history = train_autoencoder()
