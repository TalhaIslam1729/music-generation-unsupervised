import os
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import (
    vae_epochs,
    vae_batch_size,
    vae_lr,
    vae_beta,
    PROCESSED_DIR,
    TRAIN_TEST_SPLIT_DIR,
    OUTPUT_DIR,
    device
)

from src.models.vae import (
    MusicVAE,
    vae_loss_function,
    kl_annealing
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


def train_one_epoch(model, loader, optimizer, beta, epoch):

    model.train()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0

    for (batch,) in tqdm(
        loader,
        desc=f"Epoch {epoch}",
        leave=False
    ):

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed, mu, logvar, _ = model(batch)

        losses = vae_loss_function(
            reconstructed,
            batch,
            mu,
            logvar,
            beta=beta
        )

        losses["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss  += losses["total_loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl    += losses["kl_loss"].item()

    n = len(loader)

    return {
        "total": total_loss  / n,
        "recon": total_recon / n,
        "kl":    total_kl    / n
    }


@torch.no_grad()
def evaluate(model, loader, beta):

    model.eval()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0

    for (batch,) in loader:

        batch = batch.to(device)

        reconstructed, mu, logvar, _ = model(batch)

        losses = vae_loss_function(
            reconstructed,
            batch,
            mu,
            logvar,
            beta=beta
        )

        total_loss  += losses["total_loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl    += losses["kl_loss"].item()

    n = len(loader)

    return {
        "total": total_loss  / n,
        "recon": total_recon / n,
        "kl":    total_kl    / n
    }


def train_vae(
    epochs=vae_epochs,
    batch_size=vae_batch_size,
    lr=vae_lr,
    max_beta=vae_beta,
    warmup_epochs=30,
    checkpoint_dir=None
):

    if checkpoint_dir is None:
        checkpoint_dir = OUTPUT_DIR / "checkpoints" / "vae"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("\nLoading piano-roll dataset...")

    train_data, val_data = load_piano_roll_split()

    print(f"  Train : {train_data.shape}")
    print(f"  Val   : {val_data.shape}")

    train_loader = make_dataloader(train_data, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_data,   batch_size, shuffle=False)

    # ----------------------------------------------------------------- model
    model     = MusicVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_loss = float("inf")
    history = {"train_total": [], "train_recon": [], "train_kl": [],
               "val_total":   [], "val_recon":   [], "val_kl":   [],
               "beta": []}

    print(f"\nTraining MusicVAE on {device} for {epochs} epochs\n")

    for epoch in range(1, epochs + 1):

        beta = kl_annealing(
            epoch,
            warmup_epochs=warmup_epochs,
            max_beta=max_beta
        )

        train_losses = train_one_epoch(
            model, train_loader, optimizer, beta, epoch
        )

        val_losses = evaluate(model, val_loader, beta)

        scheduler.step(val_losses["total"])

        for split, losses in [("train", train_losses), ("val", val_losses)]:
            history[f"{split}_total"].append(losses["total"])
            history[f"{split}_recon"].append(losses["recon"])
            history[f"{split}_kl"].append(losses["kl"])

        history["beta"].append(beta)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"β={beta:.3f} | "
            f"Train [total={train_losses['total']:.4f} "
            f"recon={train_losses['recon']:.4f} "
            f"kl={train_losses['kl']:.4f}] | "
            f"Val [total={val_losses['total']:.4f}]"
        )

        if val_losses["total"] < best_val_loss:

            best_val_loss = val_losses["total"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_losses["total"],
                    "beta": beta
                },
                str(checkpoint_dir / "best.pt")
            )

    torch.save(
        model.state_dict(),
        str(checkpoint_dir / "final.pt")
    )

    print(f"\nCheckpoints saved to {checkpoint_dir}")
    print(f"Best val loss: {best_val_loss:.5f}")

    return model, history


if __name__ == "__main__":

    model, history = train_vae()
