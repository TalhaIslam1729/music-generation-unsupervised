import os
import torch
import numpy as np

from src.config import (
    ae_latent_dim,
    vae_latent_dim,
    OUTPUT_DIR,
    device
)

from src.models.autoencoder import LSTMAutoencoder
from src.models.vae import MusicVAE

def load_autoencoder(checkpoint_path=None):

    if checkpoint_path is None:
        checkpoint_path = OUTPUT_DIR / "checkpoints" / "autoencoder" / "best.pt"

    model = LSTMAutoencoder().to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    state = ckpt.get("model_state", ckpt)   # handle both dict and raw state

    model.load_state_dict(state)

    model.eval()

    print(f"[INFO] Loaded AE checkpoint: {checkpoint_path}")

    return model


def load_vae(checkpoint_path=None):

    if checkpoint_path is None:
        checkpoint_path = OUTPUT_DIR / "checkpoints" / "vae" / "best.pt"

    model = MusicVAE().to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    state = ckpt.get("model_state", ckpt)

    model.load_state_dict(state)

    model.eval()

    print(f"[INFO] Loaded VAE checkpoint: {checkpoint_path}")

    return model


def sample_from_ae(
    model,
    n_samples=1,
    threshold=0.35,
    temperature=1.0
):


    model.eval()

    with torch.no_grad():

        z = torch.randn(n_samples, ae_latent_dim).to(device) * temperature

        logits = model.decode(z)

        probs  = torch.sigmoid(logits)

        samples = (probs > threshold).float()

    return samples.cpu().numpy()


def sample_from_vae(
    model,
    n_samples=1,
    threshold=0.35,
    temperature=1.0
):


    model.eval()

    with torch.no_grad():

        z = torch.randn(n_samples, vae_latent_dim).to(device) * temperature

        logits = model.decode(z)

        probs  = torch.sigmoid(logits)

        samples = (probs > threshold).float()

    return samples.cpu().numpy()


def interpolate_latent(
    model,
    x1,
    x2,
    steps=8,
    threshold=0.35
):


    model.eval()

    outputs = []

    with torch.no_grad():

        x1 = x1.to(device)
        x2 = x2.to(device)

        enc_out_1 = model.encode(x1)
        enc_out_2 = model.encode(x2)

        if isinstance(enc_out_1, tuple):          # VAE returns (mu, logvar)
            z1, z2 = enc_out_1[0], enc_out_2[0]
        else:                                     # AE returns z directly
            z1, z2 = enc_out_1, enc_out_2

        for alpha in torch.linspace(0.0, 1.0, steps):

            z = (1.0 - alpha) * z1 + alpha * z2

            logits = model.decode(z)

            probs  = torch.sigmoid(logits)

            sample = (probs > threshold).float()

            outputs.append(sample.cpu().numpy())

    return outputs


def perturb_latent(
    model,
    x,
    n_samples=8,
    noise_scale=0.5,
    threshold=0.35
):

    model.eval()

    with torch.no_grad():

        x = x.to(device)

        enc_out = model.encode(x)

        z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        z_repeated = z.repeat(n_samples, 1)

        noise = torch.randn_like(z_repeated) * noise_scale

        z_noisy = z_repeated + noise

        logits  = model.decode(z_noisy)

        probs   = torch.sigmoid(logits)

        samples = (probs > threshold).float()

    return samples.cpu().numpy()


def encode_dataset(model, data_tensor):


    model.eval()

    latents = []

    with torch.no_grad():

        chunk_size = 64

        for start in range(0, len(data_tensor), chunk_size):

            chunk = data_tensor[start: start + chunk_size].to(device)

            enc_out = model.encode(chunk)

            z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)


if __name__ == "__main__":

    import numpy as np
    from src.config import PROCESSED_DIR

    try:

        vae = load_vae()

        samples = sample_from_vae(vae, n_samples=4, temperature=1.0)

        print(f"\nVAE samples shape : {samples.shape}")

        # interpolation between two random latent codes
        x1 = torch.randn(1, 64, 88)
        x2 = torch.randn(1, 64, 88)

        interp = interpolate_latent(vae, x1, x2, steps=6)

        print(f"Interpolation steps: {len(interp)}, each {interp[0].shape}")

    except FileNotFoundError as e:

        print(f"[INFO] VAE checkpoint not found – skipping demo. ({e})")

    try:

        ae = load_autoencoder()

        samples = sample_from_ae(ae, n_samples=4, temperature=1.0)

        print(f"\nAE samples shape  : {samples.shape}")

    except FileNotFoundError as e:

        print(f"[INFO] AE checkpoint not found – skipping demo. ({e})")
