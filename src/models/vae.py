# src/models/vae.py

import torch
import torch.nn as nn

from src.config import (
    n_pitches,
    seq_len,
    vae_hidden_dim,
    vae_latent_dim,
    vae_num_layers
)


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=5.0,
        gamma=2.0,
        reduction="mean"
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )

    def forward(self, logits, targets):

        bce = self.bce(
            logits,
            targets
        )

        probs = torch.sigmoid(
            logits
        )

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = (
            (1 - pt) ** self.gamma
        )

        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1.0
        )

        loss = (
            alpha_weight *
            focal_weight *
            bce
        )

        if self.reduction == "mean":

            return loss.mean()

        elif self.reduction == "sum":

            return loss.sum()

        return loss


class MusicVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_pitches,
            hidden_size=vae_hidden_dim,
            num_layers=vae_num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.mu_layer = nn.Linear(
            vae_hidden_dim,
            vae_latent_dim
        )

        self.logvar_layer = nn.Linear(
            vae_hidden_dim,
            vae_latent_dim
        )


        self.latent_to_hidden = nn.Linear(
            vae_latent_dim,
            vae_hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=vae_hidden_dim,
            hidden_size=vae_hidden_dim,
            num_layers=vae_num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.output_layer = nn.Linear(
            vae_hidden_dim,
            n_pitches
        )


    def encode(self, x):

        _, (hidden, _) = self.encoder(x)

        final_hidden = hidden[-1]

        mu = self.mu_layer(
            final_hidden
        )

        logvar = self.logvar_layer(
            final_hidden
        )

        return mu, logvar


    def reparameterize(self, mu, logvar):

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        z = mu + (
            std * epsilon
        )

        return z


    def decode(self, z):

        hidden = self.latent_to_hidden(
            z
        )

        repeated = hidden.unsqueeze(1).repeat(
            1,
            seq_len,
            1
        )

        decoded, _ = self.decoder(
            repeated
        )

        logits = self.output_layer(
            decoded
        )

        return logits


    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstructed = self.decode(z)

        return reconstructed, mu, logvar, z

    def generate(
        self,
        batch_size=1,
        threshold=0.35,
        device="cpu"
    ):

        self.eval()

        with torch.no_grad():

            z = torch.randn(
                batch_size,
                vae_latent_dim
            ).to(device)

            logits = self.decode(z)

            probs = torch.sigmoid(
                logits
            )

            samples = (
                probs > threshold
            ).float()


    def interpolate(
        self,
        x1,
        x2,
        steps=8
    ):

        self.eval()

        with torch.no_grad():

            mu1, _ = self.encode(x1)

            mu2, _ = self.encode(x2)

            outputs = []

            for alpha in torch.linspace(
                0,
                1,
                steps
            ):

                z = (
                    (1 - alpha) * mu1 +
                    alpha * mu2
                )

                logits = self.decode(z)

                probs = torch.sigmoid(
                    logits
                )

                sample = (
                    probs > 0.35
                ).float()

                outputs.append(sample)

        return outputs


def kl_divergence(mu, logvar):

    kl = -0.5 * torch.sum(

        1 +
        logvar -
        mu.pow(2) -
        logvar.exp(),

        dim=1
    )

    return kl.mean()


def vae_loss_function(
    logits,
    targets,
    mu,
    logvar,
    beta=1.0
):

    recon_loss = FocalLoss()(
        logits,
        targets
    )

    kl_loss = kl_divergence(
        mu,
        logvar
    )

    total_loss = (
        recon_loss +
        beta * kl_loss
    )

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss
    }


def kl_annealing(
    epoch,
    warmup_epochs=30,
    max_beta=1.0
):

    beta = min(
        max_beta,
        epoch / warmup_epochs
    )

    return beta


if __name__ == "__main__":

    model = MusicVAE()

    dummy = torch.randn(
        4,
        seq_len,
        n_pitches
    )

    reconstructed, mu, logvar, z = model(dummy)

    print("\nInput Shape:")

    print(dummy.shape)

    print("\nMu Shape:")

    print(mu.shape)

    print("\nLogVar Shape:")

    print(logvar.shape)

    print("\nLatent Shape:")

    print(z.shape)

    print("\nReconstructed Shape:")

    print(reconstructed.shape)

    losses = vae_loss_function(
        reconstructed,
        dummy,
        mu,
        logvar,
        beta=0.1
    )

    print("\nLosses:")

    for k, v in losses.items():

        print(k, v.item())

    generated = model.generate(
        batch_size=2
    )

    print("\nGenerated Shape:")

    print(generated.shape)