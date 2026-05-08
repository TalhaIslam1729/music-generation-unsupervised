
import torch
import torch.nn as nn

from src.config import (
    n_pitches,
    seq_len,
    ae_hidden_dim,
    ae_latent_dim,
    ae_num_layers,
    ae_dropout
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

        bce_loss = self.bce(
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
            bce_loss
        )

        if self.reduction == "mean":

            return loss.mean()

        elif self.reduction == "sum":

            return loss.sum()

        return loss


class LSTMAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_pitches,
            hidden_size=ae_hidden_dim,
            num_layers=ae_num_layers,
            dropout=ae_dropout,
            batch_first=True
        )

        self.to_latent = nn.Linear(
            ae_hidden_dim,
            ae_latent_dim
        )

        self.from_latent = nn.Linear(
            ae_latent_dim,
            ae_hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=ae_hidden_dim,
            hidden_size=ae_hidden_dim,
            num_layers=ae_num_layers,
            dropout=ae_dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            ae_hidden_dim,
            n_pitches
        )

    def encode(self, x):

        i, (hidden, i) = self.encoder(x)

        final_hidden = hidden[-1]

        z = self.to_latent(
            final_hidden
        )

        return z

    def decode(self, z):

        hidden = self.from_latent(z)

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

        z = self.encode(x)

        reconstructed = self.decode(z)

        return reconstructed, z

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
                ae_latent_dim
            ).to(device)

            logits = self.decode(z)

            probs = torch.sigmoid(
                logits
            )

            samples = (
                probs > threshold
            ).float()

        return samples


if __name__ == "__main__":

    model = LSTMAutoencoder()

    dummy = torch.randn(
        4,
        seq_len,
        n_pitches
    )

    reconstructed, z = model(dummy)

    print("\nInput Shape:")

    print(dummy.shape)

    print("\nLatent Shape:")

    print(z.shape)

    print("\nReconstructed Shape:")

    print(reconstructed.shape)

    generated = model.generate(
        batch_size=2
    )

    print("\nGenerated Shape:")

    print(generated.shape)