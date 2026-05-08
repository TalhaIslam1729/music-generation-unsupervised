# src/models/transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    vocab_size,
    n_genres,
    pad_token,
    bos_token,
    tf_d_model,
    tf_n_heads,
    tf_num_layers,
    tf_d_ff,
    tf_dropout,
    tf_max_seq_len
)

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=5000
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(

            torch.arange(
                0,
                d_model,
                2
            ) *

            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


class MusicTransformer(nn.Module):

    def __init__(self):

        super().__init__()


        self.token_embedding = nn.Embedding(
            vocab_size,
            tf_d_model,
            padding_idx=pad_token
        )


        self.genre_embedding = nn.Embedding(
            n_genres,
            tf_d_model
        )


        self.position_encoding = PositionalEncoding(
            tf_d_model,
            max_len=tf_max_seq_len
        )


        decoder_layer = nn.TransformerDecoderLayer(
            d_model=tf_d_model,
            nhead=tf_n_heads,
            dim_feedforward=tf_d_ff,
            dropout=tf_dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=tf_num_layers
        )


        self.output_layer = nn.Linear(
            tf_d_model,
            vocab_size
        )

        self.dropout = nn.Dropout(
            tf_dropout
        )


    def generate_causal_mask(
        self,
        seq_len,
        device
    ):

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device
            ),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        return mask


    def forward(
        self,
        input_ids,
        genre_ids
    ):

        batch_size, seq_len = input_ids.shape


        token_embed = self.token_embedding(
            input_ids
        )


        genre_embed = self.genre_embedding(
            genre_ids
        )

        genre_embed = genre_embed.unsqueeze(1)

        genre_embed = genre_embed.repeat(
            1,
            seq_len,
            1
        )

        x = token_embed + genre_embed


        x = self.position_encoding(x)

        x = self.dropout(x)

        causal_mask = self.generate_causal_mask(
            seq_len,
            input_ids.device
        )


        output = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask
        )


        logits = self.output_layer(
            output
        )

        return logits


    def compute_loss(
        self,
        logits,
        targets
    ):

        loss = F.cross_entropy(

            logits.reshape(
                -1,
                vocab_size
            ),

            targets.reshape(-1),

            ignore_index=pad_token
        )

        return loss


    def compute_perplexity(
        self,
        loss
    ):

        return torch.exp(loss)


    def generate(
        self,
        genre_id,
        max_length=512,
        temperature=1.0,
        device="cpu"
    ):

        self.eval()

        generated = torch.tensor(
            [[bos_token]],
            dtype=torch.long
        ).to(device)

        genre_tensor = torch.tensor(
            [genre_id],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():

            for _ in range(max_length):

                logits = self.forward(
                    generated,
                    genre_tensor
                )

                next_token_logits = logits[:, -1, :]

                next_token_logits = (
                    next_token_logits / temperature
                )

                probs = F.softmax(
                    next_token_logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

                generated = torch.cat(
                    [generated, next_token],
                    dim=1
                )

        return generated.squeeze(0).tolist()



if __name__ == "__main__":

    model = MusicTransformer()

    batch_size = 4
    seq_length = 128

    dummy_inputs = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_length)
    )

    dummy_genres = torch.randint(
        0,
        n_genres,
        (batch_size,)
    )

    logits = model(
        dummy_inputs,
        dummy_genres
    )

    print("\nInput Shape:")

    print(dummy_inputs.shape)

    print("\nLogits Shape:")

    print(logits.shape)

    dummy_targets = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_length)
    )

    loss = model.compute_loss(
        logits,
        dummy_targets
    )

    print("\nLoss:")

    print(loss.item())

    perplexity = model.compute_perplexity(
        loss
    )

    print("\nPerplexity:")

    print(perplexity.item())

    generated = model.generate(
        genre_id=0,
        max_length=100
    )

    print("\nGenerated Length:")

    print(len(generated))