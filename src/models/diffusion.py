# src/training/rlhf.py

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.config import (
    rl_lr,
    rl_steps,
    rl_gamma,
    rl_survey_path,
    bos_token,
    eos_token,
    device
)

from src.models.transformer import (
    MusicTransformer
)


class RewardModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):

        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(
            embedded
        )

        hidden = hidden[-1]

        reward = self.fc(hidden)

        return reward.squeeze(1)


def load_survey_dataset(csv_path):

    df = pd.read_csv(csv_path)

    return df

def compute_rewards(df):

    grouped = df.groupby(
        "sample_id"
    )["rating"].mean()

    return grouped.to_dict()

def normalize_rewards(rewards):

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    mean = rewards.mean()

    std = rewards.std() + 1e-8

    normalized = (
        rewards - mean
    ) / std

    return normalized


def sample_sequence(
    model,
    genre_id,
    max_length=256,
    temperature=1.0
):

    model.eval()

    generated = torch.tensor(
        [[bos_token]],
        dtype=torch.long
    ).to(device)

    genre_tensor = torch.tensor(
        [genre_id],
        dtype=torch.long
    ).to(device)

    log_probs = []

    with torch.no_grad():

        for _ in range(max_length):

            logits = model(
                generated,
                genre_tensor
            )

            next_logits = logits[:, -1, :]

            next_logits = (
                next_logits / temperature
            )

            probs = F.softmax(
                next_logits,
                dim=-1
            )

            distribution = torch.distributions.Categorical(
                probs
            )

            next_token = distribution.sample()

            log_prob = distribution.log_prob(
                next_token
            )

            log_probs.append(log_prob)

            next_token = next_token.unsqueeze(0)

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            if next_token.item() == eos_token:

                break

    generated = generated.squeeze(0)

    return generated, torch.stack(log_probs).sum()


def policy_gradient_step(
    model,
    optimizer,
    log_probs,
    rewards
):

    rewards = normalize_rewards(
        rewards
    ).to(device)

    policy_loss = []

    for log_prob, reward in zip(
        log_probs,
        rewards
    ):

        loss = -log_prob * reward

        policy_loss.append(loss)

    policy_loss = torch.stack(
        policy_loss
    ).mean()

    optimizer.zero_grad()

    policy_loss.backward()

    optimizer.step()

    return policy_loss.item()


def reinforce_train(
    model,
    reward_function,
    genre_id=0,
    steps=rl_steps,
    batch_size=8
):

    optimizer = optim.Adam(
        model.parameters(),
        lr=rl_lr
    )

    model.train()

    for step in range(steps):

        batch_log_probs = []

        batch_rewards = []

        for _ in range(batch_size):

            sequence, log_prob = sample_sequence(
                model,
                genre_id
            )

            reward = reward_function(
                sequence
            )

            batch_log_probs.append(
                log_prob
            )

            batch_rewards.append(
                reward
            )

        loss = policy_gradient_step(
            model,
            optimizer,
            batch_log_probs,
            batch_rewards
        )

        avg_reward = np.mean(
            batch_rewards
        )

        print(

            f"Step {step+1}/{steps} | "
            f"Loss: {loss:.4f} | "
            f"Reward: {avg_reward:.4f}"

        )


def dummy_reward_function(sequence):

    unique_tokens = len(
        torch.unique(sequence)
    )

    length_bonus = (
        len(sequence) / 100
    )

    reward = (
        unique_tokens * 0.01 +
        length_bonus
    )

    return reward


def train_reward_model(
    reward_model,
    token_sequences,
    scores,
    epochs=10,
    lr=0.001
):

    optimizer = optim.Adam(
        reward_model.parameters(),
        lr=lr
    )

    criterion = nn.MSELoss()

    reward_model.train()

    for epoch in range(epochs):

        total_loss = 0

        for seq, score in zip(
            token_sequences,
            scores
        ):

            seq = torch.tensor(
                seq,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            score = torch.tensor(
                [score],
                dtype=torch.float32
            ).to(device)

            prediction = reward_model(
                seq
            )

            loss = criterion(
                prediction,
                score
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(
            token_sequences
        )

        print(

            f"Epoch {epoch+1} | "
            f"Reward Loss: {avg_loss:.4f}"

        )


def save_generated_tokens(
    token_sequence,
    output_path
):

    np.save(
        output_path,
        np.array(token_sequence)
    )

    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":

    model = MusicTransformer().to(device)

    print("\nStarting RL Fine-Tuning...\n")

    reinforce_train(
        model=model,
        reward_function=dummy_reward_function,
        genre_id=0,
        steps=10,
        batch_size=4
    )

    print("\nRL Training Complete")