from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dims: list[int] | None = None, dropout: float = 0.3) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dims[1], 1)

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(tabular)
        return self.classifier(features).squeeze(-1)


class ECG1DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int] | None = None,
        kernel_size: int = 7,
        dropout: float = 0.3,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        blocks = []
        current_in = in_channels
        padding = kernel_size // 2
        for current_out in channels:
            blocks.extend(
                [
                    nn.Conv1d(current_in, current_out, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(current_out),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            current_in = current_out

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.encoder(waveform)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.encode(waveform)
        return self.classifier(features).squeeze(-1)


class ECGPlusTabularModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        tabular_dim: int = 7,
        ecg_channels: list[int] | None = None,
        ecg_dropout: float = 0.3,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ecg_branch = ECG1DCNN(
            in_channels=in_channels,
            channels=ecg_channels or [32, 64, 128, 256],
            dropout=ecg_dropout,
            embedding_dim=128,
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(ecg_dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(160, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(ecg_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        ecg_features = self.ecg_branch.encode(waveform)
        tabular_features = self.tabular_branch(tabular)
        fused = torch.cat([ecg_features, tabular_features], dim=1)
        return self.classifier(fused).squeeze(-1)


def build_model(input_mode: str, config: dict, tabular_dim: int = 7, lead_channels: int | None = None) -> nn.Module:
    model_config = config["models"]
    if input_mode == "tabular":
        return TabularMLP(input_dim=tabular_dim, hidden_dims=model_config["tabular_hidden_dims"])
    if input_mode == "full12":
        return ECG1DCNN(
            in_channels=12,
            channels=model_config["ecg_channels"],
            kernel_size=model_config["ecg_kernel_size"],
            dropout=model_config["ecg_dropout"],
        )
    if input_mode == "single":
        return ECG1DCNN(
            in_channels=lead_channels or 1,
            channels=model_config["ecg_channels"],
            kernel_size=model_config["ecg_kernel_size"],
            dropout=model_config["ecg_dropout"],
        )
    if input_mode == "single_plus_tabular":
        return ECGPlusTabularModel(
            in_channels=lead_channels or 1,
            tabular_dim=tabular_dim,
            ecg_channels=model_config["ecg_channels"],
            ecg_dropout=model_config["ecg_dropout"],
            fusion_hidden_dim=model_config["fusion_hidden_dim"],
        )
    raise ValueError(f"Unsupported input_mode '{input_mode}'.")
