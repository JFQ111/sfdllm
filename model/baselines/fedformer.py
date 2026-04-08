import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        x = x.transpose(1, 2)  # [B, D, L]
        pad = (self.kernel_size - 1) // 2
        if self.kernel_size % 2 == 0:
            x = F.pad(x, (pad, pad + 1), mode="replicate")
        else:
            x = F.pad(x, (pad, pad), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kernel_size <= 1:
            trend = x
            seasonal = torch.zeros_like(x)
            return seasonal, trend
        trend = self._moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class FourierBlock(nn.Module):
    def __init__(self, d_model: int, modes: int):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        scale = 1.0 / max(1, d_model)
        self.weight_real = nn.Parameter(torch.randn(modes, d_model) * scale)
        self.weight_imag = nn.Parameter(torch.randn(modes, d_model) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        batch_size, seq_len, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        modes = min(self.modes, x_ft.size(1))
        out_ft = torch.zeros_like(x_ft)
        if modes > 0:
            weight = torch.complex(self.weight_real[:modes], self.weight_imag[:modes])
            out_ft[:, :modes, :] = x_ft[:, :modes, :] * weight.unsqueeze(0)
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)
        return x


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, modes: int, dropout: float):
        super().__init__()
        self.fourier = FourierBlock(d_model, modes)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.fourier(x))
        x = self.norm2(x + self.ff(x))
        return x


class TrendBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.act(y)
        y = self.dropout(y)
        return self.norm(x + y)


class FEDformer(nn.Module):
    """
    FEDformer-based classifier for vibration signals.
    Input shapes supported: [B, L], [B, 1, L], [B, L, C].
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.input_dim = getattr(configs, "input_dim", 1)
        self.d_model = getattr(configs, "d_model", 128)
        self.d_ff = getattr(configs, "d_ff", self.d_model * 4)
        self.n_layers = (
            getattr(configs, "e_layers", None)
            or getattr(configs, "n_layers", None)
            or 2
        )
        self.modes = getattr(configs, "modes", 32)
        self.moving_avg = getattr(configs, "moving_avg", 25)
        self.dropout_rate = getattr(configs, "dropout", 0.1)
        self.num_classes = getattr(configs, "num_classes", None)
        if self.num_classes is None:
            raise ValueError("configs.num_classes is required for FEDformer")

        self.value_embedding = nn.Linear(self.input_dim, self.d_model)
        self.decomp = SeriesDecomposition(self.moving_avg)

        self.seasonal_layers = nn.ModuleList(
            [
                FEDformerEncoderLayer(self.d_model, self.d_ff, self.modes, self.dropout_rate)
                for _ in range(self.n_layers)
            ]
        )
        self.trend_layers = nn.ModuleList(
            [TrendBlock(self.d_model, self.dropout_rate) for _ in range(self.n_layers)]
        )

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3:
            if x.size(1) == 1:
                x = x[:, 0, :].unsqueeze(-1)
            elif x.size(1) == self.input_dim:
                x = x.transpose(1, 2)
            return x
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_input(x)
        x = self.value_embedding(x)

        seasonal, trend = self.decomp(x)
        for layer in self.seasonal_layers:
            seasonal = layer(seasonal)
        for layer in self.trend_layers:
            trend = layer(trend)

        x = self.norm(seasonal + trend)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
