#!/usr/bin/env python3
"""
Download Dorado basecalling models and export the neural network forward pass to ONNX.

This exports the model scores tensor, not Dorado's CRF decoding stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CPP = REPO_ROOT / "dorado" / "models" / "models.cpp"
DEFAULT_CDN_BASE = "https://cdn.oxfordnanoportal.com/software/analysis/dorado"


@dataclass
class ConvParams:
    insize: int
    size: int
    winlen: int
    stride: int
    activation: str
    flstm: bool = False


@dataclass
class TxEncoderParams:
    d_model: int
    nhead: int
    depth: int
    dim_feedforward: int
    attn_window: tuple[int, int]
    deepnorm_alpha: float
    theta: float
    max_seq_len: int


@dataclass
class LinearUpsampleParams:
    size: int
    scale_factor: int


@dataclass
class CRFEncoderParams:
    insize: int
    n_base: int
    state_len: int
    scale: float
    blank_score: float
    expand_blanks: bool
    permute: list[int]

    def outsize(self) -> int:
        if self.expand_blanks:
            return int(math.pow(self.n_base, self.state_len + 1))
        return (self.n_base + 1) * int(math.pow(self.n_base, self.state_len))

    def out_features(self) -> int:
        return int(math.pow(self.n_base, self.state_len + 1))


@dataclass
class ModelConfig:
    model_path: Path
    is_tx: bool
    convs: list[ConvParams]
    num_features: int
    stride: int
    out_features: int | None
    outsize: int
    blank_score: float
    scale: float
    state_len: int
    bias: bool
    clamp: bool
    lstm_size: int
    lstm_layers: int
    lstm_inner_dim: int | None
    tx: TxEncoderParams | None
    upsample: LinearUpsampleParams | None
    crf: CRFEncoderParams | None
    basecaller_chunk_size: int
    basecaller_overlap: int
    sample_rate: int | None
    sample_type: str | None
    signal_preprocessing: dict[str, Any]
    qbias: float
    qscale: float

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def scale_factor(self) -> int:
        return self.upsample.scale_factor if self.upsample is not None else 1

    def stride_inner(self) -> int:
        return self.stride * self.scale_factor()

    def chunk_size_granularity(self) -> int:
        return self.stride_inner() * (16 if self.is_tx else 1)

    def normalized_chunk_size(self) -> int:
        overlap = max(1, self.basecaller_overlap // self.stride_inner()) * self.stride_inner()
        granularity = self.chunk_size_granularity()
        minimum = overlap + granularity - 1
        return (max(minimum, self.basecaller_chunk_size) // granularity) * granularity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Dorado model and export its neural network forward pass to ONNX. "
            "The ONNX output is the score tensor before CRF decoding."
        )
    )
    parser.add_argument("--model", required=True, help="Dorado model name or local model directory")
    parser.add_argument(
        "--models-directory",
        default="models",
        help="directory used to cache downloaded models when --model is a model name",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for the ONNX export and manifest",
    )
    parser.add_argument(
        "--onnx-name",
        default="model.onnx",
        help="filename for the exported ONNX model inside --output-dir",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="example chunk size used for export; defaults to the normalized model chunk size",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download a named model even if it already exists locally",
    )
    return parser.parse_args()


def parse_model_catalog(models_cpp: Path) -> dict[str, str]:
    text = models_cpp.read_text(encoding="utf-8")
    pattern = re.compile(r'ModelInfo\{\s*"([^"]+)",\s*"([0-9a-f]{64}|)"', re.MULTILINE)
    return {name: checksum for name, checksum in pattern.findall(text)}


def get_cdn_base() -> str:
    override = os.environ.get("DORADO_CDN_URL_OVERRIDE")
    if not override:
        return DEFAULT_CDN_BASE
    return override.rstrip("/") + "/dorado"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_model(model_name: str, destination_root: Path, force_download: bool) -> Path:
    catalog = parse_model_catalog(MODELS_CPP)
    if model_name not in catalog:
        raise ValueError(f"Unknown model name: {model_name}")

    model_dir = destination_root / model_name
    if model_dir.exists() and not force_download:
        return model_dir

    destination_root.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        shutil.rmtree(model_dir)

    url = f"{get_cdn_base().rstrip('/')}/{model_name}.zip"
    with urllib.request.urlopen(url) as response:
        archive_bytes = response.read()

    expected = catalog[model_name]
    checksum = sha256_bytes(archive_bytes)
    if expected and checksum != expected:
        raise RuntimeError(
            f"Checksum mismatch for {model_name}: expected {expected}, got {checksum}"
        )

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(archive_bytes)
        archive_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination_root)
    finally:
        archive_path.unlink(missing_ok=True)

    if not model_dir.exists():
        raise RuntimeError(f"Downloaded archive did not create expected directory: {model_dir}")
    return model_dir


def load_config(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "config.toml").open("rb") as handle:
        return tomllib.load(handle)


def parse_conv(segment: dict[str, Any], has_clamp_next: bool) -> ConvParams:
    activation = segment["activation"]
    if activation == "swish":
        activation = "swish_clamp" if has_clamp_next else "swish"
    elif activation == "tanh":
        activation = "tanh"
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return ConvParams(
        insize=int(segment["insize"]),
        size=int(segment["size"]),
        winlen=int(segment["winlen"]),
        stride=int(segment["stride"]),
        activation=activation,
    )


def load_model_config(model_dir: Path) -> ModelConfig:
    cfg = load_config(model_dir)
    basecaller = cfg.get("basecaller", {})
    basecaller_chunk_size = int(basecaller.get("chunksize", 10000))
    basecaller_overlap = int(basecaller.get("overlap", 500))
    qscore = cfg.get("qscore", {})
    qbias = float(qscore.get("bias", 0.0))
    qscale = float(qscore.get("scale", 1.0))
    run_info = cfg.get("run_info", {})
    sample_rate = int(run_info["sample_rate"]) if "sample_rate" in run_info else None
    sample_type = str(run_info["sample_type"]) if "sample_type" in run_info else None
    signal_preprocessing: dict[str, Any] = {}
    if "scaling" in cfg:
        signal_preprocessing["scaling"] = dict(cfg["scaling"])
    if "normalisation" in cfg:
        signal_preprocessing["normalisation"] = dict(cfg["normalisation"])
    if "standardisation" in cfg:
        signal_preprocessing["standardisation"] = dict(cfg["standardisation"])

    model_cfg = cfg.get("model", {})
    tx_encoder_root = model_cfg.get("encoder")
    if tx_encoder_root and "transformer_encoder" in tx_encoder_root:
        encoder = tx_encoder_root
        tx_enc = encoder["transformer_encoder"]
        layer = tx_enc["layer"]
        tx = TxEncoderParams(
            d_model=int(layer["d_model"]),
            nhead=int(layer["nhead"]),
            depth=int(tx_enc["depth"]),
            dim_feedforward=int(layer["dim_feedforward"]),
            attn_window=(int(layer["attn_window"][0]), int(layer["attn_window"][1])),
            deepnorm_alpha=float(layer["deepnorm_alpha"]),
            theta=float(layer.get("theta", layer.get("rotary_base", 10000.0))),
            max_seq_len=int(layer.get("max_seq_len", 2048)),
        )
        upsample = LinearUpsampleParams(
            size=int(encoder["upsample"]["d_model"]),
            scale_factor=int(encoder["upsample"]["scale_factor"]),
        )
        crf = CRFEncoderParams(
            insize=int(encoder["crf"]["insize"]),
            n_base=int(encoder["crf"]["n_base"]),
            state_len=int(encoder["crf"]["state_len"]),
            scale=float(encoder["crf"]["scale"]),
            blank_score=float(encoder["crf"]["blank_score"]),
            expand_blanks=bool(encoder["crf"]["expand_blanks"]),
            permute=[int(v) for v in encoder["crf"]["permute"]],
        )
        convs: list[ConvParams] = []
        stride = 1
        for segment in encoder["conv"]["sublayers"]:
            if segment["type"] != "convolution":
                continue
            conv = parse_conv(segment, False)
            convs.append(conv)
            stride *= conv.stride
        stride //= upsample.scale_factor
        return ModelConfig(
            model_path=model_dir,
            is_tx=True,
            convs=convs,
            num_features=convs[0].insize,
            stride=stride,
            out_features=crf.out_features(),
            outsize=crf.outsize(),
            blank_score=crf.blank_score,
            scale=crf.scale,
            state_len=crf.state_len,
            bias=False,
            clamp=False,
            lstm_size=-1,
            lstm_layers=0,
            lstm_inner_dim=None,
            tx=tx,
            upsample=upsample,
            crf=crf,
            basecaller_chunk_size=basecaller_chunk_size,
            basecaller_overlap=basecaller_overlap,
            sample_rate=sample_rate,
            sample_type=sample_type,
            signal_preprocessing=signal_preprocessing,
            qbias=qbias,
            qscale=qscale,
        )

    encoder = cfg["encoder"]
    input_cfg = cfg["input"]
    global_norm = cfg["global_norm"]
    num_features = int(input_cfg["features"])
    state_len = int(global_norm["state_len"])
    convs = []
    stride = 1
    out_features = None
    blank_score = 0.0
    scale = 1.0
    clamp = False
    bias = True
    lstm_layers = 5
    lstm_size = 0
    lstm_inner_dim = None

    if "type" in encoder:
        sublayers = encoder["sublayers"]
        bias = False
        clamp_indices = {idx - 1 for idx, layer in enumerate(sublayers) if layer["type"] == "clamp"}
        for idx, segment in enumerate(sublayers):
            seg_type = segment["type"]
            if seg_type == "convolution":
                conv = parse_conv(segment, idx in clamp_indices)
                convs.append(conv)
                stride *= conv.stride
            elif seg_type == "linear":
                out_features = int(segment["out_features"])
                bias = bool(segment.get("bias", convs[-1].size > 128))
            elif seg_type == "linearcrfencoder":
                blank_score = float(segment["blank_score"])
                scale = float(segment.get("scale", 1.0))
            elif seg_type == "flstm":
                lstm_inner_dim = int(segment["inner_dim"])
        lstm_layers = sum(1 for segment in sublayers if segment["type"] == "lstm")
        flstm_layers = sum(1 for segment in sublayers if segment["type"] == "flstm")
        if flstm_layers:
            raise NotImplementedError("FLSTM model export is not implemented in this utility")
        lstm_size = convs[-1].size
        bias = bool(bias)
        clamp = any(segment["type"] == "clamp" for segment in sublayers)
    else:
        stride = int(encoder["stride"])
        lstm_size = int(encoder["features"])
        blank_score = float(encoder["blank_score"])
        scale = float(encoder["scale"])
        first_conv = int(encoder.get("first_conv_size", 4))
        convs = [
            ConvParams(num_features, first_conv, 5, 1, "swish"),
            ConvParams(first_conv, 16, 5, 1, "swish"),
            ConvParams(16, lstm_size, 19, stride, "swish"),
        ]

    outsize = 1 << ((state_len + 1) << 1)
    return ModelConfig(
        model_path=model_dir,
        is_tx=False,
        convs=convs,
        num_features=num_features,
        stride=stride,
        out_features=out_features,
        outsize=outsize,
        blank_score=blank_score,
        scale=scale,
        state_len=state_len,
        bias=bias,
        clamp=clamp,
        lstm_size=lstm_size,
        lstm_layers=lstm_layers,
        lstm_inner_dim=lstm_inner_dim,
        tx=None,
        upsample=None,
        crf=None,
        basecaller_chunk_size=basecaller_chunk_size,
        basecaller_overlap=basecaller_overlap,
        sample_rate=sample_rate,
        sample_type=sample_type,
        signal_preprocessing=signal_preprocessing,
        qbias=qbias,
        qscale=qscale,
    )


def load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, torch.jit.RecursiveScriptModule):
        state_dict = tensor.state_dict()
        if len(state_dict) != 1:
            raise TypeError(f"Expected exactly one tensor in {path}, found keys {list(state_dict.keys())}")
        tensor = next(iter(state_dict.values()))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, found {type(tensor)!r}")
    return tensor.detach().to(torch.float32)


def lstm_weight_names(config: ModelConfig) -> list[str]:
    if config.lstm_inner_dim is not None:
        raise NotImplementedError("FLSTM model export is not implemented in this utility")

    conv_names = [".conv.weight.tensor", ".conv.bias.tensor"]
    lstm_names = [
        ".rnn.weight_ih_l0.tensor",
        ".rnn.weight_hh_l0.tensor",
        ".rnn.bias_ih_l0.tensor",
        ".rnn.bias_hh_l0.tensor",
    ]
    names: list[str] = []
    for conv_idx in range(len(config.convs)):
        for suffix in conv_names:
            names.append(f"{conv_idx}{suffix}")
    for layer_idx in range(config.lstm_layers):
        model_layer = len(config.convs) + layer_idx + 1
        for suffix in lstm_names:
            names.append(f"{model_layer}{suffix}")
    layer = len(config.convs) + config.lstm_layers + 1
    names.append(f"{layer}.linear.weight.tensor")
    if config.bias:
        names.append(f"{layer}.linear.bias.tensor")
    if config.out_features is not None:
        names.append(f"{layer + 1}.linear.weight.tensor")
    return names


def tx_weight_names(config: ModelConfig) -> list[str]:
    assert config.tx is not None
    conv_names = [".conv.weight.tensor", ".conv.bias.tensor"]
    enc_names = [
        ".self_attn.Wqkv.weight.tensor",
        ".self_attn.out_proj.weight.tensor",
        ".self_attn.out_proj.bias.tensor",
        ".ff.fc1.weight.tensor",
        ".ff.fc2.weight.tensor",
        ".norm1.weight.tensor",
        ".norm2.weight.tensor",
    ]
    remaining = [
        "upsample.linear.weight.tensor",
        "upsample.linear.bias.tensor",
        "crf.linear.weight.tensor",
    ]
    names: list[str] = []
    for conv_idx in range(len(config.convs)):
        for suffix in conv_names:
            names.append(f"conv.{conv_idx}{suffix}")
    for enc_idx in range(config.tx.depth):
        for suffix in enc_names:
            names.append(f"transformer_encoder.{enc_idx}{suffix}")
    names.extend(remaining)
    return names


def load_model_weights(config: ModelConfig) -> list[torch.Tensor]:
    names = tx_weight_names(config) if config.is_tx else lstm_weight_names(config)
    return [load_tensor(config.model_path / name) for name in names]


class ConvStack(nn.Module):
    def __init__(self, convs: list[ConvParams]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    conv.insize,
                    conv.size,
                    conv.winlen,
                    stride=conv.stride,
                    padding=conv.winlen // 2,
                )
                for conv in convs
            ]
        )
        self.params = convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, params in zip(self.layers, self.params, strict=True):
            x = layer(x)
            if params.activation == "swish":
                x = F.silu(x)
            elif params.activation == "swish_clamp":
                x = torch.clamp(F.silu(x), max=3.5)
            elif params.activation == "tanh":
                x = torch.tanh(x)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported activation: {params.activation}")
        return x.transpose(1, 2)


class LSTMStack(nn.Module):
    def __init__(self, num_layers: int, size: int, reverse_first: bool = True) -> None:
        super().__init__()
        self.reverse_first = reverse_first
        self.layers = nn.ModuleList(
            [nn.LSTM(size, size, batch_first=True) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_reverse = not self.reverse_first
        for idx, layer in enumerate(self.layers):
            if idx != 0 or self.reverse_first:
                x = x.flip(1)
                is_reverse = not is_reverse
            x = layer(x)[0]
        return x.flip(1) if is_reverse else x


class LinearCRF(nn.Module):
    def __init__(self, insize: int, outsize: int, bias: bool, tanh_and_scale: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(insize, outsize, bias=bias)
        self.tanh_and_scale = tanh_and_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.tanh_and_scale:
            x = torch.tanh(x) * 5.0
        return x


class Clamp(nn.Module):
    def __init__(self, minimum: float, maximum: float, active: bool) -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.active = active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active:
            return torch.clamp(x, self.minimum, self.maximum)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return x * rstd * self.weight


class GatedMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(x)
        value, gate = projected.chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.float32).reshape(max_seq_len, 1, 1, 1)
        indices = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = torch.pow(torch.tensor(theta, dtype=torch.float32), indices / dim).reciprocal()
        freqs = positions * inv_freq.reshape(1, 1, 1, dim // 2)
        self.register_buffer("cos_freqs", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_freqs", torch.sin(freqs), persistent=False)

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        _, seq_len, _, _, head_dim = qkv.shape
        q, k, v = qkv.unbind(dim=2)
        cos = self.cos_freqs[:seq_len].permute(1, 0, 2, 3)
        sin = self.sin_freqs[:seq_len].permute(1, 0, 2, 3)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            even = x[..., : head_dim // 2]
            odd = x[..., head_dim // 2 :]
            return torch.cat((cos * even - sin * odd, sin * even + cos * odd), dim=-1)

        q = rotate(q).permute(0, 2, 1, 3)
        k = rotate(k).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        return torch.stack((q, k, v), dim=0)


def build_attn_window_mask(size: int, attn_window: tuple[int, int], device: torch.device) -> torch.Tensor:
    win_upper, win_lower = attn_window
    positions = torch.arange(size, device=device)
    row = positions.unsqueeze(1)
    col = positions.unsqueeze(0)
    return (col >= row - win_upper) & (col <= row + win_lower)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        attn_window: tuple[int, int],
        theta: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_window = attn_window
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.rotary_emb = RotaryEmbedding(self.head_dim, theta, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.wqkv(x).view(batch, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = self.rotary_emb(qkv).unbind(dim=0)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = build_attn_window_mask(seq_len, self.attn_window, x.device)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out_proj(output)


class TxEncoder(nn.Module):
    def __init__(self, params: TxEncoderParams) -> None:
        super().__init__()
        self.deepnorm_alpha = params.deepnorm_alpha
        self.self_attn = MultiHeadAttention(
            params.d_model,
            params.nhead,
            params.attn_window,
            params.theta,
            params.max_seq_len,
        )
        self.ff = GatedMLP(params.d_model, params.dim_feedforward)
        self.norm1 = RMSNorm(params.d_model)
        self.norm2 = RMSNorm(params.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(x)
        x = self.norm1(attn + (x * self.deepnorm_alpha))
        ff = self.ff(x)
        x = self.norm2(ff + (x * self.deepnorm_alpha))
        return x


class TxEncoderStack(nn.Module):
    def __init__(self, params: TxEncoderParams) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TxEncoder(params) for _ in range(params.depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LinearUpsample(nn.Module):
    def __init__(self, params: LinearUpsampleParams) -> None:
        super().__init__()
        self.scale_factor = params.scale_factor
        self.linear = nn.Linear(params.size, params.scale_factor * params.size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        return self.linear(x).reshape(batch, seq_len * self.scale_factor, channels)


class LinearScaledCRF(nn.Module):
    def __init__(self, params: CRFEncoderParams) -> None:
        super().__init__()
        self.scale = params.scale
        self.linear = nn.Linear(params.insize, params.outsize(), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.scale)


class CRFModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.convs = ConvStack(config.convs)
        self.rnns = LSTMStack(config.lstm_layers, config.lstm_size, reverse_first=True)
        self.linear2: LinearCRF | None = None
        self.clamp1: Clamp | None = None
        if config.out_features is not None:
            self.linear1 = LinearCRF(config.lstm_size, config.out_features, config.bias, False)
            self.linear2 = LinearCRF(config.out_features, config.outsize, False, config.scale == 5.0)
            self.clamp1 = Clamp(-5.0, 5.0, config.clamp)
        elif config.convs[0].size > 4 and config.num_features == 1:
            self.linear1 = LinearCRF(config.lstm_size, config.outsize, False, config.scale == 5.0)
            self.clamp1 = Clamp(-5.0, 5.0, config.clamp)
        else:
            self.linear1 = LinearCRF(config.lstm_size, config.outsize, True, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.rnns(x)
        x = self.linear1(x)
        if self.linear2 is not None:
            x = self.linear2(x)
        if self.clamp1 is not None:
            x = self.clamp1(x)
        return x


class TxModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.tx is not None and config.upsample is not None and config.crf is not None
        self.convs = ConvStack(config.convs)
        self.transformer_encoder = TxEncoderStack(config.tx)
        self.upsample = LinearUpsample(config.upsample)
        self.crf = LinearScaledCRF(config.crf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.transformer_encoder(x)
        x = self.upsample(x)
        x = self.crf(x)
        return x


def build_model(config: ModelConfig) -> nn.Module:
    return TxModel(config) if config.is_tx else CRFModel(config)


def load_weights_into_model(model: nn.Module, weights: list[torch.Tensor]) -> None:
    params = list(model.parameters())
    if len(params) != len(weights):
        raise RuntimeError(
            f"Parameter count mismatch: model expects {len(params)} tensors, got {len(weights)}"
        )
    with torch.no_grad():
        for idx, (param, weight) in enumerate(zip(params, weights, strict=True)):
            if tuple(param.shape) != tuple(weight.shape):
                raise RuntimeError(
                    f"Parameter {idx} shape mismatch: expected {tuple(param.shape)}, got {tuple(weight.shape)}"
                )
            param.copy_(weight.to(dtype=param.dtype))


def export_model(model: nn.Module, config: ModelConfig, output_dir: Path, onnx_name: str, opset: int, chunk_size: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / onnx_name
    example = torch.randn(1, config.num_features, chunk_size, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["scores"],
        dynamic_axes={
            "input": {0: "batch", 2: "samples"},
            "scores": {0: "batch", 1: "frames"},
        },
        opset_version=opset,
    )
    return onnx_path


def write_manifest(output_dir: Path, config: ModelConfig, onnx_path: Path, chunk_size: int, opset: int) -> None:
    manifest = {
        "model_name": config.model_name,
        "model_path": str(config.model_path),
        "onnx_path": str(onnx_path),
        "architecture": "transformer" if config.is_tx else "crf",
        "num_features": config.num_features,
        "chunk_size": chunk_size,
        "stride_inner": config.stride_inner(),
        "opset": opset,
        "decoded_output": False,
        "output_description": "network scores before CRF decoding",
        "sample_rate": config.sample_rate,
        "sample_type": config.sample_type,
        "signal_preprocessing": config.signal_preprocessing,
        "qbias": config.qbias,
        "qscale": config.qscale,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def resolve_model_path(model_arg: str, models_directory: Path, force_download: bool) -> Path:
    candidate = Path(model_arg)
    if candidate.exists():
        return candidate.resolve()
    return download_model(model_arg, models_directory, force_download)


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_path(args.model, Path(args.models_directory), args.force_download)
    config = load_model_config(model_dir)
    weights = load_model_weights(config)
    model = build_model(config)
    load_weights_into_model(model, weights)

    chunk_size = args.chunk_size if args.chunk_size > 0 else config.normalized_chunk_size()
    if chunk_size <= 0:
        raise RuntimeError("Chunk size must be positive")

    output_dir = Path(args.output_dir).resolve()
    onnx_path = export_model(model, config, output_dir, args.onnx_name, args.opset, chunk_size)
    write_manifest(output_dir, config, onnx_path, chunk_size, args.opset)

    print(f"model: {config.model_name}")
    print(f"architecture: {'transformer' if config.is_tx else 'crf'}")
    print(f"onnx: {onnx_path}")
    print(f"manifest: {output_dir / 'export_manifest.json'}")
    print("note: ONNX output is the score tensor before Dorado CRF decoding")


if __name__ == "__main__":
    main()
