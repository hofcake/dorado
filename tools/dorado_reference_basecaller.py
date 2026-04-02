#!/usr/bin/env python3
"""
Very small reference simplex basecaller using:
- POD5 Python reader
- ONNX Runtime CPU
- pure Python preprocessing, CRF decode, and stitching
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from dorado_export_onnx import (
    ModelConfig,
    build_model,
    export_model,
    load_model_config,
    load_model_weights,
    load_weights_into_model,
    resolve_model_path,
    write_manifest,
)

try:
    import pod5
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("pod5 is required for this script") from exc


EPS = 1e-9
NUM_BASE_BITS = 2
NUM_BASES = 1 << NUM_BASE_BITS
ALPHABET = np.array(["A", "C", "G", "T"])
DEFAULT_BEAM_WIDTH = 32
DEFAULT_BEAM_CUT = 100.0
CRC32C_POLYNOMIAL = 0x82F63B78
CRC32C_SEED = 0x12345678
HASH_PRESENT_BITS = 4096
HASH_PRESENT_MASK = HASH_PRESENT_BITS - 1
PORE_LEVELS = {
    "FLO_FLG114": 200.0,
    "FLO_FLG114HD": 200.0,
    "FLO_MIN004RA": 195.50,
    "FLO_PRO004RA": 194.97,
    "FLO_MIN114": 197.61,
    "FLO_MIN114HD": 197.61,
    "FLO_PRO114": 199.21,
    "FLO_PRO114HD": 199.21,
    "FLO_PRO114M": 199.21,
}


@dataclass
class ReadData:
    read_id: str
    signal: np.ndarray
    sample_rate: int
    calibration_scale: float
    calibration_offset: float
    open_pore_level: float
    flow_cell_product_code: str


@dataclass
class DecodedChunk:
    input_offset: int
    raw_chunk_size: int
    seq: str
    qstring: str
    moves: list[int]


@dataclass
class BeamElement:
    state: int
    prev_element_index: int
    stay: bool


@dataclass
class BeamFrontElement:
    hash: int
    state: int
    prev_element_index: int
    stay: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny reference basecaller from POD5 to FASTQ using ONNX Runtime CPU."
    )
    parser.add_argument("--model", required=True, help="Dorado model name or local model directory")
    parser.add_argument("--pod5", required=True, help="input POD5 file")
    parser.add_argument(
        "--models-directory",
        default="models",
        help="directory used to cache downloaded models when --model is a model name",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for ONNX export, reports, and output FASTQ",
    )
    parser.add_argument("--read-id", default="", help="optional read id to basecall")
    parser.add_argument("--onnx", default="", help="use an existing ONNX model instead of exporting")
    parser.add_argument(
        "--compare-dorado",
        default="",
        help="path to a Dorado binary for sequence/qstring comparison",
    )
    parser.add_argument(
        "--skip-model-compatibility-check",
        action="store_true",
        help="pass through to Dorado when comparing against a fixture that does not match the model metadata",
    )
    return parser.parse_args()


def read_pod5_record(pod5_path: Path, read_id: str) -> ReadData:
    with pod5.Reader(pod5_path) as reader:
        reads = reader.reads(selection=[read_id]) if read_id else reader.reads()
        record = next(iter(reads))
        return ReadData(
            read_id=str(record.read_id),
            signal=np.asarray(record.signal, dtype=np.int16),
            sample_rate=int(record.run_info.sample_rate),
            calibration_scale=float(record.calibration.scale),
            calibration_offset=float(record.calibration.offset),
            open_pore_level=float(record.open_pore_level),
            flow_cell_product_code=str(record.run_info.flow_cell_product_code),
        )


def normalize_flowcell_code(code: str) -> str:
    return code.replace("-", "_").upper()


def determine_rna_adapter_pos(signal: np.ndarray) -> int:
    window_size = 250
    stride = 50
    median_diff = 125
    median_diff_diff_only = 150
    min_median_for_rna_signal = 700
    signal_len = int(signal.shape[0])
    medians = [0] * 5
    window_pos = [0] * 5
    median_pos = 0
    signal_start = 1000
    signal_end = 3 * signal_len // 4
    for i in range(signal_start, signal_end, stride):
        median = int(np.median(signal[i : min(i + window_size, signal_len)]))
        idx = median_pos % len(medians)
        medians[idx] = median
        window_pos[idx] = median_pos
        min_pos = int(np.argmin(medians))
        max_pos = int(np.argmax(medians))
        min_median = medians[min_pos]
        max_median = medians[max_pos]
        if (
            median_pos >= len(medians)
            and window_pos[max_pos] > window_pos[min_pos]
            and (
                (max_median > min_median_for_rna_signal and (max_median - min_median > median_diff))
                or (max_median - min_median > median_diff_diff_only)
            )
        ):
            return i
        median_pos += 1
    return 0


def quantile_counting(signal: np.ndarray, quantiles: list[float]) -> list[float]:
    signal_i16 = np.asarray(signal, dtype=np.int16)
    range_min = int(signal_i16.min())
    range_max = int(signal_i16.max())
    counts = np.bincount(signal_i16 - range_min, minlength=range_max - range_min + 1)
    cumulative = np.cumsum(counts)
    size = int(signal_i16.size)
    out = []
    for q in quantiles:
        threshold = int(q * (size - 1))
        idx = int(np.searchsorted(cumulative, threshold + 1, side="left"))
        out.append(float(idx + range_min))
    return out


def med_mad(signal: np.ndarray) -> tuple[float, float]:
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)) * 1.4826 + EPS)
    return med, mad


def trim(signal: np.ndarray, threshold: float = 2.4, window_size: int = 40, min_elements: int = 3) -> int:
    min_trim = 10
    num_samples = int(signal.shape[0]) - min_trim
    num_windows = num_samples // window_size
    seen_peak = False
    signal_f32 = np.asarray(signal, dtype=np.float32)
    for pos in range(num_windows):
        start = pos * window_size + min_trim
        end = start + window_size
        num_large_enough = int(np.count_nonzero(signal_f32[start:end] > threshold))
        if num_large_enough > min_elements or seen_peak:
            seen_peak = True
            if signal_f32[end - 1] > threshold:
                continue
            return min_trim if end >= num_samples else end
    return min_trim


def preprocess_signal(read: ReadData, config: ModelConfig) -> tuple[np.ndarray, int]:
    signal_i16 = np.asarray(read.signal, dtype=np.int16)
    is_rna = (config.sample_type or "").upper().startswith("RNA")
    trim_start = 0
    rna_adapter_end_signal_pos = 0

    if is_rna:
        trim_start = determine_rna_adapter_pos(signal_i16)
        if trim_start < signal_i16.size:
            signal_i16 = signal_i16[trim_start:]
            rna_adapter_end_signal_pos = 0
        else:
            rna_adapter_end_signal_pos = trim_start
            trim_start = 0

    scale = 1.0
    shift = 0.0
    open_pore_adjustment = 0.0
    preprocessing = config.signal_preprocessing
    scaling = preprocessing.get("scaling", {})
    strategy = scaling.get("strategy", "quantile").lower()
    standardisation = preprocessing.get("standardisation", {})

    if strategy == "pa":
        standardise = int(standardisation.get("standardise", 0)) > 0
        if standardise:
            mean = float(standardisation["mean"])
            stdev = float(standardisation["stdev"])
            scale = stdev / read.calibration_scale
            shift = (mean / read.calibration_scale) - read.calibration_offset
        else:
            scale = 1.0 / read.calibration_scale
            shift = -read.calibration_offset

        flowcell_code = normalize_flowcell_code(read.flow_cell_product_code)
        if not math.isnan(read.open_pore_level) and flowcell_code in PORE_LEVELS:
            expected = PORE_LEVELS[flowcell_code]
            if expected != 0:
                open_pore_adjustment = (read.open_pore_level - expected) / read.calibration_scale
    else:
        scaling_data = signal_i16[rna_adapter_end_signal_pos:]
        if strategy == "quantile":
            norm = preprocessing.get("normalisation", {})
            q_a, q_b = quantile_counting(
                scaling_data,
                [float(norm.get("quantile_a", 0.2)), float(norm.get("quantile_b", 0.9))],
            )
            shift = max(10.0, float(norm.get("shift_multiplier", 0.51)) * (q_a + q_b))
            scale = max(1.0, float(norm.get("scale_multiplier", 0.53)) * (q_b - q_a))
        else:
            shift, scale = med_mad(scaling_data.astype(np.float32))

    signal = (signal_i16.astype(np.float32) - (shift + open_pore_adjustment)) / scale

    if not is_rna:
        if trim_start == 0 and int(standardisation.get("standardise", 0)) > 0:
            trim_start = 10
        elif trim_start == 0:
            max_samples = min(8000, signal.shape[0] // 2)
            trim_start = trim(signal[:max_samples])
        if trim_start < signal.shape[0]:
            signal = signal[trim_start:]
        else:
            trim_start = 0

    return signal.astype(np.float32, copy=False), trim_start


def generate_chunks(num_samples: int, chunk_size: int, stride: int, overlap: int) -> list[int]:
    if num_samples == 0:
        raise ValueError("empty read")
    offsets = [0]
    offset = 0
    last_offset = max(num_samples - chunk_size, 0)
    misalignment = last_offset % stride
    if misalignment:
        last_offset += stride - misalignment
    chunk_step = chunk_size - overlap
    while offset + chunk_size < num_samples:
        offset = min(offset + chunk_step, last_offset)
        offsets.append(offset)
    return offsets


def make_chunk(signal: np.ndarray, offset: int, chunk_size: int) -> np.ndarray:
    chunk = signal[offset : offset + chunk_size]
    if chunk.shape[0] == chunk_size:
        return chunk
    out = np.zeros(chunk_size, dtype=np.float32)
    out[: chunk.shape[0]] = chunk
    return out


def build_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def log_sum_exp(x: float, y: float) -> float:
    abs_diff = abs(x - y)
    return max(x, y) + (math.log1p(math.exp(-abs_diff)) if abs_diff < 17.0 else 0.0)


def crc32c(crc: int, new_bits: int, num_new_bits: int) -> int:
    for _ in range(num_new_bits):
        b = (new_bits ^ crc) & 1
        crc >>= 1
        if b:
            crc ^= CRC32C_POLYNOMIAL
        new_bits >>= 1
    return crc


def forward_scores(scores_tc: np.ndarray, blank_score: float) -> np.ndarray:
    t, c = scores_tc.shape
    state_len = round(math.log(c, NUM_BASES) - 1)
    num_states = NUM_BASES**state_len
    ms = scores_tc.reshape(t, num_states, NUM_BASES)
    alpha = np.full((t + 1, num_states), -1e38, dtype=np.float32)
    alpha[0] = 0.0
    idx = np.arange(num_states).repeat(NUM_BASES).reshape(NUM_BASES, -1).T
    for i in range(t):
        scored_steps = alpha[i, idx] + ms[i]
        scored_stay = (alpha[i] + blank_score)[:, None]
        scored = np.concatenate([scored_stay, scored_steps], axis=-1)
        m = scored.max(axis=-1, keepdims=True)
        alpha[i + 1] = (m[:, 0] + np.log(np.exp(scored - m).sum(axis=-1))).astype(np.float32)
    return alpha


def backward_scores(scores_tc: np.ndarray, blank_score: float) -> np.ndarray:
    t, c = scores_tc.shape
    state_len = round(math.log(c, NUM_BASES) - 1)
    num_states = NUM_BASES**state_len
    idx = np.arange(num_states).repeat(NUM_BASES).reshape(NUM_BASES, -1).T
    idx_t = np.argsort(idx.reshape(-1)).reshape(idx.shape)
    ms_t = scores_tc[:, idx_t]
    idx_t = idx_t >> 2
    beta = np.full((t + 1, num_states), -1e38, dtype=np.float32)
    beta[0] = 0.0
    ms = ms_t[::-1].reshape(t, num_states, NUM_BASES)
    for i in range(t):
        scored_steps = beta[i, idx_t] + ms[i]
        scored_stay = (beta[i] + blank_score)[:, None]
        scored = np.concatenate([scored_stay, scored_steps], axis=-1)
        m = scored.max(axis=-1, keepdims=True)
        beta[i + 1] = (m[:, 0] + np.log(np.exp(scored - m).sum(axis=-1))).astype(np.float32)
    return beta[::-1]


def generate_sequence(moves: list[int], states: list[int], qual_data: np.ndarray, shift: float, scale: float) -> tuple[str, str]:
    seq_len = sum(moves)
    sequence = ["N"] * seq_len
    qstring = ["!"] * seq_len
    base_probs = np.zeros(seq_len, dtype=np.float32)
    total_probs = np.zeros(seq_len, dtype=np.float32)
    seq_pos = 0
    for blk, move in enumerate(moves):
        state = states[blk]
        base = state & 3
        offset = 0 if blk == 0 else move - 1
        prob_pos = seq_pos + offset
        base_probs[prob_pos] += qual_data[blk, base]
        total_probs[prob_pos] += qual_data[blk].sum()
        if blk == 0:
            sequence[seq_pos] = ALPHABET[base]
            seq_pos += 1
        else:
            for _ in range(move):
                sequence[seq_pos] = ALPHABET[base]
                seq_pos += 1
    for i in range(seq_len):
        p = 1.0 - (base_probs[i] / total_probs[i])
        p = max(p, 1e-12)
        qscore = (-10.0 * math.log10(p)) * scale + shift
        qscore = min(max(qscore, 1.0), 50.0)
        qstring[i] = chr(int(33.5 + qscore))
    return "".join(sequence), "".join(qstring)


def beam_search_decode(
    scores_tc: np.ndarray,
    blank_score: float,
    q_shift: float,
    q_scale: float,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_cut: float = DEFAULT_BEAM_CUT,
) -> tuple[str, str, list[int]]:
    num_blocks, num_trans_states = scores_tc.shape
    if num_trans_states % NUM_BASES != 0:
        raise ValueError("unexpected number of transition states")
    num_states = num_trans_states // NUM_BASES
    num_state_bits = int(round(math.log2(num_states)))
    if 1 << num_state_bits != num_states:
        raise ValueError("num_states must be power of 2")

    fwd = forward_scores(scores_tc, blank_score)
    bwd = backward_scores(scores_tc, blank_score)
    posts = np.exp((fwd + bwd) - np.max(fwd + bwd, axis=-1, keepdims=True))
    posts /= posts.sum(axis=-1, keepdims=True)
    back_guide = bwd
    states_mask = num_states - 1
    log_beam_cut = math.log(beam_cut) if beam_cut > 0.0 else float("inf")

    beam_vector = [BeamElement(0, 0, False) for _ in range(beam_width * (num_blocks + 1))]
    current_beam_front: list[BeamFrontElement] = [BeamFrontElement(0, 0, 0, False)] * ((NUM_BASES + 1) * beam_width)
    prev_beam_front: list[BeamFrontElement] = [BeamFrontElement(0, 0, 0, False)] * ((NUM_BASES + 1) * beam_width)
    current_scores = np.full((NUM_BASES + 1) * beam_width, -np.inf, dtype=np.float32)
    prev_scores = np.full((NUM_BASES + 1) * beam_width, -np.inf, dtype=np.float32)

    init_threshold = -np.inf
    if beam_width < num_states:
        sorted_guides = np.sort(back_guide[0])[::-1]
        init_threshold = float(sorted_guides[beam_width - 1])

    beam_element = 0
    for state in range(num_states):
        if beam_element >= beam_width:
            break
        if back_guide[0, state] >= init_threshold:
            prev_beam_front[beam_element] = BeamFrontElement(crc32c(CRC32C_SEED, state, 32), state, 0, False)
            prev_scores[beam_element] = 0.0
            beam_vector[beam_element] = BeamElement(state, 0, False)
            beam_element += 1
    current_beam_width = min(beam_width, num_states)

    for block_idx in range(num_blocks):
        block_scores = scores_tc[block_idx]
        block_back_scores = back_guide[block_idx + 1]
        step_hash_present = np.zeros(HASH_PRESENT_BITS, dtype=np.bool_)
        new_elem_count = 0
        max_score = -np.inf

        for prev_idx in range(current_beam_width):
            prev_elem = prev_beam_front[prev_idx]
            for new_base in range(NUM_BASES):
                new_state = ((prev_elem.state << NUM_BASE_BITS) & states_mask) | new_base
                move_idx = (new_state << NUM_BASE_BITS) + (((prev_elem.state << NUM_BASE_BITS) >> num_state_bits))
                new_score = float(prev_scores[prev_idx] + block_scores[move_idx] + block_back_scores[new_state])
                new_hash = crc32c(prev_elem.hash, new_base, NUM_BASE_BITS)
                step_hash_present[new_hash & HASH_PRESENT_MASK] = True
                current_beam_front[new_elem_count] = BeamFrontElement(new_hash, new_state, prev_idx, False)
                current_scores[new_elem_count] = new_score
                max_score = max(max_score, new_score)
                new_elem_count += 1

        for prev_idx in range(current_beam_width):
            prev_elem = prev_beam_front[prev_idx]
            stay_score = float(prev_scores[prev_idx] + blank_score + block_back_scores[prev_elem.state])
            current_beam_front[new_elem_count] = BeamFrontElement(prev_elem.hash, prev_elem.state, prev_idx, True)
            current_scores[new_elem_count] = stay_score
            max_score = max(max_score, stay_score)
            if step_hash_present[prev_elem.hash & HASH_PRESENT_MASK]:
                stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_idx
                stay_latest_base = prev_elem.state & 3
                for prev_comp_idx in range(current_beam_width):
                    step_elem_idx = (prev_comp_idx << NUM_BASE_BITS) | stay_latest_base
                    if current_beam_front[stay_elem_idx].hash == current_beam_front[step_elem_idx].hash:
                        folded = log_sum_exp(float(current_scores[stay_elem_idx]), float(current_scores[step_elem_idx]))
                        if current_scores[stay_elem_idx] > current_scores[step_elem_idx]:
                            current_scores[stay_elem_idx] = folded
                            current_scores[step_elem_idx] = -np.inf
                        else:
                            current_scores[step_elem_idx] = folded
                            current_scores[stay_elem_idx] = -np.inf
                        max_score = max(max_score, folded)
            new_elem_count += 1

        beam_cutoff_score = max_score - log_beam_cut
        mask = current_scores[:new_elem_count] >= beam_cutoff_score
        elem_count = int(mask.sum())
        if elem_count > beam_width:
            min_beam_width = (beam_width * 8) // 10
            low_score = beam_cutoff_score
            hi_score = max_score
            num_guesses = 1
            while (elem_count > beam_width or elem_count < min_beam_width) and num_guesses < 10:
                if elem_count > beam_width:
                    low_score = beam_cutoff_score
                    beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0
                else:
                    hi_score = beam_cutoff_score
                    beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0
                mask = current_scores[:new_elem_count] >= beam_cutoff_score
                elem_count = int(mask.sum())
                num_guesses += 1
            if num_guesses == 10:
                beam_cutoff_score = hi_score
                mask = current_scores[:new_elem_count] >= beam_cutoff_score
                elem_count = int(mask.sum())
            elem_count = min(elem_count, beam_width)

        write_idx = 0
        for read_idx in range(new_elem_count):
            if current_scores[read_idx] >= beam_cutoff_score:
                if write_idx < beam_width:
                    prev_beam_front[write_idx] = current_beam_front[read_idx]
                    prev_scores[write_idx] = current_scores[read_idx]
                    write_idx += 1
                else:
                    break

        if block_idx == num_blocks - 1:
            best_idx = int(np.argmax(prev_scores[:elem_count]))
            prev_beam_front[0], prev_beam_front[best_idx] = prev_beam_front[best_idx], prev_beam_front[0]
            prev_scores[0], prev_scores[best_idx] = prev_scores[best_idx], prev_scores[0]

        beam_offset = (block_idx + 1) * beam_width
        for i in range(elem_count):
            prev_scores[i] -= block_back_scores[prev_beam_front[i].state]
            beam_vector[beam_offset + i] = BeamElement(
                prev_beam_front[i].state,
                prev_beam_front[i].prev_element_index,
                prev_beam_front[i].stay,
            )
        current_beam_width = elem_count

    states = [0] * num_blocks
    moves = [0] * num_blocks
    element_index = 0
    for beam_idx in range(num_blocks, 0, -1):
        beam_addr = beam_idx * beam_width + element_index
        states[beam_idx - 1] = beam_vector[beam_addr].state
        moves[beam_idx - 1] = 0 if beam_vector[beam_addr].stay else 1
        element_index = beam_vector[beam_addr].prev_element_index
    moves[0] = 1

    qual_data = np.zeros((num_blocks, NUM_BASES), dtype=np.float32)
    shifted_states = [0] * (2 * NUM_BASES)
    for block_idx in range(num_blocks):
        state = states[block_idx]
        states[block_idx] = state % NUM_BASES
        base_to_emit = states[block_idx]
        timestep_posts = posts[block_idx + 1]
        block_prob = float(timestep_posts[state])
        l_shift_idx = state >> NUM_BASE_BITS
        r_shift_idx = (state << NUM_BASE_BITS) % num_states
        msb = num_states >> NUM_BASE_BITS
        for shift_base in range(NUM_BASES):
            shifted_states[2 * shift_base] = l_shift_idx + msb * shift_base
            shifted_states[2 * shift_base + 1] = r_shift_idx + shift_base
        for state_idx, candidate_state in enumerate(shifted_states):
            count_state = candidate_state != state and candidate_state not in shifted_states[:state_idx]
            if count_state:
                block_prob += float(timestep_posts[candidate_state])
        block_prob = min(max(block_prob, 0.0), 1.0) ** 0.4
        wrong_base_prob = (1.0 - block_prob) / 3.0
        for base in range(NUM_BASES):
            qual_data[block_idx, base] = block_prob if base == base_to_emit else wrong_base_prob

    sequence, qstring = generate_sequence(moves, states, qual_data, q_shift, q_scale)
    return sequence, qstring, moves


def stitch_chunks(raw_signal_len: int, stride_inner: int, called_chunks: list[DecodedChunk]) -> tuple[str, str, list[int]]:
    start_pos = 0
    mid_point_front = 0
    moves: list[int] = []
    sequences: list[str] = []
    qstrings: list[str] = []

    for i in range(len(called_chunks) - 1):
        current_chunk = called_chunks[i]
        next_chunk = called_chunks[i + 1]
        overlap_size = (current_chunk.raw_chunk_size + current_chunk.input_offset) - next_chunk.input_offset
        overlap_down_sampled = overlap_size // stride_inner
        mid_point_rear = overlap_down_sampled // 2

        current_chunk_bases_to_trim = sum(current_chunk.moves[-mid_point_rear:]) if mid_point_rear else 0
        end_pos = len(current_chunk.seq) - current_chunk_bases_to_trim
        trimmed_len = end_pos - start_pos
        sequences.append(current_chunk.seq[start_pos : start_pos + trimmed_len])
        qstrings.append(current_chunk.qstring[start_pos : start_pos + trimmed_len])
        moves.extend(current_chunk.moves[mid_point_front : len(current_chunk.moves) - mid_point_rear])

        mid_point_front = overlap_down_sampled - mid_point_rear
        start_pos = sum(next_chunk.moves[:mid_point_front])

    last_chunk = called_chunks[-1]
    moves.extend(last_chunk.moves[mid_point_front:])
    if len(called_chunks) == 1:
        last_index = raw_signal_len // stride_inner
        moves = moves[:last_index]
        end = sum(moves)
        sequences.append(last_chunk.seq[start_pos : start_pos + end])
        qstrings.append(last_chunk.qstring[start_pos : start_pos + end])
    else:
        sequences.append(last_chunk.seq[start_pos:])
        qstrings.append(last_chunk.qstring[start_pos:])

    seq = "".join(sequences)
    qstring = "".join(qstrings)
    if len(moves) > raw_signal_len // stride_inner:
        if moves[-1] == 1:
            seq = seq[:-1]
            qstring = qstring[:-1]
        moves.pop()
    return seq, qstring, moves


def run_reference_basecaller(
    model_dir: Path,
    onnx_path: Path,
    read: ReadData,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_model_config(model_dir)
    if config.num_features != 1:
        raise NotImplementedError("This reference implementation only supports simplex models with one signal feature")

    session = build_session(onnx_path)
    signal, trim_start = preprocess_signal(read, config)
    chunk_size = config.normalized_chunk_size()
    overlap = max(1, config.basecaller_overlap // config.stride_inner()) * config.stride_inner()
    offsets = generate_chunks(int(signal.shape[0]), chunk_size, config.stride_inner(), overlap)

    called_chunks: list[DecodedChunk] = []
    for offset in offsets:
        chunk = make_chunk(signal, offset, chunk_size)
        scores = session.run(["scores"], {"input": chunk.reshape(1, 1, chunk_size)})[0][0]
        seq, qstring, moves = beam_search_decode(scores, config.blank_score, config.qbias, config.qscale)
        called_chunks.append(
            DecodedChunk(
                input_offset=offset,
                raw_chunk_size=chunk_size,
                seq=seq,
                qstring=qstring,
                moves=list(moves),
            )
        )

    seq, qstring, moves = stitch_chunks(int(signal.shape[0]), config.stride_inner(), called_chunks)
    result = {
        "read_id": read.read_id,
        "model_name": config.model_name,
        "trimmed_samples": trim_start,
        "raw_samples_after_trim": int(signal.shape[0]),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "stride_inner": config.stride_inner(),
        "num_chunks": len(called_chunks),
        "sequence": seq,
        "qstring": qstring,
        "moves": moves,
    }
    (output_dir / "reference_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "reference.fastq").write_text(f"@{read.read_id}\n{seq}\n+\n{qstring}\n", encoding="utf-8")
    return result


def ensure_onnx(model_dir: Path, onnx_arg: str, output_dir: Path) -> Path:
    if onnx_arg:
        return Path(onnx_arg).resolve()
    onnx_path = output_dir / "model.onnx"
    if onnx_path.exists():
        return onnx_path
    config = load_model_config(model_dir)
    weights = load_model_weights(config)
    model = build_model(config)
    load_weights_into_model(model, weights)
    chunk_size = config.normalized_chunk_size()
    export_model(model, config, output_dir, "model.onnx", 17, chunk_size)
    write_manifest(output_dir, config, onnx_path, chunk_size, 17)
    return onnx_path


def compare_with_dorado(
    dorado_bin: Path,
    model_dir: Path,
    pod5_path: Path,
    read_id: str,
    models_directory: Path,
    skip_compatibility_check: bool,
    output_dir: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dorado-ref-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_fastq = tmpdir_path / "dorado.fastq"
        cmd = [
            str(dorado_bin),
            "basecaller",
            str(model_dir),
            str(pod5_path),
            "--emit-fastq",
            "--read-ids",
            str((tmpdir_path / "reads.txt")),
            "--models-directory",
            str(models_directory),
            "-x",
            "cpu",
        ]
        (tmpdir_path / "reads.txt").write_text(read_id + "\n", encoding="utf-8")
        if skip_compatibility_check:
            cmd.append("--skip-model-compatibility-check")
        with out_fastq.open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.PIPE, text=True)
        lines = out_fastq.read_text(encoding="utf-8").splitlines()
        if len(lines) < 4:
            raise RuntimeError("Dorado comparison produced empty FASTQ output")
        result = {
            "read_id": lines[0][1:],
            "sequence": lines[1],
            "qstring": lines[3],
        }
        (output_dir / "dorado_compare.fastq").write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
        return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_path(args.model, Path(args.models_directory), False)
    onnx_path = ensure_onnx(model_dir, args.onnx, output_dir)
    read = read_pod5_record(Path(args.pod5).resolve(), args.read_id)
    result = run_reference_basecaller(model_dir, onnx_path, read, output_dir)

    report: dict[str, Any] = {
        "reference": {
            "read_id": result["read_id"],
            "sequence_length": len(result["sequence"]),
            "qstring_length": len(result["qstring"]),
            "chunk_size": result["chunk_size"],
            "num_chunks": result["num_chunks"],
        }
    }

    if args.compare_dorado:
        dorado_result = compare_with_dorado(
            dorado_bin=Path(args.compare_dorado).resolve(),
            model_dir=model_dir,
            pod5_path=Path(args.pod5).resolve(),
            read_id=read.read_id,
            models_directory=Path(args.models_directory).resolve(),
            skip_compatibility_check=args.skip_model_compatibility_check,
            output_dir=output_dir,
        )
        report["compare_dorado"] = {
            "read_id_match": result["read_id"] == dorado_result["read_id"],
            "sequence_match": result["sequence"] == dorado_result["sequence"],
            "qstring_match": result["qstring"] == dorado_result["qstring"],
            "reference_length": len(result["sequence"]),
            "dorado_length": len(dorado_result["sequence"]),
        }

    (output_dir / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
