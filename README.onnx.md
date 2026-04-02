# Dorado ONNX Export And Basecaller Reconstruction

This fork adds a CPU-only path for exporting Dorado simplex basecaller networks to ONNX without building the CUDA or ROCm runtime. The exported ONNX graph is only the neural network forward pass. Recreating Dorado basecalling still requires the signal preprocessing, chunking, CRF decoding, and chunk stitching steps described below.

## Scope

What is exported:
- the simplex neural network that maps normalized signal chunks to score tensors
- legacy and current LSTM/CRF models
- transformer simplex models

What is not exported:
- the CRF decoder
- qscore calibration logic outside the decoder
- chunk stitching
- BAM / FASTQ formatting
- modified-base, duplex, polish, or variant models

## Artifacts

`tools/dorado_export_onnx.py` writes:
- `model.onnx`
- `export_manifest.json`

The manifest records the metadata you need to drive an external runtime:
- `architecture`
- `num_features`
- `chunk_size`
- `stride_inner`
- `sample_rate`
- `sample_type`
- `signal_preprocessing`

The ONNX graph uses:
- input tensor `input` with shape `[batch, features, samples]`
- output tensor `scores` with shape `[batch, frames, states]`

`frames` is the downsampled time dimension after the encoder. `states` is the CRF state space width.

## POD5 To Model Input

### 1. Read POD5 signal and calibration metadata

Dorado loads the complete read signal from POD5 as `int16` samples and also keeps the run sample rate plus per-read calibration fields such as `calibration_scale`, `calibration_offset`, and `open_pore_level`. See:
- `dorado/data_loader/DataLoader.cpp`
- `dorado/read_pipeline/base/include/read_pipeline/base/messages.h`

If you are rebuilding this externally, your starting point is:
- raw `int16` signal samples from POD5
- sample rate from POD5 run info
- calibration scale and offset from POD5 read metadata
- model `config.toml`

### 2. Parse `config.toml`

The model config decides how the raw signal must be normalized before inference. Relevant sections:
- `run_info`
- `basecaller`
- `scaling`
- `normalisation`
- `standardisation`
- `input`
- `encoder` or `model.encoder.*`
- `qscore`

This fork preserves the preprocessing sections in `export_manifest.json` so an external runner can reconstruct the same front-end assumptions.

### 3. RNA adapter handling before scaling

RNA models can trim the RNA adapter before normalization by scanning for a strong median shift in the early signal. If the adapter is not physically trimmed, Dorado still records where the adapter ends so the scaling step can ignore it. See `dorado/read_pipeline/nodes/ScalerNode.cpp`.

For DNA models, this RNA-specific step does not apply.

### 4. Normalize the raw samples

Dorado uses one of these strategies from `config.toml`:

`scaling.strategy = pa`
- convert ADC samples into pore-current units using POD5 calibration scale and offset
- if `standardisation.standardise = 1`, apply `(x_pa - mean) / stdev`
- for supported flowcells, Dorado can also apply an open-pore correction

`scaling.strategy = quantile`
- compute two quantiles over the scaling region
- derive `shift` and `scale` from `normalisation.quantile_a`, `quantile_b`, `shift_multiplier`, and `scale_multiplier`
- normalize with `(x - shift) / scale`

Legacy median/MAD path
- some older configs do not use the newer `scaling` section
- Dorado falls back to median and MAD based normalization

Implementation reference:
- `dorado/config/BasecallModelConfig.cpp`
- `dorado/read_pipeline/nodes/ScalerNode.cpp`

Important implementation detail:
- Dorado converts the `int16` signal to `float16` in place after computing the shift and scale
- this exporter uses `float32` for ONNX export and validation
- for external runtimes, `float32` input is fine as long as the arithmetic matches

### 5. Trim the read start

After normalization, Dorado trims DNA reads at the start to remove the open-pore / adapter region:
- if pA standardisation is active, it uses a fixed trim of 10 samples
- otherwise it applies the standard trim heuristic over up to the first 8000 samples

RNA reads skip this DNA trim because the signal characteristics differ.

### 6. Chunk the normalized signal

Basecalling works on overlapping signal chunks in raw-sample space. Dorado uses:
- `chunk_size` from the basecaller config
- `overlap` from the basecaller config
- model stride for alignment

Rules from `dorado/read_pipeline/base/chunk.cpp`:
- `chunk_size` and `overlap` must align to stride
- the final chunk start is rounded up to the next stride boundary
- if the final chunk extends beyond the real read length, Dorado conceptually zero-pads the overhang

This fork exposes two useful chunk-related values:
- `chunk_size`: the export example size
- `stride_inner`: the raw-sample step size represented by one decoder frame

Chunk alignment rules for external inference:
- LSTM exports: chunk length must be divisible by `stride_inner`
- transformer exports: chunk length must be divisible by `stride_inner * 16`

### 7. Feed the ONNX model

The model input is the normalized chunk tensor:

```text
[batch, features, samples]
```

For simplex basecallers in this fork:
- `features` is usually 1
- `samples` is the raw chunk length after scaling and trimming

The ONNX output is:

```text
[batch, frames, states]
```

`frames` is the number of downsampled time blocks.

## Model Output To Basecalls

### 1. Interpret the exported scores

The ONNX output is the same tensor Dorado feeds into its CRF decoder, before `ModelRunner` transposes it to `TNC` form for batch decoding. For a single chunk, you can decode either:
- `[frames, states]` directly
- or transpose the batched output to Dorado's `TNC` layout

The decoder also needs:
- `blank_score`
- `state_len`
- qscore calibration `qscore.bias` and `qscore.scale`
- beam-search settings such as beam width and beam cut

These come from the model config, not from the ONNX graph itself.

### 2. Compute CRF helper tensors

Before beam search, Dorado computes:
- forward scores
- backward scores
- posterior probabilities `softmax(fwd + bwd)`

See:
- `dorado/basecall/decode/CPUDecoder.cpp`
- `dorado/basecall/crf_utils.h`

### 3. Run beam search

Dorado then runs CRF beam search over the per-frame scores to produce:
- `sequence`
- `qstring`
- `moves`

The move table has one value per decoder frame and indicates whether that frame emitted a new base.

Sequence/Q-score generation details:
- bases are reconstructed from the decoded CRF state path
- qualities are derived from the posterior probabilities, then mapped through `qscore.scale` and `qscore.bias`

See:
- `dorado/basecall/decode/beam_search.cpp`

### 4. Decode each chunk independently

Each chunk is decoded on its own. Dorado stores one decoded result per chunk:
- `sequence`
- `qstring`
- `moves`

See:
- `dorado/basecall/ModelRunner.cpp`
- `dorado/basecall/DecodedChunk.h`

### 5. Stitch overlapping chunks back into one read

Dorado does not just concatenate chunk outputs. It trims each overlap at the midpoint in move-space:
- take the rear half of the overlap from the earlier chunk
- take the front half from the later chunk
- trim sequence and qstring by summing moves over the trimmed frames
- concatenate trimmed sequence, qstring, and move segments

Special case:
- if the read is shorter than one chunk, Dorado truncates the final move table and sequence to the real read length

See:
- `dorado/read_pipeline/base/stitch.cpp`

### 6. Final read-level adjustments

After stitching:
- Dorado records `model_stride`, `model_q_bias`, and `model_q_scale`
- reads affected by mux changes can be end-trimmed
- RNA reads have sequence and qstring reversed after stitching

See:
- `dorado/read_pipeline/nodes/BasecallerNode.cpp`

## Recreating A Simplex Basecaller Outside Dorado

Minimum external pipeline:

1. Load raw `int16` signal and POD5 calibration metadata.
2. Parse Dorado `config.toml`.
3. Apply the same scaling and trimming rules.
4. Create overlapping signal chunks aligned to the model stride.
5. Run ONNX inference on `[batch, features, samples]`.
6. Decode each chunk with Dorado-compatible CRF beam search.
7. Stitch chunk sequences using the move-table midpoint logic.
8. Reverse RNA outputs after stitching.

If you skip steps 6 or 7, you do not have a Dorado-equivalent basecaller. You only have the neural network front end.

## Validation In This Fork

Numerical validation utilities:
- `tools/dorado_validate_onnx.py`
- `tools/dorado_validate_onnx_matrix.py`
- `tools/dorado_reference_basecaller.py`

What the validator checks:
- ONNX graph passes `onnx.checker`
- ONNX Runtime CPU output matches the reconstructed PyTorch model
- dynamic sample length works across several aligned chunk sizes
- dynamic batch works across multiple batch sizes
- default acceptance uses `rtol=1e-4` and `atol=1.5e-3`; the wider absolute tolerance is for small recurrent-kernel drift seen on some older LSTM models
- each run writes `validation_report.json`

Example single-model validation:

```bash
python3 tools/dorado_validate_onnx.py \
  --model dna_r10.4.1_e8.2_400bps_hac@v5.2.0 \
  --models-directory ./models \
  --output-dir ./validate-hac
```

Example curated matrix:

```bash
python3 tools/dorado_validate_onnx_matrix.py \
  --models-directory ./models \
  --output-dir ./validation-matrix
```

Example tiny reference basecaller run:

```bash
python3 tools/dorado_reference_basecaller.py \
  --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
  --pod5 tests/data/pod5/dna_r10.4.1_e8.2_400bps_5khz/dna_r10.4.1_e8.2_400bps_5khz-FLO_PRO114M-SQK_RAD114-5000.pod5 \
  --models-directory ./models \
  --output-dir ./reference-run
```

Current scope of the reference script:
- simplex models only
- CPU ONNX Runtime only
- pure Python preprocessing, CRF decode, and stitching
- optional comparison against a Dorado binary if `--compare-dorado` is provided

The curated matrix covers:
- legacy DNA R9 models
- R10 HAC versions from v3.5.2 through v5.2.0
- fast, HAC, and SUP families
- RNA HAC and SUP representatives
- both LSTM and transformer architectures
