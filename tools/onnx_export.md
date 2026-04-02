`tools/dorado_export_onnx.py` exports the Dorado neural network forward pass to ONNX.

It supports:
- downloading a named Dorado model from the public CDN
- exporting a local downloaded model directory
- legacy and current LSTM/CRF simplex models
- transformer simplex models

It does not export Dorado's CRF decoder. The ONNX output is the per-frame score tensor before decoding.

The exporter writes `export_manifest.json` next to the ONNX file. That manifest carries:
- model identity and architecture
- export chunk size
- `stride_inner`, which is the raw-sample to decoder-frame stride
- `run_info` fields such as sample rate and sample type when present
- signal preprocessing metadata from `config.toml`, including `scaling`, `normalisation`, and `standardisation`

The ONNX graph uses:
- input name: `input`
- input shape: `[batch, features, samples]`
- output name: `scores`
- output shape: `[batch, frames, states]`
- dynamic axes for batch and sample length

Chunk-size rules:
- LSTM models must use a chunk size aligned to `stride_inner`
- transformer models must use a chunk size aligned to `stride_inner * 16`
- if you omit `--chunk-size`, the exporter picks a normalized aligned size derived from the model config

Example:

```bash
python3 tools/dorado_export_onnx.py \
  --model dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
  --models-directory ./models \
  --output-dir ./onnx-fast
```

Validate numerical parity against the reconstructed PyTorch model:

```bash
python3 tools/dorado_validate_onnx.py \
  --model dna_r10.4.1_e8.2_400bps_hac@v5.2.0 \
  --models-directory ./models \
  --output-dir ./validate-hac
```

Run the curated cross-version validation matrix:

```bash
python3 tools/dorado_validate_onnx_matrix.py \
  --models-directory ./models \
  --output-dir ./validation-matrix
```

Run the tiny end-to-end reference basecaller on a POD5 file with CPU ONNX Runtime:

```bash
python3 tools/dorado_reference_basecaller.py \
  --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
  --pod5 tests/data/pod5/dna_r10.4.1_e8.2_400bps_5khz/dna_r10.4.1_e8.2_400bps_5khz-FLO_PRO114M-SQK_RAD114-5000.pod5 \
  --models-directory ./models \
  --output-dir ./reference-run
```

For the full POD5-to-basecall reconstruction path around the exported ONNX model, see `README.onnx.md`.
