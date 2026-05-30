# Chatbot From Scratch: MiniGPT Training, Fine-Tuning, and Comparison

This repository is an end-to-end conversational NLP project built in PyTorch. It includes a custom decoder-only MiniGPT implementation, a pretrained dialogue baseline, and a fine-tuned GPT-style model so reviewers can see model implementation, data preparation, training, inference, and comparison work in one place.

## Why This Project Matters

This project shows practical AI/ML engineering work beyond notebook-only experimentation. The repository demonstrates:

- PyTorch model implementation for a decoder-only language model
- Dialogue data preparation and label masking for response-focused training
- Separate training and inference flows for custom and pretrained models
- Automated tests and saved artifacts that make the work inspectable

## What I Built

- A custom decoder-only `MiniGPTLM` in `src/model.py`
- A DailyDialog loading, formatting, tokenization, batching, and masking pipeline in `src/data.py`
- Track A: a pretrained `DialoGPT-medium` baseline in `src/track_a.py`
- Track B: custom MiniGPT training with checkpoint save/load in `src/train_track_b.py`
- Track C: pretrained GPT-style fine-tuning in `src/train_track_c_finetune.py`
- Interactive inference flows in `src/infer_track_b.py` and `src/infer_track_c.py`
- Qualitative output comparison scripts in `src/compare.py` and `src/compare_track_c.py`
- Automated tests in `tests/`

## Technical Highlights

- Token embeddings and learned positional embeddings for the custom MiniGPT model
- Masked multi-head self-attention with an upper-triangular causal mask
- Feed-forward MLP blocks with layer normalization, residual connections, GELU, and dropout
- Autoregressive token generation with temperature, top-k sampling, and repetition penalty controls
- Shifted logits/labels loss for next-token prediction in the custom model
- Response-only label masking in the data pipeline so prompt tokens do not contribute to loss
- Checkpoint save/load support for Track B training artifacts

## Repository Structure

- [`src/model.py`](src/model.py): custom decoder-only MiniGPT architecture and generation logic
- [`src/data.py`](src/data.py): DailyDialog loading, dialogue formatting, tokenization helpers, batching, and label masking
- [`src/train_track_b.py`](src/train_track_b.py): custom MiniGPT training loop and checkpoint persistence
- [`src/train_track_c_finetune.py`](src/train_track_c_finetune.py): pretrained GPT-style fine-tuning workflow
- [`src/infer_track_b.py`](src/infer_track_b.py): interactive inference for the custom model
- [`src/compare.py`](src/compare.py): qualitative baseline vs. custom-model comparison
- [`tests/`](tests): automated tests covering model, data, inference, and comparison behavior
- [`artifacts/`](artifacts): saved checkpoints, fine-tuned model files, comparison JSON outputs, and an attention heatmap

## How to Run

Use Python 3.10+ and run these commands from the repository root.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Train the custom MiniGPT model (Track B):

```bash
python -m src.train_track_b
```

Run inference with the custom model:

```bash
python -m src.infer_track_b
```

Fine-tune the pretrained GPT-style model (Track C):

```bash
python -m src.train_track_c_finetune
```

Run inference with the fine-tuned Track C model after dependencies and saved model assets are available:

```bash
python -m src.infer_track_c
```

These commands assume the dependencies from `requirements.txt` are installed; some workflows also require external model or dataset downloads and saved artifacts.

Generate qualitative comparisons after dependencies and required model assets are available:

```bash
python -m src.compare
python -m src.compare_track_c
```

Some runtime commands require external downloads from Hugging Face. Training uses DailyDialog plus tokenizer/model assets, Track C inference expects a saved fine-tuned model directory, and the comparison scripts also depend on the pretrained Track A baseline (`microsoft/DialoGPT-medium`).

## Evidence in This Repository

- Model implementation and training code are present in `src/`, including a custom decoder-only language model and separate training paths for Track B and Track C.
- Automated tests in [`tests/`](tests) cover core model behavior, data processing, inference helpers, and comparison outputs.
- A saved Track B checkpoint is included at [`artifacts/checkpoints/track_b_checkpoint.pt`](artifacts/checkpoints/track_b_checkpoint.pt).
- Fine-tuned Track C model files are included in [`artifacts/track_c_distilgpt2/`](artifacts/track_c_distilgpt2).
- Qualitative comparison outputs are included at [`artifacts/comparison.json`](artifacts/comparison.json) and [`artifacts/comparison_track_c.json`](artifacts/comparison_track_c.json).
- An attention visualization is included at [`artifacts/attention_heatmap.png`](artifacts/attention_heatmap.png).

## Limitations

- This is an educational and portfolio project, not a production-deployed chatbot system.
- The comparison scripts are useful for qualitative inspection, but they are not a rigorous benchmark or evaluation study.
- Output quality depends on small-scale training and fine-tuning choices such as dataset subset size, model size, training duration, and decoding settings.
- The training scripts depend on external model and dataset downloads, so reproducibility also depends on the local environment and available compute.
