## Single `pip install` command

```bash
pip install \
  "transformers>=4.56.0" \
  "peft>=0.11.0" \
  "trl>=0.9.0" \
  "datasets>=2.19.0" \
  "accelerate>=0.30.0" \
  "torch>=2.3.0" \
  "sacrebleu>=2.4.0" \
  "fastapi>=0.111.0" \
  "uvicorn[standard]>=0.29.0" \
  "pydantic>=2.0.0" \
  "pyyaml>=6.0"
```

> `bitsandbytes` is only needed if you later want to run `--load_in_4bit` on a GPU. It is not required for the smoke-test or demo modes.

---

## Step-by-step: testing on Windows with WSL + Docker Desktop

### Part 1 — Get the project into WSL

Open **Windows Terminal** and start a WSL session (Ubuntu is assumed):

```bash
wsl
```

Navigate to your home directory and copy the project in. If you are working from the Claude outputs folder, it will be somewhere on your Windows drive, accessible in WSL at `/mnt/c/…`:

```bash
# Example: your project sits in C:\Users\YourName\Downloads\hack-apertus-demo
cp -r /mnt/c/Users/YourName/Downloads/hack-apertus-demo ~/hack-apertus-demo
cd ~/hack-apertus-demo
```

---

### Part 2 — Python environment

WSL ships with Python 3, but you want an isolated environment so the ML packages don't pollute your system Python.

```bash
# Confirm Python version (needs 3.10+)
python3 --version

# Create a virtual environment
python3 -m venv .venv

# Activate it — you'll need to do this every time you open a new WSL terminal
source .venv/bin/activate

# Your prompt should now show (.venv) — confirm:
which python     # should print ~/hack-apertus-demo/.venv/bin/python
```

Now install everything:

```bash
pip install --upgrade pip

pip install \
  "transformers>=4.56.0" \
  "peft>=0.11.0" \
  "trl>=0.9.0" \
  "datasets>=2.19.0" \
  "accelerate>=0.30.0" \
  "torch>=2.3.0" \
  "sacrebleu>=2.4.0" \
  "fastapi>=0.111.0" \
  "uvicorn[standard]>=0.29.0" \
  "pydantic>=2.0.0" \
  "pyyaml>=6.0"
```

This downloads roughly 3–4 GB (PyTorch is the bulk of it). Expect 5–10 minutes.

---

### Part 3 — Run the scripts individually (no GPU, no Docker)

All three commands should be run from the project root with the venv active.

**Step 1: dataset preparation**

```bash
python scripts/01_prepare_dataset.py \
  --input  data/dummy_corpus.jsonl \
  --output data/train_chat.jsonl
```

Expected output:
```
📂  Input: data/dummy_corpus.jsonl  (40 records after filtering)
📊  Lang pairs: {'it→de': 20, 'it→fr': 20}
💾  Writing to: data/train_chat.jsonl
✅  Done. 40 chat-format records written.
```

Then inspect what the training format actually looks like:
```bash
head -n 1 data/train_chat.jsonl | python3 -m json.tool
```

---

**Step 2: fine-tuning smoke-test (CPU, 5 steps)**

This will not produce a useful adapter, but it proves the entire training pipeline — data loading, LoRA wrapping, gradient steps, checkpointing — runs correctly end to end. It takes about 2–5 minutes on CPU.

```bash
python scripts/02_finetune_lora.py \
  --config configs/sft_lora.yaml \
  --cpu \
  --max_steps 5
```

Expected output (abridged):
```
⚠️   CPU mode: training will be very slow. Use --max_steps 5 for a smoke-test.
📥  Loading tokenizer from swiss-ai/Apertus-8B-Instruct-2509 ...
📥  Loading model ...        ← this step downloads ~16GB the first time
🔧  LoRA adapter applied.
    Trainable params: 41,943,040 (0.51% of total)
🏋️  Starting LoRA fine-tuning ...
{'loss': 2.43, 'step': 1} ...
{'loss': 2.11, 'step': 5}
💾  Saving LoRA adapter to output/apertus_lora/ ...
✅  Fine-tuning complete!
```

> **Note on model download:** The first run downloads Apertus 8B (~16 GB) from HuggingFace into `~/.cache/huggingface/`. Subsequent runs use the local cache. If you do not have 16 GB free on your WSL disk, see the note at the end about `HF_HOME`.

After it finishes, confirm the adapter files were created:
```bash
ls output/apertus_lora/
# adapter_config.json  adapter_model.safetensors  tokenizer.json  ...
```

---

**Step 3: evaluation — demo mode (instant, no model needed)**

This validates the metric pipeline (chrF + BLEU) without re-loading the model:

```bash
python scripts/03_evaluate.py \
  --eval_data data/eval_sentences.jsonl \
  --demo_mode \
  --output output/benchmark_demo.json
```

Expected output:
```
  ┌──────────────────────┬────────┬────────┐
  │ Model                │  chrF  │  BLEU  │
  ├──────────────────────┼────────┼────────┤
  │ Baseline (zero-shot) │  68.32 │  43.21 │  🟢 Excellent
  │ Fine-tuned (LoRA)    │  73.45 │  49.88 │  🟢 Excellent
  ├──────────────────────┼────────┼────────┤
  │ Δ (finetuned-base)   │ ▲ 5.13 │ ▲ 6.67 │
  └──────────────────────┴────────┴────────┘
💾  Full report saved to: output/benchmark_demo.json
```

If you completed Step 2, you can also run evaluation with the real (smoke-test) adapter — results won't be meaningful since it only trained 5 steps, but this confirms the load-and-merge path works:

```bash
python scripts/03_evaluate.py \
  --eval_data data/eval_sentences.jsonl \
  --adapter   output/apertus_lora \
  --output    output/benchmark_real.json
```

---

### Part 4 — Run the full stack with Docker (demo mode)

Make sure **Docker Desktop** is running on Windows with WSL integration enabled. You can verify this in Docker Desktop → Settings → Resources → WSL Integration → enable your Ubuntu distro.

Back in your WSL terminal:

```bash
# Confirm Docker is accessible from WSL
docker --version      # should print Docker version
docker compose version   # should print Compose version (v2 syntax)
```

Create the `.env` file (only needed for the real-model profiles; harmless for demo):

```bash
cp .env.example .env 2>/dev/null || echo "HF_TOKEN=" > .env
```

Start the demo stack:

```bash
docker compose --profile demo up --build
```

Docker will build the API image (~2–3 min first time) and start two containers: the FastAPI backend and an nginx serving the UI. Watch the logs — once you see `Uvicorn running on http://0.0.0.0:8080`, everything is up.

Open your browser on **Windows** (not inside WSL) and navigate to:

| URL | What you see |
|-----|-------------|
| `http://localhost` | DeepL-style translation UI |
| `http://localhost:8080/health` | `{"status":"ok","demo_mode":true}` |
| `http://localhost:8080/info` | Model info + supported languages |
| `http://localhost:8080/docs` | Auto-generated FastAPI Swagger UI |

Try a translation in the UI — it will return a mock response instantly since `DEMO_MODE=true`. To stop:

```bash
# Ctrl+C in the terminal running compose, then:
docker compose --profile demo down
```

---

### Troubleshooting quick-reference

| Symptom | Fix |
|---------|-----|
| `wsl: command not found` in PowerShell | Open Windows Terminal → click the Ubuntu tab directly |
| `python3: command not found` in WSL | `sudo apt update && sudo apt install python3 python3-venv python3-pip` |
| Model download fails / times out | Run `export HF_HUB_ENABLE_HF_TRANSFER=1` before the pip install, then `pip install hf_transfer` for faster downloads |
| WSL disk space too small for 16GB model | Set `export HF_HOME=/mnt/c/Users/YourName/.cache/huggingface` to store the cache on your Windows drive instead |
| `docker: command not found` in WSL | Open Docker Desktop → Settings → Resources → WSL Integration → toggle on your distro → Apply |
| Port 80 already in use | Change `"80:80"` to `"8090:80"` in `docker-compose.yml` and visit `http://localhost:8090` |
| `torch` import slow on CPU | Normal — PyTorch initialization takes 10–20 seconds the first time per session |