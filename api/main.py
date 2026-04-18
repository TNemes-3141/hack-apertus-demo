#!/usr/bin/env python3
"""
api/main.py
===========
FastAPI translation server wrapping Apertus 8B (+ optional LoRA adapter).

Endpoints:
  GET  /health          — liveness probe
  GET  /info            — model info, supported languages
  POST /translate       — translate a text string
  POST /translate/batch — translate multiple strings

Deployment modes (set via environment variables):
  MODEL_PATH    = swiss-ai/Apertus-8B-Instruct-2509  (default)
  ADAPTER_PATH  = /app/adapter  (optional; mount your LoRA adapter here)
  MODEL_DEVICE  = cuda | cpu    (default: auto-detect)
  USE_4BIT      = true | false  (default: false)
  DEMO_MODE     = true | false  (default: false — returns mock translations)

For the vLLM-based production deployment, see docker-compose.yml.
This file targets the simpler transformers-based setup for development.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apertus-translate")

# ---------------------------------------------------------------------------
# Supported language pairs — extend for all 20 hackathon tracks
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "de": "German (Deutsch)",
    "fr": "French (Français)",
    "it": "Italian (Italiano)",
    "en": "English",
    "pt": "Portuguese (Português)",
    "es": "Spanish (Español)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "tr": "Turkish (Türkçe)",
    "ar": "Arabic (العربية)",
    "ko": "Korean (한국어)",
    "ja": "Japanese (日本語)",
    "uk": "Ukrainian (Українська)",
    "sv": "Swedish (Svenska)",
    "hu": "Hungarian (Magyar)",
    "cs": "Czech (Čeština)",
    "el": "Greek (Ελληνικά)",
    "fa": "Persian (فارسی)",
    "sw": "Swahili",
    "id": "Indonesian (Bahasa Indonesia)",
}

# Swiss/EU official target languages (most common for hackathon tracks)
PRIMARY_TARGET_LANGS = ["de", "fr", "it"]

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
model_state = {
    "model": None,
    "tokenizer": None,
    "demo_mode": False,
    "model_path": "",
    "adapter_path": "",
    "load_time": 0.0,
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_global():
    demo_mode   = os.getenv("DEMO_MODE", "false").lower() == "true"
    model_path  = os.getenv("MODEL_PATH", "swiss-ai/Apertus-8B-Instruct-2509")
    adapter_path = os.getenv("ADAPTER_PATH", "")
    use_4bit    = os.getenv("USE_4BIT", "false").lower() == "true"

    model_state["demo_mode"]   = demo_mode
    model_state["model_path"]  = model_path
    model_state["adapter_path"] = adapter_path

    if demo_mode:
        logger.info("🎭  Demo mode enabled — model NOT loaded, returning mock translations.")
        return

    logger.info(f"📥  Loading model: {model_path}")
    t0 = time.time()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        device_env = os.getenv("MODEL_DEVICE", "auto")
        if device_env == "auto":
            device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            device_map = device_env

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

        if adapter_path and os.path.isdir(adapter_path):
            logger.info(f"🔌  Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            logger.info("✅  LoRA adapter merged.")

        model.eval()
        model_state["model"]     = model
        model_state["tokenizer"] = tokenizer
        model_state["load_time"] = time.time() - t0
        logger.info(f"✅  Model ready in {model_state['load_time']:.1f}s")

    except ImportError as e:
        logger.error(f"❌  Import error: {e}")
        logger.warning("⚠️   Falling back to demo mode.")
        model_state["demo_mode"] = True


# ---------------------------------------------------------------------------
# Translation function
# ---------------------------------------------------------------------------
LANG_NAMES = {k: v.split(" ")[0] for k, v in SUPPORTED_LANGUAGES.items()}

MOCK_TRANSLATIONS = {
    ("it", "de"): "Dies ist eine automatisch generierte Übersetzung im Demo-Modus.",
    ("it", "fr"): "Ceci est une traduction générée automatiquement en mode démo.",
    ("de", "fr"): "Ceci est une traduction générée automatiquement en mode démo.",
    ("de", "it"): "Questa è una traduzione generata automaticamente in modalità demo.",
    ("fr", "de"): "Dies ist eine automatisch generierte Übersetzung im Demo-Modus.",
    ("fr", "it"): "Questa è una traduzione generata automaticamente in modalità demo.",
}


def do_translate(source: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """
    Translate source text. Returns (translation, elapsed_seconds).
    """
    t0 = time.time()

    if model_state["demo_mode"] or model_state["model"] is None:
        # Demo mode: return a mock with the original text echoed back
        mock_key = (src_lang, tgt_lang)
        mock = MOCK_TRANSLATIONS.get(mock_key,
            f"[DEMO] Translation from {src_lang} to {tgt_lang}: {source[:80]}...")
        time.sleep(0.3)  # Simulate network latency
        return mock, time.time() - t0

    import torch
    model     = model_state["model"]
    tokenizer = model_state["tokenizer"]

    src_name = LANG_NAMES.get(src_lang, src_lang.upper())
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())

    prompt = (
        f"Translate the following {src_name} text into {tgt_name}. "
        f"This is an official administrative document. "
        f"Preserve formal register and terminology.\n\n"
        f"{source}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][len(inputs.input_ids[0]):]
    translation = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return translation, time.time() - t0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_global()
    yield


app = FastAPI(
    title="Hack Apertus — Translation API",
    description=(
        "Open-source multilingual translation for Swiss public administrations. "
        "Powered by Apertus 8B (swiss-ai/Apertus-8B-Instruct-2509) with optional "
        "LoRA fine-tuned adapter."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI (mounted at /ui — nginx would normally do this)
ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text:     str = Field(..., min_length=1, max_length=5000,
                          description="Source text to translate")
    src_lang: str = Field(..., min_length=2, max_length=10,
                          description="Source language code (e.g. 'it')")
    tgt_lang: str = Field(..., min_length=2, max_length=10,
                          description="Target language code (e.g. 'de')")


class TranslationResponse(BaseModel):
    translation:  str
    src_lang:     str
    tgt_lang:     str
    elapsed_ms:   float
    model:        str
    adapter:      Optional[str]
    demo_mode:    bool


class BatchTranslationRequest(BaseModel):
    texts:    list[str] = Field(..., max_length=20)
    src_lang: str
    tgt_lang: str


class BatchTranslationResponse(BaseModel):
    translations: list[str]
    src_lang:     str
    tgt_lang:     str
    total_ms:     float
    demo_mode:    bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_state["model"] is not None or model_state["demo_mode"],
        "demo_mode": model_state["demo_mode"],
    }


@app.get("/info")
def info():
    return {
        "model":              model_state["model_path"],
        "adapter":            model_state["adapter_path"] or None,
        "demo_mode":          model_state["demo_mode"],
        "load_time_s":        round(model_state["load_time"], 2),
        "supported_languages": SUPPORTED_LANGUAGES,
        "primary_targets":    PRIMARY_TARGET_LANGS,
    }


@app.post("/translate", response_model=TranslationResponse)
def translate_endpoint(req: TranslationRequest):
    if req.src_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported source language: {req.src_lang}")
    if req.tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported target language: {req.tgt_lang}")
    if req.src_lang == req.tgt_lang:
        raise HTTPException(400, "Source and target languages must differ")

    translation, elapsed = do_translate(req.text, req.src_lang, req.tgt_lang)

    return TranslationResponse(
        translation=translation,
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
        elapsed_ms=round(elapsed * 1000, 1),
        model=model_state["model_path"],
        adapter=model_state["adapter_path"] or None,
        demo_mode=model_state["demo_mode"],
    )


@app.post("/translate/batch", response_model=BatchTranslationResponse)
def batch_translate(req: BatchTranslationRequest):
    if req.src_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported source language: {req.src_lang}")
    if req.tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported target language: {req.tgt_lang}")

    t0 = time.time()
    translations = []
    for text in req.texts:
        translation, _ = do_translate(text, req.src_lang, req.tgt_lang)
        translations.append(translation)

    return BatchTranslationResponse(
        translations=translations,
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
        total_ms=round((time.time() - t0) * 1000, 1),
        demo_mode=model_state["demo_mode"],
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)