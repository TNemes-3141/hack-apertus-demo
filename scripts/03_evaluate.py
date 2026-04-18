#!/usr/bin/env python3
"""
03_evaluate.py
==============
Benchmarks translation quality using chrF and BLEU (sacrebleu).
Runs two models side-by-side:
  1. Baseline: zero-shot Apertus-8B-Instruct (no fine-tuning)
  2. Fine-tuned: Baseline + your LoRA adapter

Then produces a benchmark report PDF / markdown table.

Docs on metrics:
  chrF  — character n-gram F-score, robust for morphologically rich languages
  BLEU  — n-gram precision, standard MT metric since 2002

Usage:
    # Compare baseline vs fine-tuned adapter
    python scripts/03_evaluate.py \
        --eval_data  data/eval_sentences.jsonl \
        --adapter    output/apertus_lora \
        --output     output/benchmark_report.json

    # Baseline only (no adapter available yet)
    python scripts/03_evaluate.py \
        --eval_data  data/eval_sentences.jsonl \
        --baseline_only \
        --output     output/benchmark_baseline.json

    # CPU demo with mock translations (no GPU required)
    python scripts/03_evaluate.py \
        --eval_data  data/eval_sentences.jsonl \
        --demo_mode \
        --output     output/benchmark_demo.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def check_deps(demo_mode: bool):
    missing = []
    for pkg in ["sacrebleu"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not demo_mode:
        for pkg in ["transformers", "torch", "peft"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
    if missing:
        print("❌  Missing dependencies:", ", ".join(missing))
        print("    pip install sacrebleu transformers torch peft")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_model(model_path: str, adapter_path: Optional[str] = None):
    """Load Apertus model with optional LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"📥  Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device_map = "auto" if __import__("torch").cuda.is_available() else "cpu"
    dtype = __import__("torch").bfloat16 if __import__("torch").cuda.is_available() else __import__("torch").float32

    print(f"📥  Loading model {model_path} (device_map={device_map}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"🔌  Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()   # Bake into weights for faster inference
        print("    ✅ Adapter merged.")

    model.eval()
    return model, tokenizer


def translate(
    source: str,
    src_lang: str,
    tgt_lang: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
) -> str:
    """Run one translation with the Apertus instruct chat template."""
    import torch

    LANG_NAMES = {
        "de": "German", "fr": "French", "it": "Italian", "en": "English",
        "pt": "Portuguese", "es": "Spanish", "nl": "Dutch",
    }
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
            max_new_tokens=max_new_tokens,
            temperature=0.1,   # Low temp for translation (more deterministic)
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input tokens, decode only the new ones
    new_ids = output_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Demo mode: synthetic translations to validate the scoring pipeline
# without requiring a GPU or model download.
# ---------------------------------------------------------------------------
DEMO_TRANSLATIONS = {
    "eval-001": {
        "baseline": "Es wird mitgeteilt, dass der Termin für die Verlängerung des Aufenthaltserlaubnisses auf den 15. Mai festgelegt worden ist.",
        "finetuned": "Es wird mitgeteilt, dass der Termin für die Erneuerung der Aufenthaltsbewilligung auf den 15. Mai festgesetzt wurde.",
    },
    "eval-002": {
        "baseline": "Il est communiqué que le rendez-vous pour le renouvellement de la permission de séjour est prévu le 15 mai.",
        "finetuned": "Il est communiqué que le rendez-vous pour le renouvellement du permis de séjour est fixé au 15 mai.",
    },
    "eval-003": {
        "baseline": "Der Steuerzahler hat das Recht, die dokumentierten Berufskosten vom steuerbaren Einkommen abzuziehen.",
        "finetuned": "Der Steuerpflichtige hat das Recht, die belegten Berufskosten vom steuerbaren Einkommen abzuziehen.",
    },
    "eval-004": {
        "baseline": "Le contribuable a le droit de déduire du revenu imposable les frais professionnels justifiés.",
        "finetuned": "Le contribuable a le droit de déduire du revenu imposable les frais professionnels documentés.",
    },
    "eval-005": {
        "baseline": "Der Asylantrag muss persönlich beim Registrierungszentrum gestellt werden.",
        "finetuned": "Das Asylgesuch muss persönlich beim Registrationszentrum eingereicht werden.",
    },
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """
    Compute chrF and BLEU using sacrebleu.
    Returns dict with scores and a brief interpretation.
    """
    import sacrebleu

    # chrF — character n-gram F-score (recommended for morphologically rich languages)
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references])

    # BLEU — word n-gram precision with brevity penalty
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])

    return {
        "chrf": round(chrf_score.score, 2),
        "bleu": round(bleu_score.score, 2),
        "bleu_bp": round(bleu_score.bp, 4),         # Brevity penalty
        "n_sentences": len(hypotheses),
    }


def score_label(chrf: float) -> str:
    if chrf >= 70:  return "🟢 Excellent"
    if chrf >= 55:  return "🟡 Good"
    if chrf >= 40:  return "🟠 Fair"
    return "🔴 Needs improvement"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Apertus translation quality")
    parser.add_argument("--eval_data",      default="data/eval_sentences.jsonl")
    parser.add_argument("--model_path",     default="swiss-ai/Apertus-8B-Instruct-2509")
    parser.add_argument("--adapter",        default=None,
                        help="Path to LoRA adapter (output of 02_finetune_lora.py)")
    parser.add_argument("--baseline_only",  action="store_true",
                        help="Skip fine-tuned model, only eval baseline")
    parser.add_argument("--demo_mode",      action="store_true",
                        help="Use mock translations — no GPU required (for pipeline testing)")
    parser.add_argument("--output",         default="output/benchmark_report.json")
    args = parser.parse_args()

    check_deps(args.demo_mode)

    # ---- Load eval data -----------------------------------------------------
    eval_records = []
    with open(args.eval_data) as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))
    print(f"📋  Evaluating on {len(eval_records)} sentences from {args.eval_data}")

    # ---- Run translations ---------------------------------------------------
    if args.demo_mode:
        print("\n🎭  Demo mode: using pre-computed mock translations.")
        print("    (This validates the metric pipeline without requiring a GPU.)\n")
        baseline_hyps = []
        finetuned_hyps = []
        references     = []
        for rec in eval_records:
            demo = DEMO_TRANSLATIONS.get(rec["id"], {})
            baseline_hyps.append(demo.get("baseline", rec.get("reference", "")))
            finetuned_hyps.append(demo.get("finetuned", rec.get("reference", "")))
            references.append(rec["reference"])
    else:
        # Real model inference
        print(f"\n🔄  Loading baseline model ({args.model_path}) ...")
        baseline_model, tokenizer = load_model(args.model_path)

        if args.adapter and not args.baseline_only:
            print(f"\n🔄  Loading fine-tuned model (adapter: {args.adapter}) ...")
            finetuned_model, _ = load_model(args.model_path, adapter_path=args.adapter)
        else:
            finetuned_model = None

        print("\n🔄  Running inference ...")
        baseline_hyps  = []
        finetuned_hyps = []
        references     = []

        for i, rec in enumerate(eval_records):
            print(f"  [{i+1}/{len(eval_records)}] {rec['id']} ({rec['src_lang']}→{rec['tgt_lang']})")
            t0 = time.time()

            baseline_hyp = translate(
                rec["source"], rec["src_lang"], rec["tgt_lang"],
                baseline_model, tokenizer
            )
            baseline_hyps.append(baseline_hyp)

            if finetuned_model:
                ft_hyp = translate(
                    rec["source"], rec["src_lang"], rec["tgt_lang"],
                    finetuned_model, tokenizer
                )
                finetuned_hyps.append(ft_hyp)
            else:
                finetuned_hyps.append(baseline_hyp)   # Same if no adapter

            references.append(rec["reference"])
            print(f"     ✅ {time.time()-t0:.1f}s")

    # ---- Compute metrics ----------------------------------------------------
    print("\n📊  Computing chrF and BLEU scores ...")
    baseline_metrics  = compute_metrics(baseline_hyps, references)
    finetuned_metrics = compute_metrics(finetuned_hyps, references) if not args.baseline_only else None

    # ---- Report -------------------------------------------------------------
    print("\n" + "="*60)
    print("   BENCHMARK REPORT — Apertus 8B Translation Quality")
    print("="*60)
    print(f"\n  Model:      {args.model_path}")
    print(f"  Eval set:   {args.eval_data} ({len(eval_records)} sentences)")
    print(f"  Mode:       {'Demo (mock)' if args.demo_mode else 'Real inference'}")
    print()
    print("  ┌──────────────────────┬────────┬────────┐")
    print("  │ Model                │  chrF  │  BLEU  │")
    print("  ├──────────────────────┼────────┼────────┤")
    print(f"  │ Baseline (zero-shot) │ {baseline_metrics['chrf']:6.2f} │ {baseline_metrics['bleu']:6.2f} │  {score_label(baseline_metrics['chrf'])}")
    if finetuned_metrics:
        delta_chrf = finetuned_metrics['chrf'] - baseline_metrics['chrf']
        delta_bleu = finetuned_metrics['bleu'] - baseline_metrics['bleu']
        print(f"  │ Fine-tuned (LoRA)    │ {finetuned_metrics['chrf']:6.2f} │ {finetuned_metrics['bleu']:6.2f} │  {score_label(finetuned_metrics['chrf'])}")
        print("  ├──────────────────────┼────────┼────────┤")
        arrow = "▲" if delta_chrf >= 0 else "▼"
        print(f"  │ Δ (finetuned-base)   │ {arrow}{abs(delta_chrf):5.2f} │ {arrow}{abs(delta_bleu):5.2f} │")
    print("  └──────────────────────┴────────┴────────┘")

    print("\n  Per-sentence examples:")
    for i, (rec, b_hyp, ft_hyp) in enumerate(zip(eval_records, baseline_hyps, finetuned_hyps)):
        print(f"\n  [{i+1}] {rec['id']} ({rec['src_lang']}→{rec['tgt_lang']})")
        print(f"     Source:     {rec['source'][:80]}...")
        print(f"     Reference:  {rec['reference'][:80]}...")
        print(f"     Baseline:   {b_hyp[:80]}...")
        if finetuned_metrics:
            print(f"     Fine-tuned: {ft_hyp[:80]}...")

    print("\n" + "="*60)
    print("\n  Metric notes:")
    print("  • chrF (char F-score): Range 0–100. Better than BLEU for German,")
    print("    French, Italian due to morphological complexity.")
    print("  • BLEU: Classic MT metric. Range 0–100. Tends to underestimate")
    print("    quality for low-resource or domain-specific models.")
    print("  • For human evaluation, use the rubric in the hackathon judging")
    print("    criteria (fluency + accuracy + cultural fit).")
    print("="*60)

    # ---- Save JSON report ---------------------------------------------------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path":         args.model_path,
        "adapter_path":       args.adapter,
        "eval_data":          args.eval_data,
        "demo_mode":          args.demo_mode,
        "n_sentences":        len(eval_records),
        "baseline_metrics":   baseline_metrics,
        "finetuned_metrics":  finetuned_metrics,
        "per_sentence": [
            {
                "id":          rec["id"],
                "src_lang":    rec["src_lang"],
                "tgt_lang":    rec["tgt_lang"],
                "source":      rec["source"],
                "reference":   rec["reference"],
                "baseline":    b,
                "finetuned":   ft,
            }
            for rec, b, ft in zip(eval_records, baseline_hyps, finetuned_hyps)
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Full report saved to: {args.output}")


if __name__ == "__main__":
    main()