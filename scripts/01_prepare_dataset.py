#!/usr/bin/env python3
"""
01_prepare_dataset.py
=====================
Converts the raw parallel corpus (JSONL with src/tgt fields) into the
Apertus instruct chat-template format required by SFTTrainer.

Input:  data/dummy_corpus.jsonl
Output: data/train_chat.jsonl

Usage:
    python scripts/01_prepare_dataset.py \
        --input  data/dummy_corpus.jsonl \
        --output data/train_chat.jsonl \
        [--src_lang it] [--tgt_lang de]

For real hackathon use, replace the input file with a cleaned OPUS/CC-Aligned
parallel corpus (same JSONL schema: id, src_lang, tgt_lang, source, target).
"""

import json
import argparse
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Language display names (add more as needed for all 20 tracks)
# ---------------------------------------------------------------------------
LANG_NAMES = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "en": "English",
    "pt": "Portuguese",
    "es": "Spanish",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "ko": "Korean",
    "ja": "Japanese",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "sv": "Swedish",
    "hu": "Hungarian",
    "cs": "Czech",
    "el": "Greek",
    "fa": "Persian",
    "sw": "Swahili",
    "id": "Indonesian",
}


def build_translation_prompt(source: str, src_lang: str, tgt_lang: str) -> str:
    """
    Builds the instruction prompt following the Apertus instruct format.
    The phrasing 'administrative/official context' nudges the model toward
    formal register, which matters for public sector use.
    """
    src_name = LANG_NAMES.get(src_lang, src_lang.upper())
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())
    return (
        f"Translate the following {src_name} text into {tgt_name}. "
        f"This is an official administrative document. "
        f"Preserve formal register and terminology.\n\n"
        f"{source}"
    )


def convert_record(record: dict) -> dict:
    """
    Converts one parallel corpus record into the chat-template messages format
    expected by TRL's SFTTrainer with apply_chat_template=True.
    """
    prompt = build_translation_prompt(
        record["source"], record["src_lang"], record["tgt_lang"]
    )
    return {
        "id": record.get("id", ""),
        "messages": [
            {"role": "user",    "content": prompt},
            {"role": "assistant", "content": record["target"]},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare parallel corpus for Apertus SFT")
    parser.add_argument("--input",    default="data/dummy_corpus.jsonl",
                        help="Input parallel corpus JSONL")
    parser.add_argument("--output",   default="data/train_chat.jsonl",
                        help="Output chat-format JSONL")
    parser.add_argument("--src_lang", default=None,
                        help="Filter to this source language (e.g. 'it')")
    parser.add_argument("--tgt_lang", default=None,
                        help="Filter to this target language (e.g. 'de')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (useful for quick smoke-tests)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Optional language filtering
            if args.src_lang and rec.get("src_lang") != args.src_lang:
                continue
            if args.tgt_lang and rec.get("tgt_lang") != args.tgt_lang:
                continue
            records.append(rec)

    if args.max_samples:
        records = records[:args.max_samples]

    # Stats
    lang_pairs = Counter(
        f"{r['src_lang']}→{r['tgt_lang']}" for r in records
    )

    print(f"📂  Input:   {input_path}  ({len(records)} records after filtering)")
    print(f"📊  Lang pairs: {dict(lang_pairs)}")
    print(f"💾  Writing to: {output_path}")

    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in records:
            chat_rec = convert_record(rec)
            out.write(json.dumps(chat_rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅  Done. {written} chat-format records written.")
    print()
    print("Sample output:")
    print(json.dumps(convert_record(records[0]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()