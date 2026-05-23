#!/usr/bin/env python3
"""
Before/after comparison of model morph vs macron_morph for macronized Latin.

Usage:
    uv run python scripts/demo_before_after.py
    uv run python scripts/demo_before_after.py "Caesar in Galliam vēnit."
"""

import sys
from pathlib import Path

import spacy

import latincy_ext  # noqa: F401 — registers factory

LOOKUP = Path(__file__).resolve().parents[1].parent / \
    "latincy-words/data/processed/latin-forms-macronized-morph.json"

SENTENCES = [
    "A puellā magnā rēs facta est.",
    "Caesar in Galliam vēnit.",
    "Puella ad aquam veniēbat.",
    "Optō ut puella venīre velit.",
]


def _parse_feats(morph_str: str) -> dict[str, str]:
    if not morph_str:
        return {}
    return dict(kv.split("=", 1) for kv in morph_str.split("|") if "=" in kv)


def _diff_marker(model_feats: dict, macron_feats: dict) -> str:
    """Return a string describing feature-level differences."""
    changes = []
    for k, v in macron_feats.items():
        model_v = model_feats.get(k)
        if model_v is None:
            changes.append(f"+{k}={v}")
        elif model_v != v:
            changes.append(f"{k}: {model_v}→{v}")
    return "  ".join(changes)


def run(sentence: str, nlp_base: spacy.Language, nlp_ext: spacy.Language) -> None:
    doc_base = nlp_base(sentence)
    doc_ext  = nlp_ext(sentence)

    col_w = [14, 38, 38, 30]
    header = f"{'token':<{col_w[0]}}  {'model morph':<{col_w[1]}}  {'macron_morph':<{col_w[2]}}  {'diff'}"
    sep    = "  ".join("-" * w for w in col_w)

    print(f"\n  {sentence}")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    for tok_b, tok_e in zip(doc_base, doc_ext):
        orig   = getattr(tok_b._, "orig_text", None) or tok_b.text
        model  = str(tok_b.morph)
        macron = tok_e._.macron_morph

        if tok_b.is_punct or tok_b.is_space:
            continue
        # Only show tokens where macron_morph has something to say
        if not macron:
            diff = ""
            macron_col = "(no signal)"
        else:
            diff      = _diff_marker(_parse_feats(model), _parse_feats(macron))
            macron_col = macron

        marker = "**" if diff else "  "
        print(f"{marker} {orig:<{col_w[0]}}  {model:<{col_w[1]}}  {macron_col:<{col_w[2]}}  {diff}")

    print(f"  {sep}")
    print("  ** = macron_morph provides new or conflicting information")


def main() -> None:
    sentences = sys.argv[1:] if len(sys.argv) > 1 else SENTENCES

    nlp_base = spacy.load("la_core_web_lg")

    nlp_ext = spacy.load("la_core_web_lg")
    nlp_ext.add_pipe("macron_morph", last=True,
                     config={"lookup_path": str(LOOKUP)})

    for sent in sentences:
        run(sent, nlp_base, nlp_ext)


if __name__ == "__main__":
    main()
