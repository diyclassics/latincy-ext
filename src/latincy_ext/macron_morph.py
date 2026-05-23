"""macron_morph — spaCy pipeline component for macron-based morphological disambiguation.

Reads a macronized token form (from ``token._.macronized`` if the macronizer ran,
otherwise falls back to ``token._.orig_text`` for pre-macronized input) and looks
it up in a kaikki-derived table of macronized form → UD morph parses.

Sets two custom extensions:

- ``token._.macron_morph``:  UD feature string (e.g. ``"Case=Abl|Number=Sing"``)
  with only the features all parses agree on. Empty string when no signal.
- ``token._.macron_pos_``:   UD POS string when all parses agree, else empty string.

These are alternatives to the model's predictions, not overrides. Consumer code
pattern::

    morph = token._.macron_morph or str(token.morph)

Usage::

    import spacy
    import latincy_ext  # registers the factory

    nlp = spacy.load("la_core_web_lg")
    nlp.add_pipe("macron_morph", config={"lookup_path": "/path/to/latin-forms-macronized-morph.json"})

    doc = nlp("puellā")
    print(doc[0]._.macron_morph)   # "Case=Abl|Number=Sing"

Optionally chain after the macronizer for plain-text input::

    nlp.add_pipe("macronizer", ...)
    nlp.add_pipe("macron_morph", last=True, ...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc, Token

MACRONS = frozenset("āēīōūȳĀĒĪŌŪȲ")


@Language.factory(
    "macron_morph",
    default_config={"lookup_path": None},
    assigns=["token._.macron_morph", "token._.macron_pos_"],
)
def create_macron_morph(
    nlp: Language,
    name: str,
    lookup_path: Optional[str],
) -> "MacronMorphComponent":
    return MacronMorphComponent(nlp, name, lookup_path=lookup_path)


class MacronMorphComponent:
    """Macron-based morphological disambiguation component.

    For each token, resolves the macronized surface form and looks it up in
    the prebuilt kaikki table. Sets ``token._.macron_morph`` and
    ``token._.macron_pos_`` with agreed-upon features across all matching parses.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        lookup_path: Optional[str | Path] = None,
    ) -> None:
        self.name = name
        self._lookup: dict[str, list[dict]] = {}
        self._lookup_path = lookup_path
        self._loaded = False

        if not Token.has_extension("macron_morph"):
            Token.set_extension("macron_morph", default="")
        if not Token.has_extension("macron_pos_"):
            Token.set_extension("macron_pos_", default="")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._lookup_path:
            with open(self._lookup_path, encoding="utf-8") as f:
                self._lookup = json.load(f)
        self._loaded = True

    def _get_macronized(self, token: Token) -> str | None:
        """Return the macronized form for this token, or None if no macrons present."""
        # Prefer macronizer output if available
        macronized = getattr(token._, "macronized", None)
        if macronized and any(c in MACRONS for c in macronized):
            return macronized
        # Fall back to orig_text for pre-macronized input
        orig = getattr(token._, "orig_text", None)
        if orig and any(c in MACRONS for c in orig):
            return orig
        return None

    def _resolve(self, macronized: str) -> tuple[str, str]:
        """Return (agreed_morph, agreed_pos) for a macronized form.

        For single-parse forms: returns full morph and pos.
        For multi-parse forms: returns only features all parses agree on.
        For unknown forms: returns ("", "").
        """
        parses = self._lookup.get(macronized)
        if not parses:
            return "", ""

        if len(parses) == 1:
            return parses[0]["morph"], parses[0]["pos"]

        # Multiple parses — intersect features
        pos_values = {p["pos"] for p in parses}
        agreed_pos = pos_values.pop() if len(pos_values) == 1 else ""

        # Parse each morph string into feature dicts
        feat_dicts = [_parse_morph(p["morph"]) for p in parses]

        # Keep only features that are present in ALL parses with the same value
        agreed: dict[str, str] = {}
        all_keys = set().union(*feat_dicts)
        for key in all_keys:
            values = {fd.get(key) for fd in feat_dicts}
            if len(values) == 1 and None not in values:
                agreed[key] = values.pop()

        if not agreed:
            return "", agreed_pos

        morph_str = "|".join(f"{k}={v}" for k, v in sorted(agreed.items()))
        return morph_str, agreed_pos

    def __call__(self, doc: Doc) -> Doc:
        self._ensure_loaded()

        for token in doc:
            if token.is_punct or token.is_space:
                continue
            macronized = self._get_macronized(token)
            if not macronized:
                continue
            morph, pos = self._resolve(macronized)
            token._.macron_morph = morph
            token._.macron_pos_ = pos

        return doc

    def to_disk(self, path: str, *, exclude: tuple = ()) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cfg: dict = {}
        if self._lookup_path:
            cfg["lookup_path"] = str(self._lookup_path)
        if self._lookup and not self._lookup_path:
            # Lookup was loaded from bytes — serialize alongside
            with open(path / "lookup.json", "w", encoding="utf-8") as f:
                json.dump(self._lookup, f, ensure_ascii=False)
            cfg["lookup_path"] = str(path / "lookup.json")
        with open(path / "config.json", "w") as f:
            json.dump(cfg, f)

    def from_disk(self, path: str, *, exclude: tuple = ()) -> "MacronMorphComponent":
        path = Path(path)
        cfg_file = path / "config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = json.load(f)
            if cfg.get("lookup_path"):
                self._lookup_path = cfg["lookup_path"]
                self._loaded = False
        return self

    def to_bytes(self, *, exclude: tuple = ()) -> bytes:
        self._ensure_loaded()
        data: dict = {}
        if self._lookup:
            data["lookup"] = self._lookup
        return json.dumps(data, ensure_ascii=False).encode("utf-8") if data else b""

    def from_bytes(self, data: bytes, *, exclude: tuple = ()) -> "MacronMorphComponent":
        if data:
            d = json.loads(data.decode("utf-8"))
            if "lookup" in d:
                self._lookup = d["lookup"]
                self._loaded = True
        return self


def _parse_morph(morph_str: str) -> dict[str, str]:
    """Parse 'A=x|B=y' into {'A': 'x', 'B': 'y'}."""
    if not morph_str:
        return {}
    return dict(kv.split("=", 1) for kv in morph_str.split("|") if "=" in kv)
