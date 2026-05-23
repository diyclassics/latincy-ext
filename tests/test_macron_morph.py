"""Tests for MacronMorphComponent."""

import json
import tempfile
from pathlib import Path

import pytest
import spacy
from spacy.tokens import Token

import latincy_ext  # noqa: F401 — registers factory


LOOKUP = {
    "puellā":  [{"lemma": "puella", "pos": "NOUN", "morph": "Case=Abl|Number=Sing"}],
    "vēnit":   [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"},
        {"lemma": "ueneo", "pos": "VERB", "morph": "Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act"},
    ],
    "thēsaurō": [
        {"lemma": "thesaurus", "pos": "NOUN", "morph": "Case=Dat|Number=Sing"},
        {"lemma": "thesaurus", "pos": "NOUN", "morph": "Case=Abl|Number=Sing"},
    ],
    "puellīs": [
        {"lemma": "puella", "pos": "NOUN", "morph": "Case=Abl|Number=Plur"},
        {"lemma": "puella", "pos": "NOUN", "morph": "Case=Dat|Number=Plur"},
    ],
}


@pytest.fixture
def lookup_path(tmp_path):
    p = tmp_path / "lookup.json"
    p.write_text(json.dumps(LOOKUP), encoding="utf-8")
    return str(p)


@pytest.fixture
def nlp(lookup_path):
    _nlp = spacy.blank("la")
    _nlp.add_pipe("macron_morph", config={"lookup_path": lookup_path})
    return _nlp


def _doc_with_orig(nlp, text, orig_text):
    """Create a single-token doc with orig_text set manually."""
    if not Token.has_extension("orig_text"):
        Token.set_extension("orig_text", default=None)
    doc = nlp.make_doc(text)
    doc[0]._.orig_text = orig_text
    # Run only macron_morph
    for _, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc


class TestUnambiguous:
    def test_puella_ablative(self, nlp):
        doc = _doc_with_orig(nlp, "puella", "puellā")
        assert doc[0]._.macron_morph == "Case=Abl|Number=Sing"
        assert doc[0]._.macron_pos_ == "NOUN"

    def test_no_macron_no_signal(self, nlp):
        doc = _doc_with_orig(nlp, "puella", "puella")
        assert doc[0]._.macron_morph == ""
        assert doc[0]._.macron_pos_ == ""

    def test_unknown_form(self, nlp):
        doc = _doc_with_orig(nlp, "aqua", "aquā")
        assert doc[0]._.macron_morph == ""


class TestAmbiguous:
    def test_agreed_features_extracted(self, nlp):
        # thēsaurō is dat or abl — Number=Sing and pos=NOUN agreed, Case not
        doc = _doc_with_orig(nlp, "thesauro", "thēsaurō")
        assert "Number=Sing" in doc[0]._.macron_morph
        assert "Case=" not in doc[0]._.macron_morph
        assert doc[0]._.macron_pos_ == "NOUN"

    def test_plural_number_agreed(self, nlp):
        # puellīs is abl or dat plural — Number=Plur agreed
        doc = _doc_with_orig(nlp, "puellis", "puellīs")
        assert "Number=Plur" in doc[0]._.macron_morph
        assert "Case=" not in doc[0]._.macron_morph

    def test_no_pos_agreement_across_lemmas(self, nlp):
        # vēnit: two different verbs, both VERB — pos agrees
        doc = _doc_with_orig(nlp, "venit", "vēnit")
        assert doc[0]._.macron_pos_ == "VERB"
        # Tense differs (Past vs Pres) so should not be in agreed morph
        assert "Tense=" not in doc[0]._.macron_morph
        # Number=Sing, Person=3, VerbForm=Fin, Voice=Act all agree
        assert "Number=Sing" in doc[0]._.macron_morph
        assert "Person=3" in doc[0]._.macron_morph


class TestPunctSpace:
    def test_punct_skipped(self, nlp):
        if not Token.has_extension("orig_text"):
            Token.set_extension("orig_text", default=None)
        doc = nlp.make_doc(".")
        doc[0]._.orig_text = "."
        for _, pipe in nlp.pipeline:
            doc = pipe(doc)
        assert doc[0]._.macron_morph == ""
