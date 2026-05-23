"""Sentence-level tests for MacronMorphComponent.

Covers ~20 puella/venire forms across cases, numbers, tenses, plus
non-macronized input variants that must return empty strings.
"""

import json

import pytest
import spacy
from spacy.tokens import Token

import latincy_ext  # noqa: F401 — registers factory

LOOKUP = {
    # puella forms
    "puellā":   [{"lemma": "puella", "pos": "NOUN", "morph": "Case=Abl|Number=Sing"}],
    "puellārum":[{"lemma": "puella", "pos": "NOUN", "morph": "Case=Gen|Number=Plur"}],
    "puellās":  [{"lemma": "puella", "pos": "NOUN", "morph": "Case=Acc|Number=Plur"}],
    "puellīs":  [
        {"lemma": "puella",  "pos": "NOUN", "morph": "Case=Dat|Number=Plur"},
        {"lemma": "puella",  "pos": "NOUN", "morph": "Case=Abl|Number=Plur"},
        {"lemma": "puellus", "pos": "NOUN", "morph": "Case=Dat|Number=Plur"},
        {"lemma": "puellus", "pos": "NOUN", "morph": "Case=Abl|Number=Plur"},
    ],
    # venire / uenio forms
    "vēnit": [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"},
        {"lemma": "ueneo", "pos": "VERB", "morph": "Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act"},
    ],
    "vēnerat": [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Pqp|VerbForm=Fin|Voice=Act"},
    ],
    "veniēbat": [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"},
    ],
    "venīret": [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Imp|Mood=Sub|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"},
    ],
    "vēnerit": [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"},
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Perf|Mood=Sub|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"},
    ],
    "venīre": [
        {"lemma": "uenio", "pos": "VERB", "morph": "Aspect=Imp|Tense=Pres|VerbForm=Inf|Voice=Act"},
    ],
    "veniēns": [
        {"lemma": "uenio",   "pos": "VERB", "morph": "Aspect=Imp|Tense=Pres|VerbForm=Part|Voice=Act"},
        {"lemma": "ueniens", "pos": "VERB", "morph": "Case=Nom|Gender=Neut|Number=Sing"},
        {"lemma": "ueniens", "pos": "VERB", "morph": "Case=Acc|Gender=Neut|Number=Sing"},
        {"lemma": "ueniens", "pos": "VERB", "morph": "Case=Voc|Gender=Neut|Number=Sing"},
    ],
}


@pytest.fixture(scope="module")
def lookup_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("lookup") / "lookup.json"
    p.write_text(json.dumps(LOOKUP), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def nlp(lookup_path):
    if not Token.has_extension("orig_text"):
        Token.set_extension("orig_text", default=None)
    _nlp = spacy.blank("la")
    _nlp.add_pipe("macron_morph", config={"lookup_path": lookup_path})
    return _nlp


def _orig(nlp, text, orig_text):
    """Single-token doc with orig_text set."""
    doc = nlp.make_doc(text)
    doc[0]._.orig_text = orig_text
    for _, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc[0]


# ---------------------------------------------------------------------------
# Unambiguous macronized forms — full morph expected
# ---------------------------------------------------------------------------

class TestUnambiguousMacronized:
    def test_puella_ablative_singular(self, nlp):
        tok = _orig(nlp, "puella", "puellā")
        assert tok._.macron_morph == "Case=Abl|Number=Sing"
        assert tok._.macron_pos_ == "NOUN"

    def test_puella_genitive_plural(self, nlp):
        tok = _orig(nlp, "puellarum", "puellārum")
        assert tok._.macron_morph == "Case=Gen|Number=Plur"
        assert tok._.macron_pos_ == "NOUN"

    def test_puella_accusative_plural(self, nlp):
        tok = _orig(nlp, "puellas", "puellās")
        assert tok._.macron_morph == "Case=Acc|Number=Plur"
        assert tok._.macron_pos_ == "NOUN"

    def test_venire_pluperfect_indicative(self, nlp):
        tok = _orig(nlp, "uenerat", "vēnerat")
        assert tok._.macron_morph == "Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Pqp|VerbForm=Fin|Voice=Act"
        assert tok._.macron_pos_ == "VERB"

    def test_venire_imperfect_indicative(self, nlp):
        tok = _orig(nlp, "ueniebat", "veniēbat")
        assert tok._.macron_morph == "Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"
        assert tok._.macron_pos_ == "VERB"

    def test_venire_imperfect_subjunctive(self, nlp):
        tok = _orig(nlp, "ueniret", "venīret")
        assert tok._.macron_morph == "Aspect=Imp|Mood=Sub|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"
        assert tok._.macron_pos_ == "VERB"

    def test_venire_present_infinitive(self, nlp):
        tok = _orig(nlp, "uenire", "venīre")
        assert tok._.macron_morph == "Aspect=Imp|Tense=Pres|VerbForm=Inf|Voice=Act"
        assert tok._.macron_pos_ == "VERB"


# ---------------------------------------------------------------------------
# Ambiguous macronized forms — agreed features only
# ---------------------------------------------------------------------------

class TestAmbiguousMacronized:
    def test_puellis_number_agreed_case_not(self, nlp):
        # dat or abl plural — all 4 parses agree on Number=Plur and NOUN
        tok = _orig(nlp, "puellis", "puellīs")
        assert "Number=Plur" in tok._.macron_morph
        assert "Case=" not in tok._.macron_morph
        assert tok._.macron_pos_ == "NOUN"

    def test_venit_macronized_tense_not_agreed(self, nlp):
        # vēnit: uenio perfect (Tense=Past) vs ueneo present (Tense=Pres)
        # Mood, Number, Person, VerbForm, Voice all agree
        tok = _orig(nlp, "uenit", "vēnit")
        assert "Tense=" not in tok._.macron_morph
        assert "Aspect=" not in tok._.macron_morph
        assert "Mood=Ind" in tok._.macron_morph
        assert "Number=Sing" in tok._.macron_morph
        assert "Person=3" in tok._.macron_morph
        assert "VerbForm=Fin" in tok._.macron_morph
        assert "Voice=Act" in tok._.macron_morph
        assert tok._.macron_pos_ == "VERB"

    def test_venerit_mood_not_agreed(self, nlp):
        # vēnerit: ind vs subj — same lemma, Mood differs
        tok = _orig(nlp, "uenerit", "vēnerit")
        assert "Mood=" not in tok._.macron_morph
        assert "Aspect=Perf" in tok._.macron_morph
        assert "Tense=Past" in tok._.macron_morph
        assert "Number=Sing" in tok._.macron_morph
        assert "Person=3" in tok._.macron_morph
        assert "VerbForm=Fin" in tok._.macron_morph
        assert tok._.macron_pos_ == "VERB"

    def test_veniens_no_morph_agreement_pos_agreed(self, nlp):
        # veniēns: participle vs multiple noun forms — nothing agrees across all 4
        tok = _orig(nlp, "ueniens", "veniēns")
        assert tok._.macron_morph == ""
        assert tok._.macron_pos_ == "VERB"


# ---------------------------------------------------------------------------
# Non-macronized input — must produce empty strings (no false positives)
# ---------------------------------------------------------------------------

class TestNonMacronizedNoSignal:
    @pytest.mark.parametrize("text,orig", [
        # puella — nominative sg; no macron → no signal
        ("puella",    "puella"),
        # puellae — gen or dat sg; ambiguous even with macron data absent
        ("puellae",   "puellae"),
        # puellis — dat/abl pl; orig has no macron
        ("puellis",   "puellis"),
        # puellarum — gen pl; no macron
        ("puellarum", "puellarum"),
        # venit — present tense 3sg; no macron (contrast: vēnit = perfect)
        ("uenit",     "venit"),
        # veniet — future; not in lookup at all
        ("ueniet",    "veniet"),
        # venire — infinitive without macron
        ("uenire",    "venire"),
    ])
    def test_no_macrons_returns_empty(self, nlp, text, orig):
        tok = _orig(nlp, text, orig)
        assert tok._.macron_morph == "", f"Expected empty for {orig!r}, got {tok._.macron_morph!r}"
        assert tok._.macron_pos_ == "", f"Expected empty pos for {orig!r}, got {tok._.macron_pos_!r}"

    def test_no_orig_text_set(self, nlp):
        doc = nlp.make_doc("puella")
        doc[0]._.orig_text = None
        for _, pipe in nlp.pipeline:
            doc = pipe(doc)
        assert doc[0]._.macron_morph == ""

    def test_orig_text_equals_normalized(self, nlp):
        # orig_text present but identical to normalized form (no macrons)
        tok = _orig(nlp, "uenit", "uenit")
        assert tok._.macron_morph == ""


# ---------------------------------------------------------------------------
# Unknown macronized forms — lookup miss
# ---------------------------------------------------------------------------

class TestUnknownForms:
    def test_unknown_macronized_form_returns_empty(self, nlp):
        # aquā is not in our test LOOKUP
        tok = _orig(nlp, "aqua", "aquā")
        assert tok._.macron_morph == ""
        assert tok._.macron_pos_ == ""

    def test_veniet_not_in_lookup(self, nlp):
        # future veniet: kaikki doesn't macronize or store it
        tok = _orig(nlp, "ueniet", "veniēt")  # hypothetical macronized form
        assert tok._.macron_morph == ""
