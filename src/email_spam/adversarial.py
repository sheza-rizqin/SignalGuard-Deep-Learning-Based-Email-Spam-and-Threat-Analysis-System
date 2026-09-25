
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
URL_RE = re.compile(r'(?:https?://|www\.)[^\s<>"\']+', re.IGNORECASE)
PUNCT_RUN_RE = re.compile(r"([!?.]){2,}")

ZWSP = "\u200b"
SOFT_HYPHEN = "\u00ad"
WORD_JOINER = "\u2060"
RTL_OVERRIDE = "\u202e"
LEFT_TO_RIGHT_MARK = "\u200e"

LEET_MAP = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}
CYRILLIC_MAP = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у"}

SENSITIVE_WORDS = frozenset(
    {
        "verify", "account", "password", "login", "secure", "update", "confirm",
        "bank", "payment", "urgent", "click", "signin", "credential", "billing",
        "wallet", "invoice", "suspend", "unlock", "validate", "recover",
    }
)

NEUTRAL_WORDS = ("please", "note", "regarding", "kindly", "further")

INSIDE_WORD_ANCHORS = (2, 4, 6)


@dataclass(frozen=True)
class Transformation:
    """One named adversarial probe."""

    key: str
    name: str
    family: str
    description: str
    rationale: str
    hypothesis: str

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "rationale": self.rationale,
            "hypothesis": self.hypothesis,
        }


def _replace_words(text: str, mapper: Callable[[str], str]) -> str:
    return WORD_RE.sub(lambda match: mapper(match.group(0)), text)


def _is_sensitive(word: str) -> bool:
    return word.lower() in SENSITIVE_WORDS


def _insert_inside(token: str, character: str) -> str:
    """Insert a character inside a token so the token is no longer a dictionary word."""
    if len(token) < 3:
        return token
    position = max(1, len(token) // 2)
    return token[:position] + character + token[position:]


def _intersperse_words(text: str, character: str) -> str:
    return _replace_words(text, lambda word: _insert_inside(word, character))


def _map_words(text: str, mapping: dict[str, str]) -> str:
    def mapper(word: str) -> str:
        if not _is_sensitive(word):
            return word
        for index, character in enumerate(word.lower()):
            if character in mapping:
                return word[:index] + mapping[character] + word[index + 1 :]
        return word

    return _replace_words(text, mapper)


def _map_urls(text: str, mapper: Callable[[str], str]) -> str:
    return URL_RE.sub(lambda match: mapper(match.group(0)), text)


def _case_title(text: str) -> str:
    return WORD_RE.sub(lambda match: match.group(0)[:1].upper() + match.group(0)[1:].lower(), text)


def _case_alternating(text: str) -> str:
    def mapper(word: str) -> str:
        return "".join(
            character.upper() if index % 2 else character.lower()
            for index, character in enumerate(word)
        )

    return WORD_RE.sub(lambda match: mapper(match.group(0)), text)


def _case_upper(text: str) -> str:
    return text.upper()


def _case_lower(text: str) -> str:
    return text.lower()


def _whitespace_pad(text: str) -> str:
    return re.sub(r" ", "   ", text)


def _whitespace_tabs(text: str) -> str:
    return text.replace(" ", "\t")


def _whitespace_double_newline(text: str) -> str:
    return text.replace("\n", "\n\n").replace(". ", ".\n\n")


def _punct_collapse(text: str) -> str:
    return PUNCT_RUN_RE.sub(lambda match: match.group(1), text)


def _punct_strip(text: str) -> str:
    return "".join("" if unicodedata.category(character).startswith("P") else character for character in text)


def _punct_double(text: str) -> str:
    return re.sub(r"([!?])", r"\1\1", text)


def _unicode_fullwidth(text: str) -> str:
    return "".join(
        chr(ord(character) + 0xFEE0) if 0x21 <= ord(character) <= 0x7E else character
        for character in text
    )


def _unicode_math_bold(text: str) -> str:
    return "".join(
        chr(ord(character) + 0x1D400 - ord("A"))
        if "A" <= character <= "Z"
        else chr(ord(character) + 0x1D41A - ord("a"))
        if "a" <= character <= "z"
        else character
        for character in text
    )


def _unicode_cyrillic(text: str) -> str:
    return _map_words(text, CYRILLIC_MAP)


def _invisible_zwsp(text: str) -> str:
    return _intersperse_words(text, ZWSP)


def _invisible_soft_hyphen(text: str) -> str:
    return _intersperse_words(text, SOFT_HYPHEN)


def _invisible_word_joiner(text: str) -> str:
    return _intersperse_words(text, WORD_JOINER)


def _invisible_bidi_marks(text: str) -> str:
    return _replace_words(text, lambda word: LEFT_TO_RIGHT_MARK + word + RTL_OVERRIDE)


def _obfuscate_leet(text: str) -> str:
    return _map_words(text, LEET_MAP)


def _obfuscate_dot_split(text: str) -> str:
    def mapper(word: str) -> str:
        if not _is_sensitive(word):
            return word
        middle = max(1, len(word) // 2)
        return word[:middle] + "." + word[middle:]

    return _replace_words(text, mapper)


def _insert_neutral_words(text: str) -> str:
    tokens = text.split()
    output: list[str] = []
    for index, token in enumerate(tokens):
        output.append(token)
        if index and index % 12 == 0:
            output.append(NEUTRAL_WORDS[(index // 12 - 1) % len(NEUTRAL_WORDS)])
    return " ".join(output)


def _url_uppercase(text: str) -> str:
    return _map_urls(text, lambda url: url.upper())


def _url_angle_brackets(text: str) -> str:
    return _map_urls(text, lambda url: f"<{url}>")


def _url_defang_spaces(text: str) -> str:
    return _map_urls(text, lambda url: re.sub(r"\.", " . ", url))


def _url_benign_query(text: str) -> str:
    def mapper(url: str) -> str:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}ref=newsletter"

    return _map_urls(text, mapper)


IMPLEMENTATIONS: dict[str, Callable[[str], str]] = {
    "case_upper": _case_upper,
    "case_lower": _case_lower,
    "case_title": _case_title,
    "case_alternating": _case_alternating,
    "whitespace_pad": _whitespace_pad,
    "whitespace_tabs": _whitespace_tabs,
    "whitespace_newlines": _whitespace_double_newline,
    "punct_collapse": _punct_collapse,
    "punct_strip": _punct_strip,
    "punct_double": _punct_double,
    "unicode_fullwidth": _unicode_fullwidth,
    "unicode_math_bold": _unicode_math_bold,
    "unicode_cyrillic": _unicode_cyrillic,
    "invisible_zwsp": _invisible_zwsp,
    "invisible_soft_hyphen": _invisible_soft_hyphen,
    "invisible_word_joiner": _invisible_word_joiner,
    "invisible_bidi_marks": _invisible_bidi_marks,
    "obfuscate_leet": _obfuscate_leet,
    "obfuscate_dot_split": _obfuscate_dot_split,
    "insert_neutral_words": _insert_neutral_words,
    "url_uppercase": _url_uppercase,
    "url_angle_brackets": _url_angle_brackets,
    "url_defang_spaces": _url_defang_spaces,
    "url_benign_query": _url_benign_query,
}


def _probe(
    key: str, name: str, family: str, description: str, rationale: str, hypothesis: str
) -> Transformation:
    return Transformation(key, name, family, description, rationale, hypothesis)


TRANSFORMATIONS: tuple[Transformation, ...] = (
    _probe("case_upper", "Uppercase", "case",
           "Convert the entire message to upper case.",
           "Semantics are unchanged; only letter case differs.",
           "Neutralised: the trained tokenizer lowercases all input."),
    _probe("case_lower", "Lowercase", "case",
           "Convert the entire message to lower case.",
           "Semantics are unchanged; only letter case differs.",
           "Neutralised: the trained tokenizer lowercases all input."),
    _probe("case_title", "Title case", "case",
           "Title-case every word.",
           "Semantics are unchanged; only letter case differs.",
           "Neutralised: the trained tokenizer lowercases all input."),
    _probe("case_alternating", "Alternating case", "case",
           "Alternate letter case inside each word (vErIfY).",
           "Letter identity and order are unchanged.",
           "Neutralised: the trained tokenizer lowercases all input."),
    _probe("whitespace_pad", "Space padding", "whitespace",
           "Replace single spaces with three spaces.",
           "Word boundaries and order are unchanged.",
           "Neutralised: whitespace is collapsed after normalisation."),
    _probe("whitespace_tabs", "Tab separators", "whitespace",
           "Replace spaces with tab characters.",
           "Word boundaries and order are unchanged.",
           "Likely neutralised: tabs are stripped by the tokenizer filter set."),
    _probe("whitespace_newlines", "Newline injection", "whitespace",
           "Double newlines and split sentences onto separate lines.",
           "Sentence order and content are unchanged.",
           "Likely neutralised: newlines are stripped by the tokenizer filter set."),
    _probe("punct_collapse", "Punctuation collapse", "punctuation",
           "Reduce repeated !, ? and . runs to a single character.",
           "Emphasis markers are reduced, wording is unchanged.",
           "Likely neutralised: ASCII punctuation is stripped by the tokenizer."),
    _probe("punct_strip", "Punctuation removal", "punctuation",
           "Remove all punctuation characters.",
           "Wording and word order are unchanged.",
           "Likely neutralised: ASCII punctuation is stripped by the tokenizer."),
    _probe("punct_double", "Punctuation doubling", "punctuation",
           "Double every ! and ? character.",
           "Emphasis is increased, wording is unchanged.",
           "Likely neutralised: ASCII punctuation is stripped by the tokenizer."),
    _probe("unicode_fullwidth", "Full-width characters", "unicode_variation",
           "Map ASCII characters to their full-width forms.",
           "Full-width forms are a compatibility variant of the same characters.",
           "Neutralised: NFKC folds full-width characters back to ASCII."),
    _probe("unicode_math_bold", "Mathematical alphanumerics", "unicode_variation",
           "Map Latin letters to mathematical bold code points.",
           "Mathematical alphanumerics are a compatibility variant of the same letters.",
           "Neutralised: NFKC folds mathematical alphanumerics back to ASCII."),
    _probe("unicode_cyrillic", "Cyrillic homoglyphs", "unicode_variation",
           "Replace Latin letters in sensitive words with visually identical Cyrillic letters.",
           "The rendered text is visually indistinguishable from the original.",
           "Expected to survive: NFKC does not map Cyrillic to Latin."),
    _probe("invisible_zwsp", "Zero-width space", "invisible",
           "Insert U+200B inside words longer than two characters.",
           "The character renders as nothing, so the visible message is unchanged.",
           "Expected to survive: U+200B is Unicode category Cf and is not whitespace."),
    _probe("invisible_soft_hyphen", "Soft hyphen", "invisible",
           "Insert U+00AD inside words longer than two characters.",
           "Soft hyphens render only when a line breaks.",
           "Expected to survive: U+00AD is Unicode category Cf."),
    _probe("invisible_word_joiner", "Word joiner", "invisible",
           "Insert U+2060 inside words longer than two characters.",
           "The word joiner has no visible glyph.",
           "Expected to survive: U+2060 is Unicode category Cf."),
    _probe("invisible_bidi_marks", "Bidirectional marks", "invisible",
           "Wrap each word with U+200E and U+202E direction controls.",
           "Direction controls reorder rendering but not the logical text.",
           "Expected to survive: bidi controls are Unicode category Cf."),
    _probe("obfuscate_leet", "Leetspeak substitution", "obfuscation",
           "Substitute one letter per sensitive word with a lookalike symbol (o->0, e->3).",
           "Leetspeak remains readable to a human reader.",
           "Might survive: substitution changes the token identity itself."),
    _probe("obfuscate_dot_split", "Intra-word dot", "obfuscation",
           "Split sensitive words with an interior dot (ver.ify).",
           "The dot is a separator convention, not a semantic change.",
           "Likely neutralised: the dot is stripped and both halves remain."),
    _probe("insert_neutral_words", "Neutral word padding", "insertion",
           "Insert filler words every twelve words.",
           "Added words carry no persuasive content.",
           "Might survive: adds tokens and changes sequence position, not vocabulary."),
    _probe("url_uppercase", "URL uppercasing", "url",
           "Upper-case the entire URL string.",
           "URL hosts are case-insensitive.",
           "Neutralised: URLs are replaced by a single placeholder token."),
    _probe("url_angle_brackets", "URL angle brackets", "url",
           "Wrap each URL in angle brackets.",
           "Angle brackets are a conventional URL delimiter.",
           "Likely neutralised: the URL placeholder substitution still matches."),
    _probe("url_defang_spaces", "URL defanging", "url",
           "Insert spaces around the dots inside URLs.",
           "Defanging is a standard convention for writing a URL safely.",
           "Might survive: spacing breaks the URL pattern match, so it is not replaced."),
    _probe("url_benign_query", "Benign URL query", "url",
           "Append a harmless query parameter to each URL.",
           "A tracking parameter does not change the destination.",
           "Likely neutralised: URLs are replaced by a single placeholder token."),
)

TRANSFORMATIONS_BY_KEY = {transformation.key: transformation for transformation in TRANSFORMATIONS}


def apply_transformation(text: str, key: str) -> str:
    implementation = IMPLEMENTATIONS[key]
    return implementation(text)


def apply_all(text: str) -> list[tuple[Transformation, str]]:
    """Return every probe paired with its transformed text."""
    return [(transformation, apply_transformation(text, transformation.key)) for transformation in TRANSFORMATIONS]
