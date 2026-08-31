from __future__ import annotations

import re


DEFAULT_PROTECTED_TERMS = [
    "attention",
    "self-attention",
    "multi-head attention",
    "Transformer",
    "softmax",
    "embedding",
    "token",
    "logit",
]

EQUATION_RE = re.compile(
    r"(\$\$.*?\$\$|\$[^$\n]+\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))",
    re.DOTALL,
)


def protect_text(text: str, protected_terms: list[str] | None = None) -> tuple[str, dict[str, str]]:
    """Replace equations and glossary terms with stable placeholders."""
    terms = protected_terms or []
    mapping: dict[str, str] = {}

    def replace_equation(match: re.Match[str]) -> str:
        key = f"<KEEP_EQ_{len(mapping)}/>"
        mapping[key] = match.group(0)
        return key

    protected = EQUATION_RE.sub(replace_equation, text)

    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(term)}(?![A-Za-z0-9_-])")

        def replace_term(match: re.Match[str]) -> str:
            key = f"<KEEP_TERM_{len(mapping)}/>"
            mapping[key] = match.group(0)
            return key

        protected = pattern.sub(replace_term, protected)

    return protected, mapping


def restore_text(text: str, mapping: dict[str, str]) -> str:
    restored = text
    for key, original in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        restored = restored.replace(key, original)
    return restored
