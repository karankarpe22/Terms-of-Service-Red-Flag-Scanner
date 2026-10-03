"""Attention level detection module for Terms of Service clauses.

Assigns an attention level:
- High Attention
- Medium Attention
- Low Attention
- Informational

Non-negotiable guidelines:
- This is NOT a legal-risk score.
- The system must never assert or imply that a clause is illegal, unlawful,
  invalid, unenforceable, legally risky, or legally compliant.
- Evaluates transparent, deterministic contractual indicators requiring closer human review.
"""
import re
from typing import List, Dict, Any, Tuple
from backend.config import settings
from backend.core.constants import (
    ATTENTION_LEVEL_HIGH,
    ATTENTION_LEVEL_MEDIUM,
    ATTENTION_LEVEL_LOW,
    ATTENTION_LEVEL_INFORMATIONAL,
)
from backend.document.models import Clause

# Strong attention indicator rules (pattern, human-readable indicator description)
STRONG_ATTENTION_RULES: List[Tuple[re.Pattern, str]] = [
    # Payment indicators
    (
        re.compile(r"\b(auto(matic(ally)?)?[\s\-]*renew(al|s|ed)?|recurring\s+bill(ing)?)\b", re.I),
        "Contains automatic renewal or recurring subscription billing",
    ),
    (
        re.compile(r"\b(non-refundable|no\s+refund(s)?|all\s+fees\s+are\s+final)\b", re.I),
        "Specifies that paid fees and charges are non-refundable",
    ),
    (
        re.compile(r"\b(at\s+least\s+\d+\s+days\s+prior|cancell(ation)?\s+notice\s+window|prior\s+to\s+(the\s+)?(expiration|renewal|billing))\b", re.I),
        "Imposes a specific advance notice deadline to prevent automatic renewal",
    ),
    (
        re.compile(r"\b(modify\s+pricing|change\s+(the\s+)?(fees|price|pricing)\s+at\s+any\s+time)\b", re.I),
        "Reserves the right to change fees or pricing unilaterally",
    ),
    # Account indicators
    (
        re.compile(r"\b(terminat(e|ion)\s+.*without\s+(prior\s+)?notice|suspend\s+.*without\s+(prior\s+)?notice|immediate(ly)?\s+.*without\s+notice)\b", re.I),
        "Allows immediate account suspension or termination without prior notice",
    ),
    (
        re.compile(r"\b(terminat(e|ion)\s+(of\s+)?(your\s+)?account\s+at\s+(our\s+sole\s+discretion|any\s+time)|suspend\s+(or\s+terminate\s+)?(your\s+)?account\s+at\s+any\s+time)\b", re.I),
        "Permits the company to terminate or suspend user accounts at any time",
    ),
    (
        re.compile(r"\b(without\s+(any\s+)?liability|for\s+any\s+reason\s+whatsoever|at\s+our\s+sole\s+discretion)\b", re.I),
        "Permits account action without liability or at sole company discretion",
    ),
    (
        re.compile(r"\b(permanently\s+delete|forfeit(ure)?\s+of\s+account|destroy\s+all\s+data)\b", re.I),
        "Specifies permanent account deletion or forfeiture upon termination",
    ),
    # Dispute indicators
    (
        re.compile(r"\b(binding\s+(individual\s+)?arbitrat(ion|or)|exclusive(ly)?\s+through\s+.*arbitrat(ion|or))\b", re.I),
        "Mandates binding individual arbitration in place of court proceedings",
    ),
    (
        re.compile(r"\b(class\s+action\s+waiver|waive\s+(any\s+right\s+to\s+participate\s+in\s+)?a\s+class\s+action|collective\s+arbitration|representative\s+action)\b", re.I),
        "Includes a waiver of the right to participate in class action lawsuits",
    ),
    (
        re.compile(r"\b(exclusive\s+jurisdiction\s+of|venue\s+in\s+[A-Za-z]+|courts\s+of\s+[A-Za-z]+)\b", re.I),
        "Restricts legal proceedings to a specific exclusive jurisdiction or venue",
    ),
    # Liability indicators
    (
        re.compile(r"\b(aggregate\s+liability|liability\s+(shall\s+)?(not\s+)?exceed|limited\s+to\s+(\$?\d+|the\s+amount\s+paid)|exceed\s+\$?\d+)\b", re.I),
        "Caps maximum recoverable liability to a nominal amount or fees paid",
    ),
    (
        re.compile(r"\b(in\s+no\s+event\s+shall\s+.*be\s+liable\s+for\s+any\s+(special|indirect|consequential|punitive)\s+damages)\b", re.I),
        "Broadly disclaims liability for consequential, indirect, or punitive damages",
    ),
    (
        re.compile(r"\b(indemnify,\s+defend,\s+and\s+hold\s+harmless|agree\s+to\s+indemnify)\b", re.I),
        "Obligates user to indemnify and hold the company harmless from claims",
    ),
    # User Content indicators
    (
        re.compile(r"\b(perpetual|irrevocable|royalty-free|worldwide,\s+perpetual)\b", re.I),
        "Grants a perpetual, irrevocable, or royalty-free license to user content",
    ),
    (
        re.compile(r"\b(sublicensable|transferable\s+license|right\s+to\s+(commercialize|exploit|modify|distribute))\b", re.I),
        "Permits the service to sublicense, modify, or commercially distribute user submissions",
    ),
    # Privacy & Data indicators
    (
        re.compile(r"\b(share\s+(your\s+)?(personal\s+)?(data|information)\s+with\s+third\s+parties|disclose\s+to\s+(our\s+)?(partners|affiliates|third\s+parties))\b", re.I),
        "Permits sharing personal data with third-party partners or commercial affiliates",
    ),
    (
        re.compile(r"\b(retain\s+.*(after|following)\s+(account\s+)?(closure|termination)|indefinite(ly)?\s+retain|retain\s+.*for\s+up\s+to\s+\d+\s+years)\b", re.I),
        "Provides for prolonged or post-termination retention of personal data",
    ),
    (
        re.compile(r"\b(cross-site\s+tracking|behavioral\s+advertising|profile\s+your\s+activity)\b", re.I),
        "Discloses persistent user activity tracking or behavioral profiling",
    ),
    # Changes indicators
    (
        re.compile(r"\b(modify\s+(or\s+replace\s+)?these\s+terms\s+.*without\s+(prior\s+)?(individual\s+)?notice|at\s+any\s+time\s+without\s+notice)\b", re.I),
        "Reserves the right to unilaterally modify terms without advance notice",
    ),
    (
        re.compile(r"\b(continued\s+use\s+.*constitutes\s+(binding\s+)?acceptance)\b", re.I),
        "Deems continued service use as binding acceptance of revised terms",
    ),
]

# Weak or general contractual indicators
WEAK_ATTENTION_RULES: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"\b(as-is|as\s+is|without\s+warranty\s+of\s+any\s+kind)\b", re.I),
        "Standard 'as-is' warranty disclaimer",
    ),
    (
        re.compile(r"\b(terms\s+may\s+be\s+updated|periodic(ally)?\s+review)\b", re.I),
        "General advice to review terms periodically for updates",
    ),
    (
        re.compile(r"\b(cookies\s+to\s+improve\s+experience|standard\s+cookies)\b", re.I),
        "Standard cookie notice for service functionality",
    ),
]


def detect_attention_level(clause_text: str) -> Tuple[str, List[str]]:
    """Determine the attention level and transparent indicators for a clause text.

    Returns:
        Tuple of (attention_level, list_of_reasons)
    """
    matched_reasons: List[str] = []

    # Check strong attention rules
    for pattern, reason in STRONG_ATTENTION_RULES:
        if pattern.search(clause_text):
            matched_reasons.append(reason)

    strong_count = len(matched_reasons)

    # Check weak rules only if no strong rules matched
    weak_reasons: List[str] = []
    if strong_count == 0:
        for pattern, reason in WEAK_ATTENTION_RULES:
            if pattern.search(clause_text):
                weak_reasons.append(reason)

    # Transparent deterministic assignment
    if strong_count >= settings.ATTENTION_HIGH_THRESHOLD:
        level = ATTENTION_LEVEL_HIGH
        reasons = matched_reasons
    elif strong_count >= settings.ATTENTION_MEDIUM_THRESHOLD:
        level = ATTENTION_LEVEL_MEDIUM
        reasons = matched_reasons
    elif len(weak_reasons) > 0:
        level = ATTENTION_LEVEL_LOW
        reasons = weak_reasons
    else:
        level = ATTENTION_LEVEL_INFORMATIONAL
        reasons = ["Standard informational clause without high-attention contractual indicators."]

    return level, reasons


def assign_attention_levels(clauses: List[Clause]) -> List[Clause]:
    """Assign attention levels across all clauses in a document."""
    for clause in clauses:
        level, reasons = detect_attention_level(clause.text)
        clause.attention_level = level
        clause.attention_reasons = reasons
    return clauses
