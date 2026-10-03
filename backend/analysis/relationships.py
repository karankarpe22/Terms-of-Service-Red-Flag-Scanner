"""Deterministic Cross-Clause Relationship Analysis Engine.

Identifies structural, semantic, and textual relationships between clauses within
the SAME document without asserting legal validity, enforceability, or compliance.
Uses neutral, evidence-grounded reporting with transparent rule-based triggers.

Supported canonical relationship types:
- SUPPORTS
- QUALIFIES
- EXCEPTS
- OVERRIDES
- DEPENDS_ON
- TEMPORAL
- POTENTIAL_TENSION (used in place of 'CONFLICT')
"""
import re
import uuid
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set


def _make_relationship_id(source_id: str, target_id: str, rel_type: str) -> str:
    """Generate a deterministic, reproducible relationship ID."""
    token = f"{source_id}_{target_id}_{rel_type}"
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
    return f"rel_{h}"


from backend.config import settings
from backend.core.constants import (
    RELATIONSHIP_TYPE_SUPPORTS,
    RELATIONSHIP_TYPE_QUALIFIES,
    RELATIONSHIP_TYPE_EXCEPTS,
    RELATIONSHIP_TYPE_OVERRIDES,
    RELATIONSHIP_TYPE_DEPENDS_ON,
    RELATIONSHIP_TYPE_TEMPORAL,
    RELATIONSHIP_TYPE_POTENTIAL_TENSION,
    CATEGORY_RELATIONSHIP_MATRIX,
)
from backend.document.models import Clause
from backend.analysis.models import (
    ClauseRelationship,
    RelationshipEvidence,
    CandidatePair,
)
from backend.retrieval.embedder import get_embedder


def _clause_label(c: Clause) -> str:
    """Return a descriptive, human-readable label for a clause in relationship rationales."""
    has_title = (
        bool(c.section_title)
        and c.section_title.strip() != ""
        and c.section_title.strip().lower() not in ("general", "pasted text", "terms of service", "unknown")
    )
    cat = c.primary_category or c.category
    has_cat = bool(cat) and cat.strip() != "" and cat.strip().lower() not in ("general", "unknown")

    if has_title and has_cat:
        return f"Clause #{c.clause_index} ('{c.section_title}' - {cat})"
    elif has_title:
        return f"Clause #{c.clause_index} ('{c.section_title}')"
    elif has_cat:
        return f"Clause #{c.clause_index} ({cat})"
    else:
        return f"Clause #{c.clause_index}"


# ---------------------------------------------------------------------------
# Key Domain Vocabulary for Candidate Pairing
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = {
    "account", "subscription", "payment", "billing", "fee", "fees", "refund",
    "data", "privacy", "personal information", "content", "license",
    "termination", "terminate", "suspend", "cancel", "cancellation",
    "renewal", "renew", "notice", "notify", "modify", "modification",
    "change", "liability", "damage", "damages", "dispute", "arbitration",
    "waiver", "retention", "retain", "remedy", "governing law"
}

# Regex to detect section numbering or explicit section identifiers
SECTION_REF_PATTERN = re.compile(
    r"\b(?:Section|Article|Clause|Paragraph)\s+(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?)",
    re.IGNORECASE,
)
SECTION_HEADING_NUM_PATTERN = re.compile(
    r"^\s*(?:(?:SECTION|Section|ARTICLE|Article)\s+)?(\d+(?:\.\d+)*)\.?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Rule Trigger Patterns
# ---------------------------------------------------------------------------

# OVERRIDES: explicit supremacy or precedence language
OVERRIDES_PATTERNS = [
    (
        re.compile(
            r"\b(?:in the event of (?:any )?conflict|in case of conflict|in the event of inconsistency|"
            r"this (?:section|provision|agreement) shall (?:control|prevail|supersede|take precedence)|"
            r"supersedes (?:all )?(?:prior|other)|prevails over|shall control over)\b",
            re.IGNORECASE,
        ),
        "Overrides / Takes Precedence Clause",
    ),
]

# EXCEPTS: explicit exception phrasing
EXCEPTS_PATTERNS = [
    (
        re.compile(
            r"\b(?:except (?:as|for|that|in)|with the exception of|"
            r"save for|unless otherwise (?:expressly )?stated|other than as provided)\b",
            re.IGNORECASE,
        ),
        "Explicit Exception Provision",
    ),
]

# DEPENDS_ON: operational dependency or cross-reference contingency
DEPENDS_ON_PATTERNS = [
    (
        re.compile(
            r"\b(?:subject to (?:the terms of|section)?|conditional upon|conditioned upon|"
            r"contingent (?:on|upon)|only if|depending on|as (?:described|set forth|specified) in|"
            r"in accordance with (?:section)?|pursuant to (?:section)?)\b",
            re.IGNORECASE,
        ),
        "Conditional Operational Dependency",
    ),
]

# QUALIFIES: scope limitations or restrictive provisos
QUALIFIES_PATTERNS = [
    (
        re.compile(
            r"\b(?:to the (?:maximum )?extent permitted by(?: applicable)? law|"
            r"without limiting (?:the generality of)?|"
            r"provided(?:, however,)? that|subject to applicable law|solely for the purpose of|"
            r"limited to)\b",
            re.IGNORECASE,
        ),
        "Scope Limitation or Proviso",
    ),
]

# TEMPORAL: timing sequence, notice windows, termination conditions, retention horizons
TEMPORAL_PATTERNS = [
    (
        re.compile(
            r"\b(?:at least \d+ days prior|prior to (?:the )?(?:expiration|renewal|billing)|"
            r"suspend or terminate|terminate (?:your )?account|upon (?:account )?(?:termination|closure)|"
            r"after (?:account )?closure|retain (?:your )?(?:personal )?data|"
            r"for a period of (?:up to )?\d+|notice period|recurring (?:monthly|annual)|billing cycle)\b",
            re.IGNORECASE,
        ),
        "Temporal Timing or Duration Window",
    ),
]

# POTENTIAL_TENSION Heuristic Triggers
CANCEL_ANYTIME_PATTERN = re.compile(
    r"\b(?:terminate your account at any time|cancel (?:your )?(?:subscription|account) at any time|"
    r"may cancel at any time|at any time through your account settings)\b",
    re.IGNORECASE,
)
STRICT_RENEWAL_NOTICE_PATTERN = re.compile(
    r"\b(?:automatically renew|billed in advance|at least \d+ days prior|non-refundable|all (?:subscription )?fees.*non-refundable)\b",
    re.IGNORECASE,
)

MODIFY_NO_NOTICE_PATTERN = re.compile(
    r"\b(?:modify.*without (?:prior )?(?:individual )?notice|replace these terms.*without (?:prior )?notice|"
    r"without (?:prior )?notice or liability)\b",
    re.IGNORECASE,
)
NOTIFY_CHANGES_PATTERN = re.compile(
    r"\b(?:will notify (?:users|you)|prior notice before (?:material )?changes|"
    r"notice of (?:any )?(?:material )?change|\d+ days (?:prior )?notice of changes)\b",
    re.IGNORECASE,
)

IMMEDIATE_TERM_WITHOUT_NOTICE_PATTERN = re.compile(
    r"\b(?:suspend or terminate.*immediately, without notice|without notice or liability, for any reason)\b",
    re.IGNORECASE,
)
CURE_OR_NOTICE_TERM_PATTERN = re.compile(
    r"\b(?:upon \d+ days (?:written )?notice|opportunity to cure|prior written notice of termination)\b",
    re.IGNORECASE,
)

NON_REFUNDABLE_PATTERN = re.compile(
    r"\b(?:non-refundable|all fees (?:paid )?are final|no refunds)\b",
    re.IGNORECASE,
)
REFUND_ELIGIBLE_PATTERN = re.compile(
    r"\b(?:eligible for (?:a )?refund|money-back guarantee|prorated refund|refund will be issued)\b",
    re.IGNORECASE,
)


class CrossClauseRelationshipEngine:
    """Deterministic, explainable engine for identifying cross-clause relationships."""

    def __init__(
        self,
        semantic_threshold: Optional[float] = None,
        max_candidates_per_clause: Optional[int] = None,
        max_total_candidates: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ):
        self.semantic_threshold = (
            semantic_threshold
            if semantic_threshold is not None
            else settings.RELATIONSHIP_SEMANTIC_THRESHOLD
        )
        self.max_candidates_per_clause = (
            max_candidates_per_clause
            if max_candidates_per_clause is not None
            else settings.MAX_CANDIDATE_PAIRS_PER_CLAUSE
        )
        self.max_total_candidates = (
            max_total_candidates
            if max_total_candidates is not None
            else settings.MAX_TOTAL_CANDIDATE_PAIRS
        )
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else settings.RELATIONSHIP_MIN_CONFIDENCE
        )

    # -----------------------------------------------------------------------
    # 1. Candidate Pair Generation (Pruning to avoid O(N^2) explosion)
    # -----------------------------------------------------------------------
    def generate_candidate_pairs(self, clauses: List[Clause]) -> List[Tuple[Clause, Clause, List[str], float]]:
        """Generate prioritized candidate clause pairs using multiple filtering signals.

        Returns:
            List of (clause_a, clause_b, match_reasons, priority_score)
        """
        n = len(clauses)
        if n < 2:
            return []

        # Index clauses by section identifier (e.g. "1", "2", "3")
        section_index: Dict[str, Clause] = {}
        for c in clauses:
            title = c.section_title or ""
            match = SECTION_HEADING_NUM_PATTERN.match(title)
            if match:
                sec_num = match.group(1).rstrip(".")
                section_index[sec_num] = c

        # Compute embeddings once for all clauses for semantic candidate generation
        embedder = get_embedder()
        clause_texts = [c.text for c in clauses]
        embeddings = embedder.encode(clause_texts)

        # Precompute keywords set per clause
        clause_kw_sets = []
        for c in clauses:
            text_lower = c.text.lower()
            kws = {kw for kw in DOMAIN_KEYWORDS if kw in text_lower}
            clause_kw_sets.append(kws)

        candidates_per_clause: Dict[int, List[Tuple[int, List[str], float]]] = {
            i: [] for i in range(n)
        }

        # Pairwise candidate scoring
        for i in range(n):
            c_a = clauses[i]
            emb_a = embeddings[i]
            kws_a = clause_kw_sets[i]
            cat_a = c_a.primary_category or "General"

            for j in range(i + 1, n):
                c_b = clauses[j]
                emb_b = embeddings[j]
                kws_b = clause_kw_sets[j]
                cat_b = c_b.primary_category or "General"

                reasons: List[str] = []
                score = 0.0

                # A. Explicit Cross-References
                # Check if c_a references c_b's section or vice-versa
                refs_in_a = SECTION_REF_PATTERN.findall(c_a.text)
                for r in refs_in_a:
                    r_clean = r.rstrip(".")
                    if r_clean in section_index and section_index[r_clean].clause_id == c_b.clause_id:
                        reasons.append(f"Explicit cross-reference to Section {r}")
                        score += 0.40

                refs_in_b = SECTION_REF_PATTERN.findall(c_b.text)
                for r in refs_in_b:
                    r_clean = r.rstrip(".")
                    if r_clean in section_index and section_index[r_clean].clause_id == c_a.clause_id:
                        reasons.append(f"Explicit cross-reference to Section {r}")
                        score += 0.40

                # B. Category Relationship Matrix & Same Category
                if cat_a == cat_b and cat_a != "General":
                    reasons.append(f"Shared category ({cat_a})")
                    score += 0.35
                elif cat_b in CATEGORY_RELATIONSHIP_MATRIX.get(cat_a, set()) or cat_a in CATEGORY_RELATIONSHIP_MATRIX.get(cat_b, set()):
                    reasons.append(f"High-affinity category pairing ({cat_a} ↔ {cat_b})")
                    score += 0.30

                # C. Shared Important Keywords
                shared_kws = kws_a.intersection(kws_b)
                if len(shared_kws) >= 2:
                    reasons.append(f"Shared domain concepts: {', '.join(sorted(list(shared_kws))[:3])}")
                    score += 0.25
                elif len(shared_kws) == 1:
                    score += 0.15

                # D. Dense Semantic Similarity
                cos_sim = float(np.dot(emb_a, emb_b))
                if cos_sim >= self.semantic_threshold:
                    reasons.append(f"Semantic similarity score {cos_sim:.2f}")
                    score += 0.25

                # E. Rule Trigger Heuristics
                if (CANCEL_ANYTIME_PATTERN.search(c_a.text) and STRICT_RENEWAL_NOTICE_PATTERN.search(c_b.text)) or \
                   (CANCEL_ANYTIME_PATTERN.search(c_b.text) and STRICT_RENEWAL_NOTICE_PATTERN.search(c_a.text)):
                    reasons.append("Cancellation vs auto-renewal tension trigger")
                    score += 0.40

                if (MODIFY_NO_NOTICE_PATTERN.search(c_a.text) and NOTIFY_CHANGES_PATTERN.search(c_b.text)) or \
                   (MODIFY_NO_NOTICE_PATTERN.search(c_b.text) and NOTIFY_CHANGES_PATTERN.search(c_a.text)):
                    reasons.append("Modification vs change notice tension trigger")
                    score += 0.40

                src_temp = any(p[0].search(c_a.text) for p in TEMPORAL_PATTERNS)
                tgt_temp = any(p[0].search(c_b.text) for p in TEMPORAL_PATTERNS)
                if src_temp and tgt_temp:
                    reasons.append("Both clauses contain temporal/timing indicators")
                    score += 0.35

                has_prior_link = (score > 0.0 or len(reasons) > 0)
                if any(p[0].search(c_a.text) for p in EXCEPTS_PATTERNS) or any(p[0].search(c_b.text) for p in EXCEPTS_PATTERNS):
                    reasons.append("Exception trigger phrasing present")
                    score += 0.30 if (has_prior_link or n <= 5) else 0.15

                if any(p[0].search(c_a.text) for p in QUALIFIES_PATTERNS) or any(p[0].search(c_b.text) for p in QUALIFIES_PATTERNS):
                    reasons.append("Qualifying proviso phrasing present")
                    score += 0.30 if (has_prior_link or n <= 5) else 0.15

                # Small documents: guarantee candidate coverage for minimal contracts
                if n <= 5 and not reasons:
                    reasons.append("Document-level candidate pairing")
                    score = max(score, 0.30)

                # Check if pair qualifies as candidate
                if score >= 0.25 or (n <= 5 and len(reasons) >= 1):
                    candidates_per_clause[i].append((j, reasons, score))

        # Filter candidates per clause to max_candidates_per_clause
        selected_pairs: List[Tuple[Clause, Clause, List[str], float]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for i in range(n):
            candidates = candidates_per_clause[i]
            candidates.sort(key=lambda x: x[2], reverse=True)
            top_for_i = candidates[: self.max_candidates_per_clause]

            for j, reasons, score in top_for_i:
                pair_key = (clauses[i].clause_id, clauses[j].clause_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    selected_pairs.append((clauses[i], clauses[j], reasons, score))

        # Globally sort and cap at max_total_candidates
        selected_pairs.sort(key=lambda x: x[3], reverse=True)
        return selected_pairs[: self.max_total_candidates]

    # -----------------------------------------------------------------------
    # 2. Deterministic Rule Evaluation
    # -----------------------------------------------------------------------
    def evaluate_pair(
        self,
        clause_a: Clause,
        clause_b: Clause,
        document_id: str,
    ) -> List[ClauseRelationship]:
        """Evaluate deterministic rules on a candidate pair in both directions."""
        relationships: List[ClauseRelationship] = []

        # Direction 1: A -> B
        rel_ab = self._check_rules_directional(clause_a, clause_b, document_id)
        if rel_ab:
            relationships.extend(rel_ab)

        # Direction 2: B -> A
        rel_ba = self._check_rules_directional(clause_b, clause_a, document_id)
        if rel_ba:
            relationships.extend(rel_ba)

        # Bidirectional check: POTENTIAL_TENSION (evaluated pair-wise)
        tension_rel = self._check_potential_tension(clause_a, clause_b, document_id)
        if tension_rel:
            relationships.append(tension_rel)

        return relationships

    def _check_rules_directional(
        self,
        src: Clause,
        tgt: Clause,
        document_id: str,
    ) -> List[ClauseRelationship]:
        """Check directional relationship rules where src acts upon tgt."""
        found: List[ClauseRelationship] = []
        src_text = src.text
        tgt_text = tgt.text

        # Evaluate topical connection between src and tgt
        cat_src = src.primary_category or "General"
        cat_tgt = tgt.primary_category or "General"
        same_cat = (cat_src == cat_tgt and cat_src != "General")
        matrix_affinity = (
            cat_tgt in CATEGORY_RELATIONSHIP_MATRIX.get(cat_src, set())
            or cat_src in CATEGORY_RELATIONSHIP_MATRIX.get(cat_tgt, set())
        )
        kws_src = {kw for kw in DOMAIN_KEYWORDS if kw in src_text.lower()}
        kws_tgt = {kw for kw in DOMAIN_KEYWORDS if kw in tgt_text.lower()}
        shared_kws = kws_src.intersection(kws_tgt)
        has_affinity = (
            same_cat
            or matrix_affinity
            or len(shared_kws) >= 1
            or abs(src.clause_index - tgt.clause_index) <= 1
            or len(SECTION_REF_PATTERN.findall(src_text)) > 0
        )

        # 1. OVERRIDES
        for pattern, label in OVERRIDES_PATTERNS:
            match = pattern.search(src_text)
            if match and has_affinity:
                trigger = match.group(0)
                src_lbl = _clause_label(src)
                tgt_lbl = _clause_label(tgt)
                rel = ClauseRelationship(
                    relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_OVERRIDES),
                    document_id=document_id,
                    source_clause_id=src.clause_id,
                    target_clause_id=tgt.clause_id,
                    relationship_type=RELATIONSHIP_TYPE_OVERRIDES,
                    confidence=0.85,
                    rationale=(
                        f"{src_lbl} contains explicit supremacy language ('{trigger}') "
                        f"establishing that it takes precedence over conflicting provisions in {tgt_lbl}."
                    ),
                    evidence=RelationshipEvidence(
                        source_text=src_text,
                        target_text=tgt_text,
                        trigger=trigger,
                    ),
                    source_location=src.source_location.to_display_string(),
                    target_location=tgt.source_location.to_display_string(),
                )
                found.append(rel)
                break

        # 2. EXCEPTS
        for pattern, label in EXCEPTS_PATTERNS:
            match = pattern.search(src_text)
            if match and has_affinity:
                trigger = match.group(0)
                src_lbl = _clause_label(src)
                tgt_lbl = _clause_label(tgt)
                rel = ClauseRelationship(
                    relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_EXCEPTS),
                    document_id=document_id,
                    source_clause_id=src.clause_id,
                    target_clause_id=tgt.clause_id,
                    relationship_type=RELATIONSHIP_TYPE_EXCEPTS,
                    confidence=0.80,
                    rationale=(
                        f"{src_lbl} creates an explicit exception ('{trigger}') "
                        f"limiting the scope or applicability of provisions in {tgt_lbl}."
                    ),
                    evidence=RelationshipEvidence(
                        source_text=src_text,
                        target_text=tgt_text,
                        trigger=trigger,
                    ),
                    source_location=src.source_location.to_display_string(),
                    target_location=tgt.source_location.to_display_string(),
                )
                found.append(rel)
                break

        # 3. DEPENDS_ON
        for pattern, label in DEPENDS_ON_PATTERNS:
            match = pattern.search(src_text)
            if match and has_affinity:
                trigger = match.group(0)
                src_lbl = _clause_label(src)
                tgt_lbl = _clause_label(tgt)
                rel = ClauseRelationship(
                    relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_DEPENDS_ON),
                    document_id=document_id,
                    source_clause_id=src.clause_id,
                    target_clause_id=tgt.clause_id,
                    relationship_type=RELATIONSHIP_TYPE_DEPENDS_ON,
                    confidence=0.78,
                    rationale=(
                        f"{src_lbl} contains an operational or conditional dependency ('{trigger}') "
                        f"governed by terms in {tgt_lbl}."
                    ),
                    evidence=RelationshipEvidence(
                        source_text=src_text,
                        target_text=tgt_text,
                        trigger=trigger,
                    ),
                    source_location=src.source_location.to_display_string(),
                    target_location=tgt.source_location.to_display_string(),
                )
                found.append(rel)
                break

        # 4. QUALIFIES
        for pattern, label in QUALIFIES_PATTERNS:
            match = pattern.search(src_text)
            if match and has_affinity:
                trigger = match.group(0)
                src_lbl = _clause_label(src)
                tgt_lbl = _clause_label(tgt)
                rel = ClauseRelationship(
                    relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_QUALIFIES),
                    document_id=document_id,
                    source_clause_id=src.clause_id,
                    target_clause_id=tgt.clause_id,
                    relationship_type=RELATIONSHIP_TYPE_QUALIFIES,
                    confidence=0.72,
                    rationale=(
                        f"{src_lbl} qualifies or limits the scope of obligations or rights ('{trigger}') "
                        f"described in {tgt_lbl}."
                    ),
                    evidence=RelationshipEvidence(
                        source_text=src_text,
                        target_text=tgt_text,
                        trigger=trigger,
                    ),
                    source_location=src.source_location.to_display_string(),
                    target_location=tgt.source_location.to_display_string(),
                )
                found.append(rel)
                break

        # 5. TEMPORAL
        # Check interaction along time, notice, retention, or renewal
        src_temp = any(p[0].search(src_text) for p in TEMPORAL_PATTERNS)
        tgt_temp = any(p[0].search(tgt_text) for p in TEMPORAL_PATTERNS)
        if src_temp and tgt_temp:
            # Look for specific trigger in source
            trigger_match = next((p[0].search(src_text) for p in TEMPORAL_PATTERNS if p[0].search(src_text)), None)
            trigger_text = trigger_match.group(0) if trigger_match else "Timing / Notice Window"
            src_lbl = _clause_label(src)
            tgt_lbl = _clause_label(tgt)
            rel = ClauseRelationship(
                relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_TEMPORAL),
                document_id=document_id,
                source_clause_id=src.clause_id,
                target_clause_id=tgt.clause_id,
                relationship_type=RELATIONSHIP_TYPE_TEMPORAL,
                confidence=0.75,
                rationale=(
                    f"{src_lbl} and {tgt_lbl} interact along a timeline ('{trigger_text}'), "
                    f"establishing sequential notice requirements, billing cycles, or post-termination retention horizons."
                ),
                evidence=RelationshipEvidence(
                    source_text=src_text,
                    target_text=tgt_text,
                    trigger=trigger_text,
                ),
                source_location=src.source_location.to_display_string(),
                target_location=tgt.source_location.to_display_string(),
            )
            found.append(rel)

        # 6. SUPPORTS
        # Cross-references or complementary categories that provide elaborative procedures
        if not found:
            # Check if one clause explicitly references another's section
            refs = SECTION_REF_PATTERN.findall(src_text)
            if any(r in (tgt.section_title or "") for r in refs):
                src_lbl = _clause_label(src)
                tgt_lbl = _clause_label(tgt)
                rel = ClauseRelationship(
                    relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_SUPPORTS),
                    document_id=document_id,
                    source_clause_id=src.clause_id,
                    target_clause_id=tgt.clause_id,
                    relationship_type=RELATIONSHIP_TYPE_SUPPORTS,
                    confidence=0.68,
                    rationale=(
                        f"{src_lbl} explicitly references and provides supporting details for {tgt_lbl}."
                    ),
                    evidence=RelationshipEvidence(
                        source_text=src_text,
                        target_text=tgt_text,
                        trigger=f"Section reference in text",
                    ),
                    source_location=src.source_location.to_display_string(),
                    target_location=tgt.source_location.to_display_string(),
                )
                found.append(rel)

        return found

    def _check_potential_tension(
        self,
        c_a: Clause,
        c_b: Clause,
        document_id: str,
    ) -> Optional[ClauseRelationship]:
        """Detect potential textual tension between two clauses without asserting legal inconsistency."""
        text_a = c_a.text
        text_b = c_b.text

        # Heuristic A: Cancel/Terminate at any time VS Strict Auto-Renewal / Advance Notice
        if (CANCEL_ANYTIME_PATTERN.search(text_a) and STRICT_RENEWAL_NOTICE_PATTERN.search(text_b)) or \
           (CANCEL_ANYTIME_PATTERN.search(text_b) and STRICT_RENEWAL_NOTICE_PATTERN.search(text_a)):
            src, tgt = (c_a, c_b) if CANCEL_ANYTIME_PATTERN.search(text_a) else (c_b, c_a)
            src_lbl = _clause_label(src)
            tgt_lbl = _clause_label(tgt)
            return ClauseRelationship(
                relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_POTENTIAL_TENSION),
                document_id=document_id,
                source_clause_id=src.clause_id,
                target_clause_id=tgt.clause_id,
                relationship_type=RELATIONSHIP_TYPE_POTENTIAL_TENSION,
                confidence=0.76,
                rationale=(
                    f"{src_lbl} and {tgt_lbl} describe different cancellation timing conditions: "
                    f"one provision allows cancellation at any time while the other requires advance notice prior to auto-renewal."
                ),
                evidence=RelationshipEvidence(
                    source_text=src.text,
                    target_text=tgt.text,
                    trigger="cancel at any time vs advance renewal notice requirement",
                ),
                source_location=src.source_location.to_display_string(),
                target_location=tgt.source_location.to_display_string(),
            )

        # Heuristic B: Modification without notice VS Notice commitment for changes
        if (MODIFY_NO_NOTICE_PATTERN.search(text_a) and NOTIFY_CHANGES_PATTERN.search(text_b)) or \
           (MODIFY_NO_NOTICE_PATTERN.search(text_b) and NOTIFY_CHANGES_PATTERN.search(text_a)):
            src, tgt = (c_a, c_b) if MODIFY_NO_NOTICE_PATTERN.search(text_a) else (c_b, c_a)
            src_lbl = _clause_label(src)
            tgt_lbl = _clause_label(tgt)
            return ClauseRelationship(
                relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_POTENTIAL_TENSION),
                document_id=document_id,
                source_clause_id=src.clause_id,
                target_clause_id=tgt.clause_id,
                relationship_type=RELATIONSHIP_TYPE_POTENTIAL_TENSION,
                confidence=0.82,
                rationale=(
                    f"{src_lbl} states that the company reserves the right to modify terms without prior notice, "
                    f"while {tgt_lbl} references notifying users of material changes."
                ),
                evidence=RelationshipEvidence(
                    source_text=src.text,
                    target_text=tgt.text,
                    trigger="unilateral modification without notice vs notification of changes",
                ),
                source_location=src.source_location.to_display_string(),
                target_location=tgt.source_location.to_display_string(),
            )

        # Heuristic C: Immediate termination without notice VS Notice / Cure period
        if (IMMEDIATE_TERM_WITHOUT_NOTICE_PATTERN.search(text_a) and CURE_OR_NOTICE_TERM_PATTERN.search(text_b)) or \
           (IMMEDIATE_TERM_WITHOUT_NOTICE_PATTERN.search(text_b) and CURE_OR_NOTICE_TERM_PATTERN.search(text_a)):
            src, tgt = (c_a, c_b) if IMMEDIATE_TERM_WITHOUT_NOTICE_PATTERN.search(text_a) else (c_b, c_a)
            src_lbl = _clause_label(src)
            tgt_lbl = _clause_label(tgt)
            return ClauseRelationship(
                relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_POTENTIAL_TENSION),
                document_id=document_id,
                source_clause_id=src.clause_id,
                target_clause_id=tgt.clause_id,
                relationship_type=RELATIONSHIP_TYPE_POTENTIAL_TENSION,
                confidence=0.78,
                rationale=(
                    f"{src_lbl} and {tgt_lbl} describe differing procedures for account termination: "
                    f"immediate termination without notice alongside provisions indicating an advance notice or cure window."
                ),
                evidence=RelationshipEvidence(
                    source_text=src.text,
                    target_text=tgt.text,
                    trigger="immediate termination without notice vs notice/cure period",
                ),
                source_location=src.source_location.to_display_string(),
                target_location=tgt.source_location.to_display_string(),
            )

        # Heuristic D: Non-refundable fees VS Explicit refund eligibility
        if (NON_REFUNDABLE_PATTERN.search(text_a) and REFUND_ELIGIBLE_PATTERN.search(text_b)) or \
           (NON_REFUNDABLE_PATTERN.search(text_b) and REFUND_ELIGIBLE_PATTERN.search(text_a)):
            src, tgt = (c_a, c_b) if NON_REFUNDABLE_PATTERN.search(text_a) else (c_b, c_a)
            src_lbl = _clause_label(src)
            tgt_lbl = _clause_label(tgt)
            return ClauseRelationship(
                relationship_id=_make_relationship_id(src.clause_id, tgt.clause_id, RELATIONSHIP_TYPE_POTENTIAL_TENSION),
                document_id=document_id,
                source_clause_id=src.clause_id,
                target_clause_id=tgt.clause_id,
                relationship_type=RELATIONSHIP_TYPE_POTENTIAL_TENSION,
                confidence=0.79,
                rationale=(
                    f"{src_lbl} specifies that fees are non-refundable, "
                    f"while {tgt_lbl} outlines circumstances for refund eligibility."
                ),
                evidence=RelationshipEvidence(
                    source_text=src.text,
                    target_text=tgt.text,
                    trigger="non-refundable fee statement vs refund eligibility condition",
                ),
                source_location=src.source_location.to_display_string(),
                target_location=tgt.source_location.to_display_string(),
            )

        return None


# ---------------------------------------------------------------------------
# Public Functional Interface
# ---------------------------------------------------------------------------

def detect_relationships(
    clauses: List[Clause],
    document_id: Optional[str] = None,
    max_relationships: Optional[int] = 100,
    min_confidence: Optional[float] = 0.50,
) -> Tuple[List[ClauseRelationship], Dict[str, Any]]:
    """Detect cross-clause relationships within a single document.

    Enforces strict document isolation:
    - All clauses must belong to the exact same document.
    - Relationships are never formed across different documents.

    Args:
        clauses: List of Clause objects from the same document.
        document_id: Optional document ID to validate against.
        max_relationships: Max count of relationships to return.
        min_confidence: Minimum algorithmic confidence threshold.

    Returns:
        Tuple of (list_of_relationships, analysis_metadata)
    """
    if not clauses:
        return [], {"candidate_pairs": 0, "evaluated_pairs": 0, "rules_used": []}

    # Verify same-document isolation
    doc_id = document_id or clauses[0].document_id
    for c in clauses:
        if c.document_id != doc_id:
            raise ValueError(
                f"Document isolation violation: Clause '{c.clause_id}' belongs to document "
                f"'{c.document_id}', but expected document is '{doc_id}'."
            )

    engine = CrossClauseRelationshipEngine(min_confidence=min_confidence)
    candidates = engine.generate_candidate_pairs(clauses)

    detected: List[ClauseRelationship] = []
    seen_rel_keys: Set[Tuple[str, str, str]] = set()

    evaluated_count = 0
    for c_a, c_b, match_reasons, score in candidates:
        evaluated_count += 1
        rels = engine.evaluate_pair(c_a, c_b, doc_id)
        for r in rels:
            if r.confidence < (min_confidence or 0.50):
                continue
            key = (r.source_clause_id, r.target_clause_id, r.relationship_type)
            rev_key = (r.target_clause_id, r.source_clause_id, r.relationship_type)
            if key not in seen_rel_keys and rev_key not in seen_rel_keys:
                seen_rel_keys.add(key)
                detected.append(r)

    # Sort descending by confidence
    detected.sort(key=lambda r: r.confidence, reverse=True)

    if max_relationships:
        detected = detected[:max_relationships]

    metadata = {
        "candidate_pairs": len(candidates),
        "evaluated_pairs": evaluated_count,
        "rules_used": [
            "OVERRIDES",
            "EXCEPTS",
            "DEPENDS_ON",
            "QUALIFIES",
            "TEMPORAL",
            "POTENTIAL_TENSION",
            "SUPPORTS",
        ],
    }

    return detected, metadata
