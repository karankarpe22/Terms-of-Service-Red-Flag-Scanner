"""Hybrid clause classification module for Terms of Service documents.

Combines:
1. High-precision regex keyword/phrase patterns.
2. Dense semantic similarity using SentenceTransformers (all-MiniLM-L6-v2).

Supported PRD categories:
1. Privacy & Data
2. Payment
3. Account
4. Dispute
5. Liability
6. User Content
7. Changes

Non-negotiable guidelines:
- Do NOT train a traditional ML classifier.
- Category confidence is a model score, NOT a probability of legal correctness.
- Handles clauses that express the same concept using varied, non-keyword phrasing.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from backend.config import settings
from backend.core.constants import SUPPORTED_CATEGORIES
from backend.document.models import Clause
from backend.retrieval.embedder import get_embedding_service, EmbeddingService

# Explicit category seed descriptions for semantic grounding
CATEGORY_SEEDS: Dict[str, List[str]] = {
    "Privacy & Data": [
        "collection, gathering, and recording of personal user information and data",
        "sharing personal data, telemetry, and analytics with third party partners",
        "retention, storage, and deletion of user records and profile details",
        "cookies, tracking technologies, analytics, and privacy policy compliance",
        "processing personal data, behavioral monitoring, and data subjects",
    ],
    "Payment": [
        "subscription fees, billing schedules, and recurring payment charges",
        "automatic renewal of subscription and auto-renewing billing cycle",
        "non-refundable payments, refund policies, and transaction charges",
        "cancellation notice period, billing deadline, and payment terms",
        "credit card billing, invoicing, and unilateral price modifications",
        "monetary fees, financial consideration, and recurring payment cycles for access",
    ],
    "Account": [
        "user account registration, passwords, and security credentials",
        "immediate suspension or termination of user account and platform access",
        "account closure, cancellation of account, and permanent record deletion",
        "unauthorized access, compromised credentials, and account responsibility",
        "maintaining confidentiality of login details and user profile",
    ],
    "Dispute": [
        "mandatory binding arbitration and individual dispute resolution procedure",
        "class action lawsuit waiver and jury trial waiver agreement",
        "governing law, jurisdiction, venue, and legal dispute forum",
        "informal dispute negotiation, legal claims, and statute of limitations",
        "arbitration tribunal, dispute forum, and resolution of legal claims",
    ],
    "Liability": [
        "limitation of liability, damage cap, and maximum monetary liability",
        "disclaimer of warranties, as is basis, and as available service",
        "indemnification, defense, and hold harmless obligations by user",
        "exclusion of consequential, indirect, special, or punitive damages",
        "sole and exclusive remedy and limitation of recoverable damages",
    ],
    "User Content": [
        "ownership, copyright, and intellectual property rights in user content",
        "worldwide, perpetual, irrevocable, royalty-free license to use user content",
        "right to modify, adapt, publish, reproduce, and distribute user submissions",
        "user posted materials, submissions, and content infringement policy",
        "granting license to company to commercialize and display content",
    ],
    "Changes": [
        "unilateral modification and right to change these terms at any time",
        "updates, revisions, and amendments to terms without prior notice",
        "continued use of service constitutes binding acceptance of modified terms",
        "right to modify, alter, suspend, or discontinue services and features",
        "revisions and amendments effective immediately upon posting",
    ],
}

# High-precision regex keyword rules for deterministic signals
CATEGORY_KEYWORD_PATTERNS: Dict[str, List[re.Pattern]] = {
    "Privacy & Data": [
        re.compile(r"\b(personal\s+(data|information)|privacy\s+policy|cookies|track(ing)?)\b", re.I),
        re.compile(r"\b(third-part(y|ies)|data\s+retention|collect(ion|s)?\s+(of\s+)?(personal\s+)?data)\b", re.I),
        re.compile(r"\b(usage\s+(logs|data)|telemetry|geolocation|biometric|data\s+subject)\b", re.I),
    ],
    "Payment": [
        re.compile(r"\b(subscri(ption|be|ptions)|bill(ing)?|fee(s)?|auto-renew(al)?)\b", re.I),
        re.compile(r"\b(renew(s|ed|al)?\s+automatic(ally)?|non-refundable|refund(s)?|charg(e|es|ed|ing))\b", re.I),
        re.compile(r"\b(payment(s)?|credit\s+card|invoice|pricing|price\s+increase)\b", re.I),
    ],
    "Account": [
        re.compile(r"\b(account(s)?|password(s)?|login|credentials)\b", re.I),
        re.compile(r"\b(terminat(e|ion)\s+(of\s+)?(your\s+)?account|suspend\s+(your\s+)?account|suspension)\b", re.I),
        re.compile(r"\b(unauthorized\s+access|delet(e|ion)\s+(of\s+)?(your\s+)?account|register\s+an\s+account)\b", re.I),
    ],
    "Dispute": [
        re.compile(r"\b(arbitrat(e|ion|or|ors)|class\s+action\s+waiver|dispute(s)?)\b", re.I),
        re.compile(r"\b(governing\s+law|jurisdiction|venue|court(s)?|litigat(e|ion))\b", re.I),
        re.compile(r"\b(jury\s+trial\s+waiver|claim(s)?\s+shall\s+be\s+resolved|binding\s+individual\s+arbitration)\b", re.I),
    ],
    "Liability": [
        re.compile(r"\b(limit(ation)?\s+of\s+liability|indemnif(y|ication)|hold\s+harmless)\b", re.I),
        re.compile(r"\b(disclaim(er)?\s+of\s+warrant(y|ies)|consequential\s+damages|punitive\s+damages)\b", re.I),
        re.compile(r"\b(as-is|as\s+is|aggregate\s+liability|maximum\s+liability|sole\s+remedy)\b", re.I),
    ],
    "User Content": [
        re.compile(r"\b(user\s+content|user\s+submissions?|posted\s+content)\b", re.I),
        re.compile(r"\b(perpetual|irrevocable|royalty-free|license\s+to\s+(use|reproduce|modify|distribute))\b", re.I),
        re.compile(r"\b(grant\s+(us|the\s+company)\s+a\s+license|intellectual\s+property|copyright\s+infringement)\b", re.I),
    ],
    "Changes": [
        re.compile(r"\b(modify\s+(these\s+)?terms|change(s)?\s+to\s+(these\s+)?terms|unilateral(ly)?)\b", re.I),
        re.compile(r"\b(without\s+(prior\s+)?(individual\s+)?notice|at\s+any\s+time\s+without\s+notice)\b", re.I),
        re.compile(r"\b(continued\s+use\s+constitutes\s+acceptance|amend(ment|s)?|right\s+to\s+modify)\b", re.I),
    ],
}


class ClauseClassifier:
    """Hybrid clause classifier combining semantic embeddings and keyword rules."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedder = embedding_service or get_embedding_service()
        self.categories = list(SUPPORTED_CATEGORIES)
        self._seed_embeddings: Dict[str, np.ndarray] = {}
        self._precompute_seed_embeddings()

    def _precompute_seed_embeddings(self) -> None:
        """Pre-compute normalized vector embeddings for all category seeds once."""
        for category, seeds in CATEGORY_SEEDS.items():
            seed_vecs = self.embedder.encode(seeds)
            self._seed_embeddings[category] = seed_vecs

    def _score_category(
        self,
        clause_text: str,
        clause_embedding: np.ndarray,
        category: str,
    ) -> float:
        """Compute hybrid score for a single category."""
        # 1. Semantic similarity against category seeds
        seed_vecs = self._seed_embeddings[category]
        similarities = self.embedder.compute_similarity(clause_embedding, seed_vecs)
        # Take the top-1 maximum similarity score
        semantic_score = float(np.max(similarities))
        semantic_score = max(0.0, min(1.0, semantic_score))

        # 2. Keyword/phrase rule matching
        patterns = CATEGORY_KEYWORD_PATTERNS.get(category, [])
        match_count = 0
        for pattern in patterns:
            if pattern.search(clause_text):
                match_count += 1

        keyword_score = min(1.0, match_count * 0.35)

        # 3. Hybrid combination
        if match_count > 0:
            # When keywords match, combine weighted semantic + keyword
            hybrid_score = 0.55 * semantic_score + 0.45 * keyword_score
            if match_count >= 2:
                hybrid_score = min(1.0, hybrid_score + 0.08)
        else:
            # Pure semantic similarity when no exact keywords match
            hybrid_score = semantic_score

        return float(max(0.0, min(1.0, hybrid_score)))

    def classify_clause(self, clause: Clause) -> Dict[str, Any]:
        """Classify a single Clause object and populate its category fields."""
        results = self.classify_clauses([clause])
        return results[0]

    def classify_clauses(self, clauses: List[Clause]) -> List[Dict[str, Any]]:
        """Batch classify a list of clauses using batched embedding generation."""
        if not clauses:
            return []

        # Batched embedding generation
        texts = [c.text for c in clauses]
        clause_embeddings = self.embedder.encode(texts)

        confidence_threshold = settings.CATEGORY_CONFIDENCE_THRESHOLD
        secondary_threshold = settings.SECONDARY_CATEGORY_THRESHOLD

        output_records = []

        for idx, clause in enumerate(clauses):
            clause_emb = clause_embeddings[idx]
            clause_text = clause.text

            # Compute scores across all 7 supported categories
            category_scores: List[Tuple[str, float]] = []
            for cat in self.categories:
                score = self._score_category(clause_text, clause_emb, cat)
                category_scores.append((cat, score))

            # Sort descending by score
            category_scores.sort(key=lambda x: x[1], reverse=True)

            top_cat, top_score = category_scores[0]

            if top_score >= confidence_threshold:
                primary_cat = top_cat
                conf_score = round(top_score, 2)
                # Find secondary categories meeting secondary threshold and close to top score
                secondary_cats = [
                    cat for cat, s in category_scores[1:]
                    if s >= secondary_threshold and (top_score - s) <= 0.18
                ]
            else:
                primary_cat = "General"
                conf_score = round(top_score, 2)
                secondary_cats = []

            # Populate domain model
            clause.primary_category = primary_cat
            clause.category = primary_cat  # backward-compatible alias
            clause.secondary_categories = secondary_cats
            clause.category_confidence = conf_score

            output_records.append({
                "clause_id": clause.clause_id,
                "primary_category": primary_cat,
                "secondary_categories": secondary_cats,
                "category_confidence": conf_score,
            })

        return output_records


_CLASSIFIER_INSTANCE: Optional[ClauseClassifier] = None


def get_classifier() -> ClauseClassifier:
    """Return or initialize the singleton ClauseClassifier."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = ClauseClassifier()
    return _CLASSIFIER_INSTANCE


def classify_clause(clause: Clause) -> Dict[str, Any]:
    """Convenience function to classify a single clause."""
    return get_classifier().classify_clause(clause)


def classify_clauses(clauses: List[Clause]) -> List[Dict[str, Any]]:
    """Convenience function to batch-classify clauses."""
    return get_classifier().classify_clauses(clauses)
