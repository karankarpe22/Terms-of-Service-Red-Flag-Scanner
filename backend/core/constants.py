"""Domain constants for categories, attention levels, relationship types, and system disclaimers.

Non-negotiable rule:
- This is an informational document-analysis system, not a legal-advice system.
- Never state that a clause is illegal, invalid, enforceable, or compliant.
- Use 'attention level', not 'legal risk score'.
"""

# Supported PRD Categories (PRD Section 5)
SUPPORTED_CATEGORIES = [
    "Privacy & Data",
    "Payment",
    "Account",
    "Dispute",
    "Liability",
    "User Content",
    "Changes",
]

# Attention Levels (PRD Section 6)
ATTENTION_LEVEL_HIGH = "High Attention"
ATTENTION_LEVEL_MEDIUM = "Medium Attention"
ATTENTION_LEVEL_LOW = "Low Attention"
ATTENTION_LEVEL_INFORMATIONAL = "Informational"

ATTENTION_LEVELS = [
    ATTENTION_LEVEL_HIGH,
    ATTENTION_LEVEL_MEDIUM,
    ATTENTION_LEVEL_LOW,
    ATTENTION_LEVEL_INFORMATIONAL,
]

# Canonical Phase 5 Relationship Types
RELATIONSHIP_TYPE_SUPPORTS = "SUPPORTS"
RELATIONSHIP_TYPE_QUALIFIES = "QUALIFIES"
RELATIONSHIP_TYPE_EXCEPTS = "EXCEPTS"
RELATIONSHIP_TYPE_OVERRIDES = "OVERRIDES"
RELATIONSHIP_TYPE_DEPENDS_ON = "DEPENDS_ON"
RELATIONSHIP_TYPE_TEMPORAL = "TEMPORAL"
RELATIONSHIP_TYPE_POTENTIAL_TENSION = "POTENTIAL_TENSION"

RELATIONSHIP_TYPES = [
    RELATIONSHIP_TYPE_SUPPORTS,
    RELATIONSHIP_TYPE_QUALIFIES,
    RELATIONSHIP_TYPE_EXCEPTS,
    RELATIONSHIP_TYPE_OVERRIDES,
    RELATIONSHIP_TYPE_DEPENDS_ON,
    RELATIONSHIP_TYPE_TEMPORAL,
    RELATIONSHIP_TYPE_POTENTIAL_TENSION,
    # Backward-compatible aliases
    "conditional_dependency",
    "combined_effect",
    "exception",
    "reference/dependency",
    "potentially_conflicting_condition",
]

# High-value category pairings for candidate generation (Phase 5 Section 9)
CATEGORY_RELATIONSHIP_MATRIX = {
    "Payment": {"Account", "Changes"},
    "Account": {"Payment", "Privacy & Data", "User Content", "Dispute", "Changes"},
    "Privacy & Data": {"Account", "User Content"},
    "User Content": {"Account", "Privacy & Data", "Liability"},
    "Dispute": {"Liability", "Account"},
    "Liability": {"Dispute", "User Content"},
    "Changes": {"Payment", "Account", "Privacy & Data", "User Content", "Dispute", "Liability"},
}

# Non-legal Disclaimer (PRD FR-17 & Rules)
SYSTEM_DISCLAIMER = (
    "This tool is an informational document-analysis system, not a legal-advice service. "
    "It does not provide legal advice, compliance certification, or enforceability determinations. "
    "All findings represent attention flags to guide user review against original contractual text."
)
