"""Script to compile manual ground-truth annotations for Phase 9 Evaluation.

Builds:
- evaluation/data/ground_truth.json (74 labeled clauses, 24 relationship pairs)
- evaluation/data/retrieval_ground_truth.json (18 retrieval queries)
- evaluation/data/qa_ground_truth.json (24 Q&A items across Types A-F)

All labels are defined manually with human rationales.
Zero LLM generation.
"""
import json
from pathlib import Path
from backend.document.cleaner import clean_text
from backend.document.segmenter import segment_document
from backend.document.models import Document

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = BASE_DIR / "evaluation" / "data" / "documents"
DATA_DIR = BASE_DIR / "evaluation" / "data"

# Segment all 4 documents to obtain exact texts and IDs
docs_map = {}
for doc_name in ["doc_saas_cloud", "doc_ecommerce_marketplace", "doc_social_media", "doc_developer_api"]:
    raw = (DOCS_DIR / f"{doc_name}.txt").read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    doc = Document(document_id=doc_name, raw_text=raw, cleaned_text=cleaned, source_type="text")
    clauses = segment_document(doc)
    docs_map[doc_name] = clauses

# ---------------------------------------------------------------------------
# Manual Clause Ground Truth (74 clauses)
# ---------------------------------------------------------------------------
# Format: (doc_id, idx, primary_cat, secondary_cats, attention_level, rationale)
CLAUSE_SPECS = [
    # doc_saas_cloud (20 clauses)
    ("doc_saas_cloud", 1, "General", [], "Informational",
     "Introductory acceptance of terms clause stating binding nature of agreement upon platform use."),
    ("doc_saas_cloud", 2, "Account", [], "Informational",
     "Account registration requirement and user responsibility to keep login credentials confidential."),
    ("doc_saas_cloud", 3, "Account", [], "High Attention",
     "Allows company immediate account suspension/termination without notice or liability for any conduct."),
    ("doc_saas_cloud", 4, "Account", [], "High Attention",
     "Specifies immediate cessation of access and permanent irreversible deletion of stored user workspace files."),
    ("doc_saas_cloud", 5, "Payment", [], "High Attention",
     "Imposes recurring subscription billing with automatic renewal and a mandatory 30-day cancellation deadline."),
    ("doc_saas_cloud", 6, "Payment", [], "Medium Attention",
     "Expressly declares all paid subscription fees strictly non-refundable with no prorated refunds."),
    ("doc_saas_cloud", 7, "Changes", ["Payment"], "High Attention",
     "Reserves right to modify subscription fees and change pricing at any time without prior individual notice."),
    ("doc_saas_cloud", 8, "User Content", [], "High Attention",
     "Demands worldwide, perpetual, irrevocable, royalty-free, transferable and sublicensable license to user content."),
    ("doc_saas_cloud", 9, "User Content", [], "Informational",
     "User warranty that uploaded content does not infringe third-party copyrights or trademarks."),
    ("doc_saas_cloud", 10, "Changes", [], "High Attention",
     "Unilateral right to modify terms at any time without notice, with continued use deemed acceptance."),
    ("doc_saas_cloud", 11, "Changes", ["Payment"], "Medium Attention",
     "Commitment to endeavor to provide 30 days notice for material changes to pricing or customer data usage."),
    ("doc_saas_cloud", 12, "Privacy & Data", [], "Low Attention",
     "Standard collection of user data, device telemetry, and analytics under privacy policy."),
    ("doc_saas_cloud", 13, "Privacy & Data", [], "High Attention",
     "Authorizes sharing personal data and telemetry with third-party ad partners for cross-site tracking and profiling."),
    ("doc_saas_cloud", 14, "Privacy & Data", [], "High Attention",
     "Retains personal data and transaction records for up to 7 years following account closure."),
    ("doc_saas_cloud", 15, "Privacy & Data", [], "Low Attention",
     "Standard cookie notice for service functionality and workflow session maintenance."),
    ("doc_saas_cloud", 16, "Liability", [], "Low Attention",
     "Standard disclaimer of express, implied, and statutory warranties on an as-is basis."),
    ("doc_saas_cloud", 17, "Liability", [], "High Attention",
     "Excludes consequential damages and severely caps total aggregate company liability at $50.00."),
    ("doc_saas_cloud", 18, "Dispute", [], "Medium Attention",
     "Mandates binding individual arbitration under AAA rules in place of court proceedings."),
    ("doc_saas_cloud", 19, "Dispute", [], "High Attention",
     "Express waiver of right to participate in class action lawsuits, collective arbitration, or representative claims."),
    ("doc_saas_cloud", 20, "Dispute", [], "Informational",
     "Specifies Delaware governing law and exclusive venue in Wilmington state or federal courts."),

    # doc_ecommerce_marketplace (18 clauses)
    ("doc_ecommerce_marketplace", 1, "General", [], "Informational",
     "Introductory preamble establishing scope and legally binding nature of merchant terms."),
    ("doc_ecommerce_marketplace", 2, "Account", [], "Informational",
     "Merchant onboarding credential requirement and obligation to safeguard store login access."),
    ("doc_ecommerce_marketplace", 3, "Account", [], "High Attention",
     "Allows marketplace immediate account suspension without notice or liability for policy violations."),
    ("doc_ecommerce_marketplace", 4, "Account", [], "Informational",
     "Allows merchant to cancel store at any time via dashboard upon fulfilling orders."),
    ("doc_ecommerce_marketplace", 5, "Payment", [], "High Attention",
     "Automatic monthly subscription renewal and recurring deductions with a 15-day cancellation notice window."),
    ("doc_ecommerce_marketplace", 6, "Payment", [], "Medium Attention",
     "Declares all monthly subscription and transaction processing fees strictly non-refundable."),
    ("doc_ecommerce_marketplace", 7, "Changes", ["Payment"], "High Attention",
     "Unilateral right to change commission rates and transaction pricing at any time without notice."),
    ("doc_ecommerce_marketplace", 8, "User Content", [], "High Attention",
     "Grants worldwide, perpetual, royalty-free, transferable license to product media and descriptions."),
    ("doc_ecommerce_marketplace", 9, "Liability", [], "High Attention",
     "Obligates merchant to indemnify, defend, and hold harmless marketplace from third-party product claims."),
    ("doc_ecommerce_marketplace", 10, "Liability", [], "High Attention",
     "Excludes consequential damages and limits liability to fees paid in preceding three months."),
    ("doc_ecommerce_marketplace", 11, "Liability", [], "Low Attention",
     "Standard as-is marketplace and payment gateway warranty disclaimer."),
    ("doc_ecommerce_marketplace", 12, "Changes", [], "High Attention",
     "Unilateral terms modification at any time with continued use constituting acceptance."),
    ("doc_ecommerce_marketplace", 13, "Privacy & Data", [], "Low Attention",
     "Requirement for merchants to handle customer data according to privacy policies."),
    ("doc_ecommerce_marketplace", 14, "Privacy & Data", [], "High Attention",
     "Retains buyer transaction history and merchant logs for 7 years following account termination."),
    ("doc_ecommerce_marketplace", 15, "Dispute", [], "Medium Attention",
     "Mandates binding individual arbitration administered by JAMS for all disputes."),
    ("doc_ecommerce_marketplace", 16, "Dispute", [], "Medium Attention",
     "Explicit exception permitting small claims court and injunctive relief for IP claims."),
    ("doc_ecommerce_marketplace", 17, "Privacy & Data", [], "Low Attention",
     "Standard cookie and tracking notice for maintaining session states and checkout completion."),
    ("doc_ecommerce_marketplace", 18, "General", [], "Informational",
     "Standard boilerplate severability and entire agreement integration clause."),

    # doc_social_media (18 clauses)
    ("doc_social_media", 1, "General", [], "Informational",
     "Welcome preamble and agreement to be bound by mobile application terms."),
    ("doc_social_media", 2, "Account", [], "Informational",
     "User profile credential requirements and password security responsibility."),
    ("doc_social_media", 3, "Account", [], "Informational",
     "Permits user to deactivate or terminate account at any time via profile settings."),
    ("doc_social_media", 4, "Account", [], "High Attention",
     "Immediate account termination at company sole discretion without notice or liability for any reason."),
    ("doc_social_media", 5, "Payment", [], "High Attention",
     "Premium subscription auto-renews recurring charges monthly unless cancelled 48 hours prior."),
    ("doc_social_media", 6, "Payment", [], "Medium Attention",
     "Declares digital gifts, virtual badges, and premium membership fees non-refundable."),
    ("doc_social_media", 7, "User Content", [], "High Attention",
     "Demands worldwide, perpetual, irrevocable, royalty-free license to user posts, text, photos, and videos."),
    ("doc_social_media", 8, "User Content", [], "High Attention",
     "Sublicensable and transferable right to commercialize user submissions for promotional ads."),
    ("doc_social_media", 9, "User Content", [], "Informational",
     "DMCA copyright infringement notification and takedown policy description."),
    ("doc_social_media", 10, "Privacy & Data", [], "High Attention",
     "Shares personal data and browsing telemetry with ad networks for behavioral cross-site tracking."),
    ("doc_social_media", 11, "Privacy & Data", [], "High Attention",
     "Retains personal data and interaction logs indefinitely after account deletion."),
    ("doc_social_media", 12, "Changes", [], "High Attention",
     "Unilateral right to amend terms at any time with continued use deemed acceptance."),
    ("doc_social_media", 13, "Changes", [], "Low Attention",
     "Encourages users to periodically review terms for updates."),
    ("doc_social_media", 14, "Liability", [], "Low Attention",
     "As-is and as-available warranty disclaimer for community features."),
    ("doc_social_media", 15, "Liability", [], "Medium Attention",
     "Excludes consequential, special, and punitive damages arising from platform use."),
    ("doc_social_media", 16, "Dispute", [], "High Attention",
     "Waives right to participate in class action lawsuits or representative proceedings."),
    ("doc_social_media", 17, "Dispute", [], "Informational",
     "Governing California law and exclusive venue in San Francisco state/federal courts."),
    ("doc_social_media", 18, "General", [], "Informational",
     "Standard headings clause stating section titles have no contractual significance."),

    # doc_developer_api (18 clauses)
    ("doc_developer_api", 1, "General", [], "Informational",
     "Preamble defining scope and binding agreement for programmatic API access."),
    ("doc_developer_api", 2, "Account", [], "Informational",
     "API key confidentiality requirements and responsibility for data egress calls."),
    ("doc_developer_api", 3, "Account", [], "High Attention",
     "Immediate API access revocation without notice for rate limit abuse or security issues."),
    ("doc_developer_api", 4, "Account", [], "Informational",
     "Permits developer account deletion once compute and bandwidth invoices are settled."),
    ("doc_developer_api", 5, "Payment", [], "High Attention",
     "Auto-renewal of usage tiers with recurring monthly overage fees billed directly."),
    ("doc_developer_api", 6, "Payment", [], "Medium Attention",
     "Prepaid API request credits are non-refundable and expire after 12 months."),
    ("doc_developer_api", 7, "Changes", ["Payment"], "High Attention",
     "Unilateral right to modify pricing structure and increase request rates without notice."),
    ("doc_developer_api", 8, "User Content", [], "Informational",
     "Developer retains IP in built applications, subject to processing license for payloads."),
    ("doc_developer_api", 9, "User Content", [], "Informational",
     "Company retains proprietary rights, title, and interest in API endpoints and schemas."),
    ("doc_developer_api", 10, "Privacy & Data", [], "Low Attention",
     "Collection of developer personal data, IP addresses, and request latency telemetry."),
    ("doc_developer_api", 11, "Privacy & Data", [], "High Attention",
     "Retains API usage logs and developer telemetry records for up to 7 years."),
    ("doc_developer_api", 12, "Changes", [], "High Attention",
     "Unilateral right to modify or deprecate endpoints at any time with continued use as acceptance."),
    ("doc_developer_api", 13, "General", [], "High Attention",
     "Precedence clause: signed Enterprise Master Agreement supersedes and controls over standard terms."),
    ("doc_developer_api", 14, "Liability", [], "Low Attention",
     "As-is service disclaimer without warranties of uptime or API performance."),
    ("doc_developer_api", 15, "Liability", [], "High Attention",
     "Caps aggregate liability for all claims at one hundred US dollars ($100.00)."),
    ("doc_developer_api", 16, "Liability", [], "High Attention",
     "Obligates developer to indemnify and hold harmless company against third-party claims."),
    ("doc_developer_api", 17, "Dispute", [], "Medium Attention",
     "Mandates binding arbitration in Delaware for all legal claims."),
    ("doc_developer_api", 18, "General", [], "Informational",
     "Standard severability boilerplate clause preserving validity of surviving provisions."),
]

ground_truth_clauses = []
for doc_id, idx, pri_cat, sec_cats, att_lvl, rationale in CLAUSE_SPECS:
    clauses = docs_map[doc_id]
    c = clauses[idx - 1]
    ground_truth_clauses.append({
        "document_id": doc_id,
        "clause_id": c.clause_id,
        "clause_index": c.clause_index,
        "section_title": c.section_title,
        "text": c.text,
        "expected_primary_category": pri_cat,
        "expected_secondary_categories": sec_cats,
        "expected_attention_level": att_lvl,
        "source": f"{doc_id}.txt - Section {idx}",
        "rationale": rationale,
    })

# ---------------------------------------------------------------------------
# Manual Cross-Clause Relationship Ground Truth (24 relationship pairs)
# ---------------------------------------------------------------------------
RELATIONSHIP_SPECS = [
    # doc_saas_cloud relationships
    ("doc_saas_cloud", "clause_010", "clause_011", "POTENTIAL_TENSION",
     "Section 10 claims terms may be modified at any time without notice, whereas Section 11 commits to 30 days notice for material changes."),
    ("doc_saas_cloud", "clause_003", "clause_004", "TEMPORAL",
     "Account suspension/termination in Section 3 is immediately followed by irreversible workspace data deletion in Section 4."),
    ("doc_saas_cloud", "clause_005", "clause_006", "SUPPORTS",
     "Non-refundable payment rule in Section 6 reinforces and supports automatic renewal billing in Section 5."),
    ("doc_saas_cloud", "clause_008", "clause_009", "QUALIFIES",
     "License grant in Section 8 is qualified by user representation in Section 9 that user holds required rights."),
    ("doc_saas_cloud", "clause_016", "clause_017", "SUPPORTS",
     "Warranty disclaimer in Section 16 supports and aligns with total damage exclusion and $50 liability cap in Section 17."),
    ("doc_saas_cloud", "clause_018", "clause_019", "SUPPORTS",
     "Arbitration clause in Section 18 is supported by class action waiver in Section 19 restricting dispute resolution to individual forum."),
    ("doc_saas_cloud", "clause_003", "clause_014", "TEMPORAL",
     "Upon account termination in Section 3, Section 14 mandates retention of personal records for up to 7 years."),

    # doc_ecommerce_marketplace relationships
    ("doc_ecommerce_marketplace", "clause_003", "clause_004", "POTENTIAL_TENSION",
     "Section 4 allows merchant to cancel at any time, but Section 3 reserves company right to immediately suspend without notice or liability."),
    ("doc_ecommerce_marketplace", "clause_015", "clause_016", "EXCEPTS",
     "Section 16 explicitly carves out small claims court and IP injunctive relief from the mandatory arbitration rule in Section 15."),
    ("doc_ecommerce_marketplace", "clause_005", "clause_006", "SUPPORTS",
     "Recurring monthly subscription in Section 5 is supported by Section 6 declaring all fees non-refundable."),
    ("doc_ecommerce_marketplace", "clause_008", "clause_009", "DEPENDS_ON",
     "Product content license in Section 8 depends on merchant indemnification in Section 9 protecting platform from infringement claims."),
    ("doc_ecommerce_marketplace", "clause_010", "clause_011", "SUPPORTS",
     "Section 10 liability limitation operates alongside and supports Section 11 as-is warranty disclaimer."),
    ("doc_ecommerce_marketplace", "clause_003", "clause_014", "TEMPORAL",
     "Account termination in Section 3 triggers 7-year data retention requirement in Section 14."),

    # doc_social_media relationships
    ("doc_social_media", "clause_003", "clause_004", "POTENTIAL_TENSION",
     "Section 3 allows user deactivation at any time, whereas Section 4 gives platform unconditional immediate termination rights."),
    ("doc_social_media", "clause_007", "clause_008", "SUPPORTS",
     "Perpetual user content license in Section 7 is expanded and supported by Section 8 sublicensing and commercialization rights."),
    ("doc_social_media", "clause_005", "clause_006", "SUPPORTS",
     "Recurring subscription billing in Section 5 is supported by Section 6 declaring virtual purchases non-refundable."),
    ("doc_social_media", "clause_004", "clause_011", "TEMPORAL",
     "Profile termination in Section 4 is followed by indefinite retention of personal data and interaction logs in Section 11."),
    ("doc_social_media", "clause_012", "clause_013", "QUALIFIES",
     "Section 13 advice to periodically review terms qualifies unilateral modification rule in Section 12."),
    ("doc_social_media", "clause_014", "clause_015", "SUPPORTS",
     "As-is service disclaimer in Section 14 supports consequential damages exclusion in Section 15."),

    # doc_developer_api relationships
    ("doc_developer_api", "clause_012", "clause_013", "OVERRIDES",
     "Section 13 explicitly dictates that in case of conflict, an executed Enterprise Master Agreement shall control and take precedence over Section 12."),
    ("doc_developer_api", "clause_003", "clause_004", "POTENTIAL_TENSION",
     "Section 4 allows voluntary deletion after invoice settlement, but Section 3 permits immediate revocation without notice."),
    ("doc_developer_api", "clause_005", "clause_006", "SUPPORTS",
     "Auto-renewing usage tiers in Section 5 are supported by Section 6 stating API request credits are non-refundable."),
    ("doc_developer_api", "clause_008", "clause_009", "QUALIFIES",
     "Developer ownership of built applications in Section 8 is qualified by Section 9 reserving all proprietary rights in endpoints to company."),
    ("doc_developer_api", "clause_003", "clause_011", "TEMPORAL",
     "API key revocation in Section 3 triggers 7-year telemetry retention in Section 11."),
]

ground_truth_relationships = []
for doc_id, src_suffix, tgt_suffix, rel_type, rationale in RELATIONSHIP_SPECS:
    src_id = f"{doc_id}_{src_suffix}"
    tgt_id = f"{doc_id}_{tgt_suffix}"
    ground_truth_relationships.append({
        "document_id": doc_id,
        "source_clause_id": src_id,
        "target_clause_id": tgt_id,
        "expected_relationship_type": rel_type,
        "source": f"{doc_id}.txt - {src_suffix} & {tgt_suffix}",
        "rationale": rationale,
    })

# ---------------------------------------------------------------------------
# Manual Retrieval Ground Truth (18 queries)
# ---------------------------------------------------------------------------
RETRIEVAL_SPECS = [
    # doc_saas_cloud
    {
        "query": "Will my subscription renew automatically?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_005"],
        "rationale": "Query directly targets auto-renewal clause Section 5.",
    },
    {
        "query": "Can I get a refund if I cancel my subscription early?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_006"],
        "rationale": "Query targets the non-refundable payment policy in Section 6.",
    },
    {
        "query": "How much advance notice is required to cancel my subscription?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_005"],
        "rationale": "Query targets Section 5 notice period (at least 30 days prior).",
    },
    {
        "query": "Does the company share my personal information with advertisers?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_013"],
        "rationale": "Query targets Section 13 sharing data with ad partners for cross-site tracking.",
    },
    {
        "query": "How long is my data stored after closing my account?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_014"],
        "rationale": "Query targets Section 14 7-year data retention post-termination.",
    },
    {
        "query": "What is the maximum monetary liability of the company?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_017"],
        "rationale": "Query targets Section 17 $50 aggregate liability damage cap.",
    },
    {
        "query": "Can I sue the company in a class action lawsuit?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_019"],
        "rationale": "Query targets Section 19 class action waiver.",
    },
    {
        "query": "Does the company have the right to modify prices without notice?",
        "document_id": "doc_saas_cloud",
        "relevant_clause_ids": ["doc_saas_cloud_clause_007", "doc_saas_cloud_clause_010"],
        "rationale": "Query relates to unilateral price modifications in Section 7 and terms modifications in Section 10.",
    },

    # doc_ecommerce_marketplace
    {
        "query": "What are the rules regarding marketplace commission fee changes?",
        "document_id": "doc_ecommerce_marketplace",
        "relevant_clause_ids": ["doc_ecommerce_marketplace_clause_007"],
        "rationale": "Query targets Section 7 unilateral commission fee changes.",
    },
    {
        "query": "Can disputes be resolved in small claims court instead of arbitration?",
        "document_id": "doc_ecommerce_marketplace",
        "relevant_clause_ids": ["doc_ecommerce_marketplace_clause_015", "doc_ecommerce_marketplace_clause_016"],
        "rationale": "Query targets Section 15 arbitration and Section 16 small claims exception.",
    },
    {
        "query": "Am I required to indemnify the marketplace for copyright claims?",
        "document_id": "doc_ecommerce_marketplace",
        "relevant_clause_ids": ["doc_ecommerce_marketplace_clause_009"],
        "rationale": "Query targets Section 9 merchant indemnification clause.",
    },
    {
        "query": "How many days notice must I give to cancel my store subscription?",
        "document_id": "doc_ecommerce_marketplace",
        "relevant_clause_ids": ["doc_ecommerce_marketplace_clause_005"],
        "rationale": "Query targets Section 5 15-day cancellation notice requirement.",
    },

    # doc_social_media
    {
        "query": "Does the app grant a perpetual license to use my uploaded photos?",
        "document_id": "doc_social_media",
        "relevant_clause_ids": ["doc_social_media_clause_007", "doc_social_media_clause_008"],
        "rationale": "Query targets Section 7 perpetual license and Section 8 sublicensing.",
    },
    {
        "query": "Can the platform terminate my account without any prior notice?",
        "document_id": "doc_social_media",
        "relevant_clause_ids": ["doc_social_media_clause_004"],
        "rationale": "Query targets Section 4 immediate termination without prior notice.",
    },
    {
        "query": "Are digital gifts and virtual badges refundable if I change my mind?",
        "document_id": "doc_social_media",
        "relevant_clause_ids": ["doc_social_media_clause_006"],
        "rationale": "Query targets Section 6 non-refundable digital goods policy.",
    },

    # doc_developer_api
    {
        "query": "What happens if these API terms conflict with a signed enterprise contract?",
        "document_id": "doc_developer_api",
        "relevant_clause_ids": ["doc_developer_api_clause_013"],
        "rationale": "Query targets Section 13 enterprise agreement precedence override.",
    },
    {
        "query": "What is the maximum damages cap under these API terms?",
        "document_id": "doc_developer_api",
        "relevant_clause_ids": ["doc_developer_api_clause_015"],
        "rationale": "Query targets Section 15 $100 damage limitation cap.",
    },
    {
        "query": "Can DataForge deprecate API endpoints without advance notice?",
        "document_id": "doc_developer_api",
        "relevant_clause_ids": ["doc_developer_api_clause_012"],
        "rationale": "Query targets Section 12 unilateral API modifications and deprecation.",
    },
]

# ---------------------------------------------------------------------------
# Manual Q&A Ground Truth (24 items covering Types A to F)
# ---------------------------------------------------------------------------
QA_SPECS = [
    # TYPE A: Direct Answerable Questions (single clause)
    {
        "question_id": "qa_001",
        "document_id": "doc_saas_cloud",
        "question": "Will my subscription renew automatically?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_005"],
        "required_facts": ["automatic renewal", "recurring monthly or annual basis"],
        "forbidden_claims": ["subscription is free", "no auto-renewal"],
        "category": "Payment",
        "ground_truth_notes": "Section 5 explicitly specifies auto-renewal at end of billing cycle.",
    },
    {
        "question_id": "qa_002",
        "document_id": "doc_saas_cloud",
        "question": "Can I obtain a refund if I terminate my account before the end of the month?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_006"],
        "required_facts": ["non-refundable", "no credits or prorated refunds"],
        "forbidden_claims": ["refunds are guaranteed within 30 days"],
        "category": "Payment",
        "ground_truth_notes": "Section 6 clearly states subscription fees are strictly non-refundable.",
    },
    {
        "question_id": "qa_003",
        "document_id": "doc_ecommerce_marketplace",
        "question": "How much advance notice must a merchant provide to prevent subscription renewal?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_ecommerce_marketplace_clause_005"],
        "required_facts": ["15 days", "prior to renewal"],
        "forbidden_claims": ["cancel anytime without notice"],
        "category": "Payment",
        "ground_truth_notes": "Section 5 requires cancellation at least 15 days prior to renewal.",
    },
    {
        "question_id": "qa_004",
        "document_id": "doc_developer_api",
        "question": "What is the maximum monetary liability cap under the API terms?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_developer_api_clause_015"],
        "required_facts": ["$100", "aggregate liability shall not exceed"],
        "forbidden_claims": ["unlimited liability", "$50 liability cap"],
        "category": "Liability",
        "ground_truth_notes": "Section 15 sets maximum aggregate liability at $100.00.",
    },

    # TYPE B: Multi-Clause Questions
    {
        "question_id": "qa_005",
        "document_id": "doc_saas_cloud",
        "question": "What happens to my stored workspace files and personal data after account termination?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_004", "doc_saas_cloud_clause_014"],
        "required_facts": ["permanently deleted", "retained for up to seven years", "personal data"],
        "forbidden_claims": ["all data is returned on USB drive"],
        "category": "Account / Privacy",
        "ground_truth_notes": "Section 4 dictates workspace files deleted; Section 14 dictates personal data retained up to 7 years.",
    },
    {
        "question_id": "qa_006",
        "document_id": "doc_social_media",
        "question": "What rights does PulseMedia have regarding user submitted posts and photos?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_social_media_clause_007", "doc_social_media_clause_008"],
        "required_facts": ["perpetual", "royalty-free", "sublicensable", "commercialize"],
        "forbidden_claims": ["user retains exclusive commercial rights"],
        "category": "User Content",
        "ground_truth_notes": "Sections 7 and 8 grant worldwide perpetual license with sublicensing and commercialization rights.",
    },
    {
        "question_id": "qa_007",
        "document_id": "doc_saas_cloud",
        "question": "How are legal disputes resolved and can I join a class action lawsuit?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_018", "doc_saas_cloud_clause_019"],
        "required_facts": ["binding individual arbitration", "waive any right to participate in a class action"],
        "forbidden_claims": ["class actions are permitted in Delaware"],
        "category": "Dispute",
        "ground_truth_notes": "Section 18 mandates arbitration and Section 19 expressly waives class actions.",
    },

    # TYPE C: Relationship Questions
    {
        "question_id": "qa_008",
        "document_id": "doc_saas_cloud",
        "question": "Do the terms contain conflicting statements regarding advance notice for changing terms?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_010", "doc_saas_cloud_clause_011"],
        "required_facts": ["without prior individual notice", "30 days advance notice for material changes"],
        "forbidden_claims": ["there is no notice provision at all"],
        "category": "Changes",
        "ground_truth_notes": "Potential tension between Section 10 (modify at any time without notice) and Section 11 (30 days notice for material changes).",
    },
    {
        "question_id": "qa_009",
        "document_id": "doc_ecommerce_marketplace",
        "question": "Can a merchant take a dispute to court notwithstanding the arbitration provision?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_ecommerce_marketplace_clause_015", "doc_ecommerce_marketplace_clause_016"],
        "required_facts": ["small claims court", "intellectual property infringement", "exception"],
        "forbidden_claims": ["arbitration is mandatory without any exceptions"],
        "category": "Dispute",
        "ground_truth_notes": "Section 16 carves out explicit exception for small claims court and IP claims from Section 15.",
    },
    {
        "question_id": "qa_010",
        "document_id": "doc_developer_api",
        "question": "Which document governs if these standard API terms conflict with a signed Enterprise agreement?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_developer_api_clause_013"],
        "required_facts": ["Enterprise Master Services Agreement", "control and take precedence"],
        "forbidden_claims": ["standard terms always override enterprise agreements"],
        "category": "General",
        "ground_truth_notes": "Section 13 explicitly provides that the Enterprise Agreement controls and supersedes standard terms.",
    },

    # TYPE D: Verbatim Source Evidence Questions
    {
        "question_id": "qa_011",
        "document_id": "doc_saas_cloud",
        "question": "What is the exact maximum dollar damage cap stated in the limitation of liability section?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_017"],
        "required_facts": ["fifty US dollars ($50.00)", "aggregate liability"],
        "forbidden_claims": ["one hundred dollars", "no dollar cap"],
        "category": "Liability",
        "ground_truth_notes": "Section 17 explicitly states aggregate liability shall not exceed fifty US dollars ($50.00).",
    },
    {
        "question_id": "qa_012",
        "document_id": "doc_social_media",
        "question": "How many hours before billing must I cancel my premium social media membership?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_social_media_clause_005"],
        "required_facts": ["48 hours", "prior to the next billing date"],
        "forbidden_claims": ["30 days", "cancel anytime with no notice"],
        "category": "Payment",
        "ground_truth_notes": "Section 5 specifies cancellation at least 48 hours prior to next billing date.",
    },

    # TYPE E: Unsupported Questions (Must Abstain - status=insufficient_evidence)
    {
        "question_id": "qa_013",
        "document_id": "doc_saas_cloud",
        "question": "Does the company sell my biometric facial recognition data to foreign governments?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["company sells biometric data", "facial recognition is permitted"],
        "category": "Unsupported",
        "ground_truth_notes": "The document contains no mention of biometric or facial recognition data. System must abstain.",
    },
    {
        "question_id": "qa_014",
        "document_id": "doc_saas_cloud",
        "question": "What is the phone number and mailing address for the customer support team in London?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["London office address", "phone number"],
        "category": "Unsupported",
        "ground_truth_notes": "No London address or phone contact is provided. System must abstain.",
    },
    {
        "question_id": "qa_015",
        "document_id": "doc_ecommerce_marketplace",
        "question": "Does MarketGrid offer free shipping insurance on all international orders over $500?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["free shipping insurance is provided", "coverage up to $500"],
        "category": "Unsupported",
        "ground_truth_notes": "No shipping insurance provisions exist. System must abstain.",
    },
    {
        "question_id": "qa_016",
        "document_id": "doc_social_media",
        "question": "Does PulseMedia pay creators royalties based on view counts on viral videos?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["creators are paid royalties per view"],
        "category": "Unsupported",
        "ground_truth_notes": "The terms specify royalty-free user content license, no creator revenue share is defined. System must abstain.",
    },
    {
        "question_id": "qa_017",
        "document_id": "doc_developer_api",
        "question": "What is the guaranteed service level agreement (SLA) percentage for API endpoint uptime?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["99.9% uptime SLA", "99.99% SLA"],
        "category": "Unsupported",
        "ground_truth_notes": "Section 14 explicitly provides as-is service without uptime warranties; no SLA percentage is promised. System must abstain.",
    },
    {
        "question_id": "qa_018",
        "document_id": "doc_developer_api",
        "question": "Which health insurance coverage is provided to developers using the platform?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["health insurance plans"],
        "category": "Unsupported",
        "ground_truth_notes": "Completely irrelevant to API terms. System must abstain.",
    },

    # TYPE F: Legal-Conclusion & Adversarial Framing Questions
    {
        "question_id": "qa_019",
        "document_id": "doc_saas_cloud",
        "question": "The Terms say the company guarantees a 90-day no-questions-asked cash refund. Explain how to claim it.",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["company guarantees a 90-day refund", "cash refund procedure"],
        "category": "Adversarial Hallucination Trap",
        "ground_truth_notes": "Adversarial premise. Section 6 explicitly states fees are strictly non-refundable. System must refuse false premise / abstain.",
    },
    {
        "question_id": "qa_020",
        "document_id": "doc_saas_cloud",
        "question": "Is the $50 liability damage cap illegal under federal consumer protection laws?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_017"],
        "required_facts": ["$50.00", "non-legal advice disclaimer"],
        "forbidden_claims": ["is illegal", "is legally invalid", "violates federal law"],
        "category": "Legal Advice Trap",
        "ground_truth_notes": "System must describe the clause text neutrally and include legal disclaimer without declaring legality/illegality.",
    },
    {
        "question_id": "qa_021",
        "document_id": "doc_saas_cloud",
        "question": "Give me legal advice on whether I can successfully sue ApexCloud despite the arbitration clause.",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_saas_cloud_clause_018"],
        "required_facts": ["binding individual arbitration", "disclaimer"],
        "forbidden_claims": ["you can definitely sue", "this is legal advice", "clause is void"],
        "category": "Legal Advice Request",
        "ground_truth_notes": "System must refuse legal advice, cite Section 18, and provide standard disclaimer.",
    },
    {
        "question_id": "qa_022",
        "document_id": "doc_social_media",
        "question": "Which section says PulseMedia sells biometric data to advertisers?",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["biometric data is sold", "Section 10 sells biometric data"],
        "category": "Adversarial Hallucination Trap",
        "ground_truth_notes": "Section 10 mentions browsing telemetry, not biometric data. System must abstain or refuse false premise.",
    },
    {
        "question_id": "qa_023",
        "document_id": "doc_developer_api",
        "question": "Is the unilateral modification of API endpoints legally void and unenforceable?",
        "expected_answerable": True,
        "expected_source_clause_ids": ["doc_developer_api_clause_012"],
        "required_facts": ["Section 12", "modify, alter, or deprecate"],
        "forbidden_claims": ["is legally void", "is unenforceable", "violates contract law"],
        "category": "Legal Advice Trap",
        "ground_truth_notes": "System must report what the clause says without certifying legal enforceability.",
    },
    {
        "question_id": "qa_024",
        "document_id": "doc_ecommerce_marketplace",
        "question": "Explain the 180-day money-back guarantee mentioned in Section 4.",
        "expected_answerable": False,
        "expected_source_clause_ids": [],
        "required_facts": [],
        "forbidden_claims": ["180-day money-back guarantee exists in Section 4"],
        "category": "Adversarial Hallucination Trap",
        "ground_truth_notes": "Section 4 contains no money-back guarantee; Section 6 states fees are non-refundable. System must reject false premise.",
    },
]

# Write ground_truth.json
gt_payload = {
    "metadata": {
        "dataset_name": "Terms of Service Academic Evaluation Dataset",
        "description": "Manually labeled dataset across 4 diverse service agreements for system benchmarking.",
        "version": "1.0",
        "total_documents": 4,
        "total_clauses": len(ground_truth_clauses),
        "total_relationships": len(ground_truth_relationships),
        "categories": [
            "Privacy & Data",
            "Payment",
            "Account",
            "Dispute",
            "Liability",
            "User Content",
            "Changes",
            "General",
        ],
        "attention_levels": [
            "High Attention",
            "Medium Attention",
            "Low Attention",
            "Informational",
        ],
        "relationship_types": [
            "SUPPORTS",
            "QUALIFIES",
            "EXCEPTS",
            "OVERRIDES",
            "DEPENDS_ON",
            "TEMPORAL",
            "POTENTIAL_TENSION",
        ],
    },
    "clauses": ground_truth_clauses,
    "relationships": ground_truth_relationships,
}

(DATA_DIR / "ground_truth.json").write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")
print(f"Wrote ground_truth.json: {len(ground_truth_clauses)} clauses, {len(ground_truth_relationships)} relationships.")

# Write retrieval_ground_truth.json
retrieval_payload = {
    "metadata": {
        "description": "Ground truth queries for FAISS semantic retrieval evaluation across multiple similarity thresholds.",
        "query_count": len(RETRIEVAL_SPECS),
    },
    "queries": RETRIEVAL_SPECS,
}
(DATA_DIR / "retrieval_ground_truth.json").write_text(json.dumps(retrieval_payload, indent=2), encoding="utf-8")
print(f"Wrote retrieval_ground_truth.json: {len(RETRIEVAL_SPECS)} queries.")

# Write qa_ground_truth.json
qa_payload = {
    "metadata": {
        "description": "Ground truth questions covering factual Q&A, multi-clause synthesis, relationships, abstention, and adversarial grounding.",
        "question_count": len(QA_SPECS),
    },
    "questions": QA_SPECS,
}
(DATA_DIR / "qa_ground_truth.json").write_text(json.dumps(qa_payload, indent=2), encoding="utf-8")
print(f"Wrote qa_ground_truth.json: {len(QA_SPECS)} questions.")
