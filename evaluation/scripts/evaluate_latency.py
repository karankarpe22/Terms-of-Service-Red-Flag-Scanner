"""Evaluation script for End-to-End Pipeline Latency Benchmarking.

Collects multi-run timing observations across:
1. Document Ingestion & Text Cleaning
2. Clause Segmentation
3. Hybrid Clause Classification
4. FAISS Embedding & Indexing
5. Semantic Retrieval (Top-K Search)
6. Cross-Clause Relationship Analysis
7. Evidence-Grounded LLM Explanation
8. End-to-End Q&A Pipeline

Calculates:
- Mean latency (ms)
- Median latency (ms)
- Minimum latency (ms)
- Maximum latency (ms)
- P95 latency (ms)
- Detailed hardware and environment specifications
"""
import os
import sys
import time
import platform
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.document.cleaner import clean_text
from backend.document.segmenter import segment_document
from backend.document.models import Document
from backend.document.store import save_document
from backend.analysis.classifier import get_classifier
from backend.analysis.attention import assign_attention_levels
from backend.analysis.relationships import detect_relationships
from backend.retrieval.vector_store import get_vector_store
from backend.llm.client import get_llm_service
from backend.evaluation.evaluator import compute_latency_stats


def get_environment_info() -> Dict[str, Any]:
    """Capture environment and hardware context without external psutil dependency."""
    ram_gb = "Unknown"
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass

    cpu_count = os.cpu_count() or 4

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical_cores": cpu_count,
        "ram_gb": ram_gb,
        "environment_type": "Local Development (Windows Workstation)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_engine": "faiss-cpu",
    }


def run_latency_benchmarks(num_runs: int = 5) -> Dict[str, Any]:
    """Execute multi-iteration latency benchmarks across all pipeline components."""
    docs_dir = BASE_DIR / "evaluation" / "data" / "documents"
    sample_doc_path = docs_dir / "doc_saas_cloud.txt"
    raw_text = sample_doc_path.read_text(encoding="utf-8")

    import backend.llm.client as client_mod
    original_mock = settings.GEMINI_MOCK_MODE
    settings.GEMINI_MOCK_MODE = True
    client_mod._LLM_SERVICE_INSTANCE = None

    try:
        classifier = get_classifier()
        vector_store = get_vector_store()
        llm_service = get_llm_service()

        benchmarks: Dict[str, List[float]] = {
            "ingestion_cleaning": [],
            "clause_segmentation": [],
            "clause_classification": [],
            "faiss_indexing": [],
            "retrieval_search": [],
            "relationship_analysis": [],
            "llm_explanation": [],
            "end_to_end_qa": [],
        }

        # Warmup runs
        cleaned = clean_text(raw_text)
        doc_warm = Document(document_id="doc_warmup", raw_text=raw_text, cleaned_text=cleaned, source_type="text")
        clauses_warm = segment_document(doc_warm)
        classifier.classify_clauses(clauses_warm)
        vector_store.index_document("doc_warmup", clauses_warm)

        for run_idx in range(num_runs):
            # 1. Ingestion & Text Cleaning
            t0 = time.perf_counter()
            cleaned_text = clean_text(raw_text)
            t1 = time.perf_counter()
            benchmarks["ingestion_cleaning"].append((t1 - t0) * 1000)

            # 2. Clause Segmentation
            doc_bench = Document(
                document_id=f"doc_bench_{run_idx}",
                filename="doc_saas_cloud.txt",
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                source_type="text",
            )
            t0 = time.perf_counter()
            clauses = segment_document(doc_bench)
            t1 = time.perf_counter()
            benchmarks["clause_segmentation"].append((t1 - t0) * 1000)

            # 3. Clause Classification
            t0 = time.perf_counter()
            classifier.classify_clauses(clauses)
            assign_attention_levels(clauses)
            t1 = time.perf_counter()
            benchmarks["clause_classification"].append((t1 - t0) * 1000)

            # 4. FAISS Embedding & Indexing
            t0 = time.perf_counter()
            vector_store.index_document(doc_bench.document_id, clauses)
            t1 = time.perf_counter()
            benchmarks["faiss_indexing"].append((t1 - t0) * 1000)

            # 5. Semantic Retrieval (Top-5 search)
            t0 = time.perf_counter()
            search_res = vector_store.search(
                document_id=doc_bench.document_id,
                query="Will my subscription renew automatically?",
                top_k=5,
                threshold=0.45,
            )
            t1 = time.perf_counter()
            benchmarks["retrieval_search"].append((t1 - t0) * 1000)

            # 6. Cross-Clause Relationship Analysis
            t0 = time.perf_counter()
            relationships, _ = detect_relationships(clauses, document_id=doc_bench.document_id)
            t1 = time.perf_counter()
            benchmarks["relationship_analysis"].append((t1 - t0) * 1000)

            # 7. LLM Explanation Synthesis
            t0 = time.perf_counter()
            output = llm_service.synthesize_clauses(
                clauses=clauses[:5],
                relationships=relationships[:3],
                user_concerns=["Payment", "Privacy & Data"],
            )
            t1 = time.perf_counter()
            benchmarks["llm_explanation"].append((t1 - t0) * 1000)

            # 8. End-to-End Q&A Pipeline (Search + LLM Answering)
            t0 = time.perf_counter()
            qa_res = vector_store.search(
                document_id=doc_bench.document_id,
                query="Can I get a refund if I cancel early?",
                top_k=5,
                threshold=0.45,
            )
            c_map = {c.clause_id: c for c in clauses}
            matched = [c_map[r.clause_id] for r in qa_res.results if r.meets_threshold and r.clause_id in c_map]
            s_map = {r.clause_id: r.similarity_score for r in qa_res.results if r.meets_threshold}
            qa_out = llm_service.answer_question(
                question="Can I get a refund if I cancel early?",
                clauses=matched,
                scores=s_map,
            )
            t1 = time.perf_counter()
            benchmarks["end_to_end_qa"].append((t1 - t0) * 1000)

        # Compile latency metrics
        stats = {}
        for stage_name, measurements in benchmarks.items():
            stats[stage_name] = compute_latency_stats(measurements)
            stats[stage_name]["raw_measurements_ms"] = [round(m, 2) for m in measurements]

        env_info = get_environment_info()

        return {
            "benchmark_runs": num_runs,
            "environment": env_info,
            "stages": stats,
        }
    finally:
        settings.GEMINI_MOCK_MODE = original_mock
        client_mod._LLM_SERVICE_INSTANCE = None


if __name__ == "__main__":
    res = run_latency_benchmarks(num_runs=5)
    print("=== LATENCY BENCHMARKING RESULTS ===")
    print(f"Iterations: {res['benchmark_runs']}")
    print(f"Platform:   {res['environment']['platform']}")
    print(f"Python:     {res['environment']['python_version']}")
    print(f"RAM:        {res['environment']['ram_gb']} GB | CPU Cores: {res['environment']['cpu_logical_cores']}")
    print("\nStage Latency Summary (ms):")
    print(f"{'Pipeline Stage':<28} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 92)
    for stage, s in res["stages"].items():
        print(f"{stage:<28} | {s['mean_ms']:<10.2f} | {s['median_ms']:<12.2f} | {s['min_ms']:<10.2f} | {s['max_ms']:<10.2f} | {s['p95_ms']:<10.2f}")
