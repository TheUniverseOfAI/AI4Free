import os

structure = [
    "rag_universe/docs/architecture",
    "rag_universe/docs/evaluation",
    "rag_universe/docs/patterns",
    "rag_universe/notebooks",
    "rag_universe/rag_core/python/rag_pipeline",
    "rag_universe/rag_core/python/evaluation",
    "rag_universe/rag_core/python/utils",
    "rag_universe/rag_core/node/src/pipeline",
    "rag_universe/rag_core/node/src/evaluation",
    "rag_universe/rag_core/node/src/utils",
    "rag_universe/examples/python",
    "rag_universe/examples/node",
    "rag_universe/data/sample_docs/finance",
    "rag_universe/data/sample_docs/health",
    "rag_universe/data/sample_docs/legal",
    "rag_universe/data/sample_docs/code",
    "rag_universe/data/eval",
    "rag_universe/projects/project_01_simple_rag_qna",
    "rag_universe/projects/project_02_hybrid_rag",
    "rag_universe/projects/project_03_agentic_rag",
    "rag_universe/projects/project_04_rag_with_reranker",
    "rag_universe/projects/project_05_domain_rag_finance",
    "rag_universe/projects/project_06_domain_rag_health",
    "rag_universe/projects/project_07_domain_rag_legal",
    "rag_universe/projects/project_08_codebase_rag_assistant"
]

for path in structure:
    os.makedirs(path, exist_ok=True)

# Create placeholder files
open("rag_universe/README.md", "w").write("# RAG Universe\n")
open("rag_universe/LICENSE", "w").write("MIT License\n")
open("rag_universe/CHANGELOG.md", "w").write("# Changelog\n")

print("RAG Universe folder structure created successfully!")
