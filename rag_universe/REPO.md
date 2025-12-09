rag_universe/
│
├── README.md
├── LICENSE
├── CHANGELOG.md
│
├── docs/
│   ├── architecture/
│   │   ├── simple_rag.png
│   │   ├── hybrid_rag.png
│   │   ├── agentic_rag.png
│   │   ├── rag_mcp_architecture.png
│   │   └── production_rag_pipeline.png
│   ├── evaluation/
│   │   ├── metrics_overview.md
│   │   ├── llm_as_judge_prompts.md
│   │   └── ragas_concepts.md
│   └── patterns/
│       ├── finance_rag_playbook.md
│       ├── medical_rag_playbook.md
│       ├── legal_rag_playbook.md
│       └── codebase_rag_playbook.md
│
├── notebooks/
│   ├── 01_RAG_TOC.ipynb
│   ├── 02_RAG_Learning_Roadmap.ipynb
│   ├── 03_RAG_DeepDive_Concepts.ipynb
│   ├── 04_RAG_HandsOn_Python.ipynb
│   ├── 05_RAG_HandsOn_Node.ipynb
│   ├── 06_RAG_Architecture_Diagrams.ipynb
│   ├── 07_RAG_Evaluation_and_Benchmarks.ipynb
│   └── 08_RAG_Real_World_Patterns.ipynb
│
├── rag_core/
│   ├── python/
│   │   ├── rag_pipeline/
│   │   │   ├── chunker.py
│   │   │   ├── embedder.py
│   │   │   ├── vector_store.py
│   │   │   ├── retriever.py
│   │   │   ├── reranker.py
│   │   │   ├── generator.py
│   │   │   └── pipeline.py
│   │   ├── evaluation/
│   │   │   ├── llm_judge.py
│   │   │   ├── metrics.py
│   │   │   └── evaluator.py
│   │   └── utils/
│   │       ├── file_loader.py
│   │       ├── converters.py
│   │       └── settings.py
│   │
│   └── node/
│       ├── src/
│       │   ├── pipeline/
│       │   │   ├── chunker.js
│       │   │   ├── embedder.js
│       │   │   ├── vectorStore.js
│       │   │   ├── retriever.js
│       │   │   ├── reranker.js
│       │   │   └── generator.js
│       │   ├── evaluation/
│       │   │   ├── llmJudge.js
│       │   │   ├── metrics.js
│       │   │   └── evaluator.js
│       │   └── utils/
│       │       ├── loader.js
│       │       ├── converter.js
│       │       └── config.js
│       └── package.json
│
├── examples/
│   ├── python/
│   │   ├── simple_qna.py
│   │   ├── hybrid_rag_example.py
│   │   ├── agentic_rag_demo.py
│   │   └── mcp_rag_demo.py
│   └── node/
│       ├── simpleQna.js
│       ├── hybridRagExample.js
│       ├── agenticRagDemo.js
│       └── mcpRagDemo.js
│
├── data/
│   ├── sample_docs/
│   │   ├── finance/
│   │   ├── health/
│   │   ├── legal/
│   │   └── code/
│   └── eval/
│       ├── eval_questions.jsonl
│       └── gold_answers.jsonl
│
└── projects/
    ├── project_01_simple_rag_qna/
    │   ├── python/
    │   └── node/
    ├── project_02_hybrid_rag/
    │   ├── python/
    │   └── node/
    ├── project_03_agentic_rag/
    │   ├── python/
    │   └── node/
    ├── project_04_rag_with_reranker/
    │   ├── python/
    │   └── node/
    ├── project_05_domain_rag_finance/
    ├── project_06_domain_rag_health/
    ├── project_07_domain_rag_legal/
    └── project_08_codebase_rag_assistant/
# 🌌 RAG Universe  
### Retrieval-Augmented Generation — A Complete Learning, Hands-On, and Project Ecosystem  
Part of the **AI for Free** initiative.

---

## 📖 Overview  
The **RAG Universe** is a world-class, full-stack repository for learning, building, and mastering **Retrieval-Augmented Generation (RAG)**.  
It combines:

- Theory (TOC, roadmap, deep dives)  
- Real architectures (diagrams, patterns)  
- Practical labs (Python + Node.js)  
- Evaluation playbooks (RAGAS, LLM-as-judge, benchmarks)  
- Real-world templates (finance, health, legal, code)  
- End-to-end project blueprints  

This repo is intentionally structured so **any learner or engineer** can go from:

> **Zero → Practical RAG Engineer → Production-Ready RAG Architect**

---

# 📁 Repository Structure  
This repo follows a **unified pattern** used across all universes (Agents, MCP, ML, etc.) in your **AI for Free** ecosystem.


---

# 🧱 Contents Explained

### ✔ **1. `docs/` — Architecture, Evaluation, Patterns**  
Production-ready diagrams, evaluation metrics, and domain playbooks (Finance, Health, Legal, Code).

### ✔ **2. `notebooks/` — The Learning Path**  
8 notebooks forming a complete RAG curriculum:
- TOC  
- Roadmap  
- Deep Dive  
- Hands-On Python  
- Hands-On Node  
- Architecture diagrams  
- Evaluation & Benchmarks  
- Real-World RAG Patterns  

### ✔ **3. `rag_core/` — The Framework Layer**  
Reusable RAG components in Python + Node.js:

- Chunkers  
- Embedders  
- Vector stores  
- Retrievers  
- Rerankers  
- LLM generators  
- Evaluation utilities  

This becomes the “core library” used by examples and projects.

### ✔ **4. `examples/` — Quick-Start Scripts**  
Minimal runnable demos in both languages.

### ✔ **5. `data/` — Sample Corpora + Evaluation Sets**  
PDFs, markdown, policies, finance docs, medical text, contracts, and evaluation pairs.

### ✔ **6. `projects/` — Real RAG Projects**  
Each folder is a portfolio-grade build:
- simple RAG Q&A  
- hybrid RAG  
- Agentic RAG  
- reranker RAG  
- finance assistant  
- medical explainer  
- legal navigator  
- codebase assistant

---

# 🚀 Goals of RAG Universe

- Be the **best open RAG curriculum** online  
- Teach RAG from foundations → production  
- Provide hands-on Python/Node pipelines  
- Support domain-specific real-world RAG  
- Integrate with Agents & MCP universes  
- Produce true **AI Engineers**, not hobbyists  

This repo is meant to be readable, forkable, teachable, and ready for real deployments.

---

# 🛠️ Requirements

### **Python**
- Python 3.10+
- `pip install -r requirements.txt`

### **Node.js**
- Node 18+
- `npm install`

---

# 🧪 Quick Start (Python)

```bash
python examples/python/simple_qna.py \
  --docs "./data/sample_docs/finance" \
  --query "Explain dollar-cost averaging"

---

# 🧱 **Python Script to Auto-Generate This Entire Folder Structure**  
*(Run this once to scaffold your repo)*

```python
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
