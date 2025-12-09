# 📘 THE COMPLETE RAG BOOK

## 📑 Table of Contents (with links)

### PART I — FOUNDATIONS

#### Chapter 1 — Introduction to RAG

* [1.1 What Is Retrieval-Augmented Generation](#11-what-is-retrieval-augmented-generation)
* [1.2 Why RAG Exists](#12-why-rag-exists)
* [1.3 RAG vs Fine-Tuning](#13-rag-vs-fine-tuning)
* [1.4 RAG in the LLM System Stack](#14-rag-in-the-llm-system-stack)
* [1.5 When RAG Fails / When It Dominates](#15-when-rag-fails--when-it-dominates)

#### Chapter 2 — Pre-RAG Landscape

* [2.1 Traditional IR (TF-IDF, BM25)](#21-traditional-ir-tf-idf-bm25)
* [2.2 Embedding-Based Retrieval Emergence](#22-embedding-based-retrieval-emergence)
* [2.3 Memory-Based Architectures](#23-memory-based-architectures)
* [2.4 Early Hybrid QA Systems](#24-early-hybrid-qa-systems)

#### Chapter 3 — LLM Knowledge Requirements

* [3.1 Tokenization](#31-tokenization)
* [3.2 Context Windows](#32-context-windows)
* [3.3 System vs User Prompts](#33-system-vs-user-prompts)
* [3.4 Attention Mechanisms](#34-attention-mechanisms)
* [3.5 Hallucination Mechanisms](#35-hallucination-mechanisms)

---

### PART II — RAG PIPELINE BASICS

#### Chapter 4 — Data Preparation

* [4.1 Data Sources (PDF, HTML, Markdown, DB, API)](#41-data-sources-pdf-html-markdown-db-api)
* [4.2 Parsing & Normalization](#42-parsing--normalization)
* [4.3 Metadata Extraction](#43-metadata-extraction)
* [4.4 Document Cleaning](#44-document-cleaning)
* [4.5 Text Standardization](#45-text-standardization)

#### Chapter 5 — Chunking Strategies

* [5.1 Fixed-size Chunking](#51-fixed-size-chunking)
* [5.2 Recursive Chunking](#52-recursive-chunking)
* [5.3 Semantic Chunking](#53-semantic-chunking)
* [5.4 Sentence/Paragraph Boundary Chunking](#54-sentenceparagraph-boundary-chunking)
* [5.5 Contextual Overlap](#55-contextual-overlap)
* [5.6 Sliding Window Chunking](#56-sliding-window-chunking)
* [5.7 Domain-Specific Chunking (Code, Legal, Medical)](#57-domain-specific-chunking-code-legal-medical)

#### Chapter 6 — Embeddings

* [6.1 Dense Embeddings](#61-dense-embeddings)
* [6.2 Sparse Embeddings](#62-sparse-embeddings)
* [6.3 Multi-Vector Embeddings](#63-multi-vector-embeddings)
* [6.4 Cross-Encoder Embeddings](#64-cross-encoder-embeddings)
* [6.5 2-Tower vs Cross-Encoder Models](#65-2-tower-vs-cross-encoder-models)
* [6.6 Embedding Dimensionality](#66-embedding-dimensionality)
* [6.7 Embedding Quantization](#67-embedding-quantization)

#### Chapter 7 — Vector Databases

* [7.1 FAISS](#71-faiss)
* [7.2 Chroma](#72-chroma)
* [7.3 Pinecone](#73-pinecone)
* [7.4 Weaviate](#74-weaviate)
* [7.5 Milvus](#75-milvus)
* [7.6 PGVector / Elastic Vector Search](#76-pgvector--elastic-vector-search)
* [7.7 HNSW, IVF, PQ Index Types](#77-hnsw-ivf-pq-index-types)

---

### PART III — RETRIEVAL & RANKING MECHANISMS

#### Chapter 8 — Retrieval Techniques

* [8.1 k-NN Search](#81-k-nn-search)
* [8.2 Approximate Nearest Neighbor](#82-approximate-nearest-neighbor)
* [8.3 Hybrid Retrieval (BM25 + Vectors)](#83-hybrid-retrieval-bm25--vectors)
* [8.4 Metadata Filtering](#84-metadata-filtering)
* [8.5 Semantic Compression](#85-semantic-compression)
* [8.6 Multi-Hop Retrieval](#86-multi-hop-retrieval)
* [8.7 Routing-Based Retrieval](#87-routing-based-retrieval)

#### Chapter 9 — Reranking

* [9.1 Cross-Encoder Reranking](#91-cross-encoder-reranking)
* [9.2 LLM-Based Reranking](#92-llm-based-reranking)
* [9.3 Distilled Rerankers](#93-distilled-rerankers)
* [9.4 Multi-Stage Reranking Pipelines](#94-multi-stage-reranking-pipelines)
* [9.5 Score Fusion](#95-score-fusion)

#### Chapter 10 — Retrieval Optimization

* [10.1 Query Transformation](#101-query-transformation)
* [10.2 Query Expansion (Q2Q)](#102-query-expansion-q2q)
* [10.3 Query Rewriting (Q2C - Q2D)](#103-query-rewriting-q2c--q2d)
* [10.4 Self-Querying](#104-self-querying)
* [10.5 LLM-As-Retriever](#105-llm-as-retriever)

---

### PART IV — GENERATION LAYERS

#### Chapter 11 — Prompt Engineering for RAG

* [11.1 Prompt Templates](#111-prompt-templates)
* [11.2 Context Incorporation](#112-context-incorporation)
* [11.3 Guardrails & Constraints](#113-guardrails--constraints)
* [11.4 Chain-of-Thought in RAG](#114-chain-of-thought-in-rag)
* [11.5 Multi-Document Synthesis](#115-multi-document-synthesis)

#### Chapter 12 — Answer Generation

* [12.1 Extractive vs Abstractive Answers](#121-extractive-vs-abstractive-answers)
* [12.2 Multi-Pass Generation](#122-multi-pass-generation)
* [12.3 Citation-Aware Generation](#123-citation-aware-generation)
* [12.4 Source Attribution](#124-source-attribution)
* [12.5 Style Control](#125-style-control)
* [12.6 Structured Output Generation](#126-structured-output-generation)

#### Chapter 13 — Advanced Generation Strategies

* [13.1 LLM-as-a-Judge](#131-llm-as-a-judge)
* [13.2 LLM-as-a-Reranker](#132-llm-as-a-reranker)
* [13.3 RAG-Fusion](#133-rag-fusion)
* [13.4 Generative Retrieval](#134-generative-retrieval)
* [13.5 ReAct + RAG](#135-react--rag)
* [13.6 GraphRAG](#136-graphrag)
* [13.7 Agentic RAG](#137-agentic-rag)

---

### PART V — RAG TYPES

#### Chapter 14 — Basic RAG

* [14.1 Standard 2-Stage RAG](#141-standard-2-stage-rag)
* [14.2 One-Shot RAG](#142-one-shot-rag)

#### Chapter 15 — Advanced RAG Variants

* [15.1 RAG-Fusion](#151-rag-fusion)
* [15.2 Multi-Query RAG](#152-multi-query-rag)
* [15.3 HyDE (Generate Hypothetical Document)](#153-hyde-generate-hypothetical-document)
* [15.4 Adaptive RAG](#154-adaptive-rag)
* [15.5 GraphRAG](#155-graphrag)
* [15.6 Hierarchical RAG](#156-hierarchical-rag)

#### Chapter 16 — Multi-Modal RAG

* [16.1 Image Retrieval](#161-image-retrieval)
* [16.2 Audio Retrieval](#162-audio-retrieval)
* [16.3 Vector-Mixed Modalities](#163-vector-mixed-modalities)
* [16.4 OCR Pipelines](#164-ocr-pipelines)

#### Chapter 17 — Agentic RAG

* [17.1 Tool-Use Agents](#171-tool-use-agents)
* [17.2 Planner-Executor Agents](#172-planner-executor-agents)
* [17.3 Multi-Agent Retrieval Teams](#173-multi-agent-retrieval-teams)
* [17.4 Agents + RAG Memory Systems](#174-agents--rag-memory-systems)

---

### PART VI — EVALUATION & QUALITY FRAMEWORKS

#### Chapter 18 — RAG Evaluation

* [18.1 Precision, Recall, MRR](#181-precision-recall-mrr)
* [18.2 Semantic Similarity Metrics](#182-semantic-similarity-metrics)
* [18.3 Human Eval](#183-human-eval)

#### Chapter 19 — Automatic RAG Evaluation

* [19.1 RAGAS Framework](#191-ragas-framework)
* [19.2 TruLens](#192-trulens)
* [19.3 LLM-As-A-Judge Metrics](#193-llm-as-a-judge-metrics)
* [19.4 Context Relevance Scoring](#194-context-relevance-scoring)
* [19.5 Faithfulness Evaluation](#195-faithfulness-evaluation)
* [19.6 Output Groundedness](#196-output-groundedness)

#### Chapter 20 — Observability

* [20.1 Logging](#201-logging)
* [20.2 Span Tracing](#202-span-tracing)
* [20.3 Retrieval Heatmaps](#203-retrieval-heatmaps)
* [20.4 Context Window Diagnostics](#204-context-window-diagnostics)

---

### PART VII — PERFORMANCE ENGINEERING

#### Chapter 21 — Latency Optimization

* [21.1 Embedding Cache](#211-embedding-cache)
* [21.2 Retrieval Cache](#212-retrieval-cache)
* [21.3 Prompt Cache](#213-prompt-cache)
* [21.4 Parallel Retrieval](#214-parallel-retrieval)
* [21.5 Batch Retrieval](#215-batch-retrieval)

#### Chapter 22 — Cost Optimization

* [22.1 Embedding Cost Planning](#221-embedding-cost-planning)
* [22.2 Generation Cost Control](#222-generation-cost-control)
* [22.3 Context Compression](#223-context-compression)
* [22.4 Distillation Techniques](#224-distillation-techniques)

#### Chapter 23 — Scalability

* [23.1 Sharding](#231-sharding)
* [23.2 Replicated Indexes](#232-replicated-indexes)
* [23.3 Tiered Storage](#233-tiered-storage)
* [23.4 Distributed Vector DB](#234-distributed-vector-db)

---

### PART VIII — SYSTEM ARCHITECTURE

#### Chapter 24 — RAG Architecture Patterns

* [24.1 Local RAG](#241-local-rag)
* [24.2 Cloud RAG](#242-cloud-rag)
* [24.3 Hybrid RAG](#243-hybrid-rag)
* [24.4 Streaming RAG](#244-streaming-rag)
* [24.5 On-Device RAG](#245-on-device-rag)

#### Chapter 25 — Knowledge Graph + RAG

* [25.1 Entity Extraction](#251-entity-extraction)
* [25.2 Ranking via Graph Signals](#252-ranking-via-graph-signals)
* [25.3 GraphRAG Pipeline](#253-graphrag-pipeline)
* [25.4 Community Detection](#254-community-detection)

#### Chapter 26 — Enterprise RAG

* [26.1 Access Control](#261-access-control)
* [26.2 Multi-Tenant Embeddings](#262-multi-tenant-embeddings)
* [26.3 Data Governance](#263-data-governance)
* [26.4 Semantic Caching](#264-semantic-caching)
* [26.5 Compliance Layers](#265-compliance-layers)

---

### PART IX — TOOLING & FRAMEWORKS

#### Chapter 27 — RAG Frameworks

* [27.1 LangChain](#271-langchain)
* [27.2 LlamaIndex](#272-llamaindex)
* [27.3 Haystack](#273-haystack)
* [27.4 DSPy](#274-dspy)
* [27.5 Semantic Kernel](#275-semantic-kernel)

#### Chapter 28 — Vector Databases (Deep Dive)

* [28.1 Indexing](#281-indexing)
* [28.2 Replication](#282-replication)
* [28.3 Compaction](#283-compaction)
* [28.4 Filtering](#284-filtering)
* [28.5 Batch Insert Optimization](#285-batch-insert-optimization)

#### Chapter 29 — Deployment

* [29.1 Docker](#291-docker)
* [29.2 Serverless](#292-serverless)
* [29.3 GPU Inference](#293-gpu-inference)
* [29.4 Kubernetes](#294-kubernetes)
* [29.5 CI/CD for RAG Systems](#295-cicd-for-rag-systems)

---

### PART X — FRONTIER RAG RESEARCH

#### Chapter 30 — RAG 2.0

* [30.1 Self-Correcting RAG](#301-self-correcting-rag)
* [30.2 Memory-Augmented RAG](#302-memory-augmented-rag)
* [30.3 Dynamic Retrieval](#303-dynamic-retrieval)
* [30.4 LLM-Optimized Chunking](#304-llm-optimized-chunking)

#### Chapter 31 — Model-Assisted Retrieval

* [31.1 Generative Search Augmentation](#311-generative-search-augmentation)
* [31.2 Model-Driven Indexing](#312-model-driven-indexing)
* [31.3 Embedding Fine-Tuning](#313-embedding-fine-tuning)

#### Chapter 32 — Future of RAG

* [32.1 RAG + Agents](#321-rag--agents)
* [32.2 RAG + Autonomous Systems](#322-rag--autonomous-systems)
* [32.3 RAG + Multi-Modal LLMs](#323-rag--multi-modal-llms)
* [32.4 RAG-as-Infrastructure](#324-rag-as-infrastructure)

---

## 📄 Empty Sections (stubs to fill later)

> All sections are intentionally empty; just start typing under each heading.

---

### Chapter 1 — Introduction to RAG

#### 1.1 What Is Retrieval-Augmented Generation

#### 1.2 Why RAG Exists

#### 1.3 RAG vs Fine-Tuning

#### 1.4 RAG in the LLM System Stack

#### 1.5 When RAG Fails / When It Dominates

---

### Chapter 2 — Pre-RAG Landscape

#### 2.1 Traditional IR (TF-IDF, BM25)

#### 2.2 Embedding-Based Retrieval Emergence

#### 2.3 Memory-Based Architectures

#### 2.4 Early Hybrid QA Systems

---

### Chapter 3 — LLM Knowledge Requirements

#### 3.1 Tokenization

#### 3.2 Context Windows

#### 3.3 System vs User Prompts

#### 3.4 Attention Mechanisms

#### 3.5 Hallucination Mechanisms

---

### Chapter 4 — Data Preparation

#### 4.1 Data Sources (PDF, HTML, Markdown, DB, API)

#### 4.2 Parsing & Normalization

#### 4.3 Metadata Extraction

#### 4.4 Document Cleaning

#### 4.5 Text Standardization

---

### Chapter 5 — Chunking Strategies

#### 5.1 Fixed-size Chunking

#### 5.2 Recursive Chunking

#### 5.3 Semantic Chunking

#### 5.4 Sentence/Paragraph Boundary Chunking

#### 5.5 Contextual Overlap

#### 5.6 Sliding Window Chunking

#### 5.7 Domain-Specific Chunking (Code, Legal, Medical)

---

### Chapter 6 — Embeddings

#### 6.1 Dense Embeddings

#### 6.2 Sparse Embeddings

#### 6.3 Multi-Vector Embeddings

#### 6.4 Cross-Encoder Embeddings

#### 6.5 2-Tower vs Cross-Encoder Models

#### 6.6 Embedding Dimensionality

#### 6.7 Embedding Quantization

---

### Chapter 7 — Vector Databases

#### 7.1 FAISS

#### 7.2 Chroma

#### 7.3 Pinecone

#### 7.4 Weaviate

#### 7.5 Milvus

#### 7.6 PGVector / Elastic Vector Search

#### 7.7 HNSW, IVF, PQ Index Types

---

### Chapter 8 — Retrieval Techniques

#### 8.1 k-NN Search

#### 8.2 Approximate Nearest Neighbor

#### 8.3 Hybrid Retrieval (BM25 + Vectors)

#### 8.4 Metadata Filtering

#### 8.5 Semantic Compression

#### 8.6 Multi-Hop Retrieval

#### 8.7 Routing-Based Retrieval

---

### Chapter 9 — Reranking

#### 9.1 Cross-Encoder Reranking

#### 9.2 LLM-Based Reranking

#### 9.3 Distilled Rerankers

#### 9.4 Multi-Stage Reranking Pipelines

#### 9.5 Score Fusion

---

### Chapter 10 — Retrieval Optimization

#### 10.1 Query Transformation

#### 10.2 Query Expansion (Q2Q)

#### 10.3 Query Rewriting (Q2C - Q2D)

#### 10.4 Self-Querying

#### 10.5 LLM-As-Retriever

---

### Chapter 11 — Prompt Engineering for RAG

#### 11.1 Prompt Templates

#### 11.2 Context Incorporation

#### 11.3 Guardrails & Constraints

#### 11.4 Chain-of-Thought in RAG

#### 11.5 Multi-Document Synthesis

---

### Chapter 12 — Answer Generation

#### 12.1 Extractive vs Abstractive Answers

#### 12.2 Multi-Pass Generation

#### 12.3 Citation-Aware Generation

#### 12.4 Source Attribution

#### 12.5 Style Control

#### 12.6 Structured Output Generation

---

### Chapter 13 — Advanced Generation Strategies

#### 13.1 LLM-as-a-Judge

#### 13.2 LLM-as-a-Reranker

#### 13.3 RAG-Fusion

#### 13.4 Generative Retrieval

#### 13.5 ReAct + RAG

#### 13.6 GraphRAG

#### 13.7 Agentic RAG

---

### Chapter 14 — Basic RAG

#### 14.1 Standard 2-Stage RAG

#### 14.2 One-Shot RAG

---

### Chapter 15 — Advanced RAG Variants

#### 15.1 RAG-Fusion

#### 15.2 Multi-Query RAG

#### 15.3 HyDE (Generate Hypothetical Document)

#### 15.4 Adaptive RAG

#### 15.5 GraphRAG

#### 15.6 Hierarchical RAG

---

### Chapter 16 — Multi-Modal RAG

#### 16.1 Image Retrieval

#### 16.2 Audio Retrieval

#### 16.3 Vector-Mixed Modalities

#### 16.4 OCR Pipelines

---

### Chapter 17 — Agentic RAG

#### 17.1 Tool-Use Agents

#### 17.2 Planner-Executor Agents

#### 17.3 Multi-Agent Retrieval Teams

#### 17.4 Agents + RAG Memory Systems

---

### Chapter 18 — RAG Evaluation

#### 18.1 Precision, Recall, MRR

#### 18.2 Semantic Similarity Metrics

#### 18.3 Human Eval

---

### Chapter 19 — Automatic RAG Evaluation

#### 19.1 RAGAS Framework

#### 19.2 TruLens

#### 19.3 LLM-As-A-Judge Metrics

#### 19.4 Context Relevance Scoring

#### 19.5 Faithfulness Evaluation

#### 19.6 Output Groundedness

---

### Chapter 20 — Observability

#### 20.1 Logging

#### 20.2 Span Tracing

#### 20.3 Retrieval Heatmaps

#### 20.4 Context Window Diagnostics

---

### Chapter 21 — Latency Optimization

#### 21.1 Embedding Cache

#### 21.2 Retrieval Cache

#### 21.3 Prompt Cache

#### 21.4 Parallel Retrieval

#### 21.5 Batch Retrieval

---

### Chapter 22 — Cost Optimization

#### 22.1 Embedding Cost Planning

#### 22.2 Generation Cost Control

#### 22.3 Context Compression

#### 22.4 Distillation Techniques

---

### Chapter 23 — Scalability

#### 23.1 Sharding

#### 23.2 Replicated Indexes

#### 23.3 Tiered Storage

#### 23.4 Distributed Vector DB

---

### Chapter 24 — RAG Architecture Patterns

#### 24.1 Local RAG

#### 24.2 Cloud RAG

#### 24.3 Hybrid RAG

#### 24.4 Streaming RAG

#### 24.5 On-Device RAG

---

### Chapter 25 — Knowledge Graph + RAG

#### 25.1 Entity Extraction

#### 25.2 Ranking via Graph Signals

#### 25.3 GraphRAG Pipeline

#### 25.4 Community Detection

---

### Chapter 26 — Enterprise RAG

#### 26.1 Access Control

#### 26.2 Multi-Tenant Embeddings

#### 26.3 Data Governance

#### 26.4 Semantic Caching

#### 26.5 Compliance Layers

---

### Chapter 27 — RAG Frameworks

#### 27.1 LangChain

#### 27.2 LlamaIndex

#### 27.3 Haystack

#### 27.4 DSPy

#### 27.5 Semantic Kernel

---

### Chapter 28 — Vector Databases (Deep Dive)

#### 28.1 Indexing

#### 28.2 Replication

#### 28.3 Compaction

#### 28.4 Filtering

#### 28.5 Batch Insert Optimization

---

### Chapter 29 — Deployment

#### 29.1 Docker

#### 29.2 Serverless

#### 29.3 GPU Inference

#### 29.4 Kubernetes

#### 29.5 CI/CD for RAG Systems

---

### Chapter 30 — RAG 2.0

#### 30.1 Self-Correcting RAG

#### 30.2 Memory-Augmented RAG

#### 30.3 Dynamic Retrieval

#### 30.4 LLM-Optimized Chunking

---

### Chapter 31 — Model-Assisted Retrieval

#### 31.1 Generative Search Augmentation

#### 31.2 Model-Driven Indexing

#### 31.3 Embedding Fine-Tuning

---

### Chapter 32 — Future of RAG

#### 32.1 RAG + Agents

#### 32.2 RAG + Autonomous Systems

#### 32.3 RAG + Multi-Modal LLMs

#### 32.4 RAG-as-Infrastructure



| **Roadmap Phase**                                  | **Learning Focus**                               | **Aligned TOC Chapters** | **What These Chapters Cover (Short Labels Only)**                                                  |
| -------------------------------------------------- | ------------------------------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------- |
| **Phase 1 — Foundations**                          | Understand RAG at a conceptual level             | **Ch. 1–3**              | Introduction, pre-RAG history, LLM fundamentals                                                    |
| **Phase 2 — Core RAG Pipeline**                    | Learn the essential components of any RAG system | **Ch. 4–13**             | Data prep, chunking, embeddings, vector DBs, retrieval, reranking, generation, advanced generation |
| **Phase 3 — RAG Types & Variants**                 | Explore specialized, more powerful RAG designs   | **Ch. 14–17**            | Basic RAG, Fusion, HyDE, GraphRAG, multimodal, agentic RAG                                         |
| **Phase 4 — Evaluation & Observability**           | Learn how to measure, debug, and improve RAG     | **Ch. 18–20**            | Metrics, RAGAS, LLM-as-judge, TruLens, tracing, diagnostics                                        |
| **Phase 5 — Performance Engineering**              | Make RAG fast, efficient, scalable               | **Ch. 21–23**            | Latency, cost, caching, parallel retrieval, scaling                                                |
| **Phase 6 — System Architecture & Enterprise RAG** | Learn how big companies build production RAG     | **Ch. 24–26**            | Architecture patterns, graph integration, multi-tenant systems, governance                         |
| **Phase 7 — Frameworks & Deployment**              | Master the tools and infrastructure              | **Ch. 27–29**            | LangChain, LlamaIndex, vector DB internals, Docker, K8s, CI/CD                                     |
| **Phase 8 — Frontier Research (RAG 2.0+)**         | Learn cutting-edge RAG techniques                | **Ch. 30–32**            | Self-correcting RAG, dynamic retrieval, generative indexing, future directions                     |

| **Phase**               | **TOC Chapters** |
| ----------------------- | ---------------- |
| Phase 1 — Foundations   | 1–3              |
| Phase 2 — Core Pipeline | 4–13             |
| Phase 3 — RAG Types     | 14–17            |
| Phase 4 — Evaluation    | 18–20            |
| Phase 5 — Performance   | 21–23            |
| Phase 6 — Architecture  | 24–26            |
| Phase 7 — Frameworks    | 27–29            |
| Phase 8 — Frontier      | 30–32            |


-----

## 🐍 Notebook 4 — RAG Hands-On (Python) — Company Files Q&A

**What it does:**

* Loads **PDF / DOCX / CSV** from `./data`
* Chunking with `RecursiveCharacterTextSplitter`
* Multiple embedding backends:

  * `text-embedding-3-small`
  * `text-embedding-3-large`
  * `sentence-transformers/all-mpnet-base-v2` (local)
* Multiple vector DBs:

  * **FAISS** (in-memory/file, offline)
  * **Chroma** (local, persistent)
* Multiple chat models (switchable):

  * e.g. `gpt-4o-mini`, `gpt-4.1` (you can rename to models you actually have)
* Simple but explicit **conversation memory**
* One clean function: `rag_answer(question)` for Q&A over company files
* Interactive loop at the end to test QA in the notebook

---

## 🟦 Notebook 5 — RAG Hands-On (Node.js) — Company Files Q&A (Backend Only)

This one is a **blueprint notebook** with full Node.js code laid out as files.

**What it includes:**

* Suggested project structure (`rag-node-backend/…`)
* `package.json` with core deps (LangChain, OpenAI, FAISS, Chroma, etc.)
* Loaders for:

  * `pdfLoader.ts`
  * `docxLoader.ts` (simplified / placeholder extraction)
  * `csvLoader.ts`
* Aggregated loader `loadAllDocuments(dataDir)`
* Chunking with `RecursiveCharacterTextSplitter`
* Embedding registry:

  * `openai_small`, `openai_large` (easy to extend to HF embeddings)
* Vector store registry:

  * `memory` (MemoryVectorStore)
  * `faiss` (via `@langchain/community/vectorstores/faiss`)
  * `chroma` client (requires running a Chroma server)
* Chat model registry:

  * `gpt_4_small`, `gpt_4_full` (via `@langchain/openai`)
* Conversation memory class
* `RAGPipeline` class with:

  * `init()` → load + chunk + index
  * `answer(question)` → retrieve + build prompt + call LLM + update memory
* Optional tiny HTTP server (`/ask`) for manual testing with curl/Postman

---

### How these fit with your other notebooks

* **TOC notebook** → full RAG topic universe
* **Roadmap notebook** → what to learn, in which order
* **Deep-Dive notebook** → explanations and intuition
* **Hands-On Python** → runnable pipeline for company-file Q&A
* **Hands-On Node** → backend blueprint for the same use case
