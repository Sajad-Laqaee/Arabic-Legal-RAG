# 🏛️ Arabic Legal RAG Pipeline 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Latest-purple)](https://github.com/facebookresearch/faiss)
[![CrossEncoder](https://img.shields.io/badge/CrossEncoder-Latest-orange)](https://www.sbert.net/examples/applications/re-ranking/README.html)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-lightblue)](https://ollama.ai/)

> A high‑precision Retrieval‑Augmented Generation (RAG) system for Arabic legal texts, designed for correctness over confidence.


---

## ✨ Overview

This repository contains an **end‑to‑end Arabic Legal RAG pipeline** built for Omani laws and royal decrees. It combines:

* Context‑aware legal chunking
* Dense vector retrieval (FAISS)
* Neural re‑ranking (Cross‑Encoder)
* A strictly controlled LLM prompt for grounded answers

The system is opinionated by design: **it refuses to answer when the evidence is insufficient**.

---

## 🧠 Architecture at a Glance

```
JSON Laws
   ↓
Legal Chunking (Articles + Text)
   ↓
Arabic Normalization
   ↓
Sentence‑Transformer Embeddings
   ↓
FAISS (IVF + Cosine Similarity)
   ↓
Cross‑Encoder Re‑Ranking
   ↓
LLM with Evidence‑Bound Prompt
   ↓
Cited Arabic Legal Answer
```

---

## 📂 Dataset Assumptions

The pipeline expects a JSON file containing a **list of legal documents (laws / decrees)**. While the real dataset may include additional metadata, only a subset of fields is required for the RAG logic.

### 📄 Minimal Example (Simplified)

```json
{
  "canonical_link": "https://qanoon.om/p/2025/rd2025100/",
  "text": "نحن هيثم بن طارق سلطان عمان ...",
  "issue_at": "صدر في: ٢٨ من جمادى الأولى سنة ١٤٤٧ هـ",
  "publication": "نشر في عدد الجريدة الرسمية رقم (١٦٢٣)...",
  "articles": [
    {
      "title": "المادة الأولى",
      "text": "التصديق على الاتفاقية المشار إليها..."
    }
  ]
}
```

### 🔑 Field Usage

* **`canonical_link`** → Extracts decree year and number for legal context
* **`text`** → Source for preamble extraction and raw text chunking
* **`articles[].title`** → Determines article numbering
* **`articles[].text`** → Clause-level splitting and article chunks
* **`issue_at`, `publication`** → Context enrichment for higher retrieval precision

Additional fields (e.g. source URLs, signatures, approval dates) may exist in the dataset but are not required by the current pipeline.

---

## 🧩 Chunking Strategy (Core Design)

This project intentionally uses **two parallel chunking paths** to balance recall and precision.

### 1️⃣ Article‑Based Chunks (Primary)

High‑signal chunks built around legal semantics:

* Decree metadata (year, number, publication)
* Extracted short preamble (ديباجة مختصرة)
* Article text
* Clause‑level splitting when applicable
* Line overlap to preserve continuity

These chunks are the backbone of accurate legal answers.

### 2️⃣ Text‑Based Chunks (Supplementary)

The raw `text` field of each law is also chunked:

* Fixed number of lines per chunk
* Configurable overlap

This improves recall for:

* Agreements
* Introductions
* Non‑article provisions

---

## 📝 Arabic Normalization

Before embedding, both documents and queries undergo lightweight normalization:

* Remove tatweel (ـ)
* Normalize Alef variants → ا
* Normalize ى → ي, ة → ه
* Collapse extra whitespace

This keeps the vector space stable without harming semantics.

---

## 📐 Embeddings & Vector Search

### 🔹 Embedding Model

* **Model:** `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2`
* Dense Arabic semantic embeddings

### 🔹 FAISS Index

* Type: `IndexIVFFlat`
* Metric: Inner Product (cosine similarity after normalization)
* Trained on a subset of chunks

Optimized for **recall**, not final ranking.

---

## 🔍 Retrieval & Re‑Ranking

### Step 1: Bi‑Encoder Retrieval

* Query → embedding
* FAISS retrieves top‑K candidate chunks

Fast, scalable, and intentionally noisy.

### Step 2: Cross‑Encoder Re‑Ranking

* **Model:** `Omartificial-Intelligence-Space/ARA-Reranker-V1`
* Scores each (query, chunk) pair jointly

### 🔢 Hybrid Ranking Formula

```
FinalRank = α · CrossEncoderRank + (1 − α) · FAISSRank
```

* `α → 1`: trust semantic relevance
* `α → 0`: trust vector similarity

This avoids brittle ranking behavior.

---

## 🤖 Answer Generation (LLM)

Top re‑ranked chunks are passed to an LLM with a **strict Arabic legal prompt** enforcing:

* Evidence‑only answers
* No hallucination
* Explicit verification of relevance
* Mandatory source citation

If no chunk answers the question, the system **must reply**:

> متأسف، لم أجد أي إجابة مناسبة.

This is a feature, not a failure.

---

## 📤 Output Characteristics

Final answers are:

* Formal Arabic
* Concise but complete
* Fully grounded in retrieved text
* Accompanied by clear sources (URL, rank, relevance)

Correctness is prioritized over fluency.

---




