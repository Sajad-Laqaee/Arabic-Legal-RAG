import json
import re
from typing import List, Dict, Tuple
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from scipy.stats import rankdata


JSON_PATH = "/content/drive/MyDrive/Project/my-qanoon-data.json"

# -----------------------------
# Load JSON data
# -----------------------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

assert isinstance(data, list), "Top-level JSON must be a list of laws"

# -----------------------------
# Helper Functions
# -----------------------------

def extract_decree_info(canonical_link: str) -> Tuple[str, str]:
    """
    Extract the decree year and number from a canonical link.
    Returns (year, number) or (None, None) if not found.
    Example: "/2025/rd2025100" -> ("2025", "100")
    """
    m = re.search(r"/(\d{4})/rd(\d+)", canonical_link or "")
    if not m:
        return None, None
    year = m.group(1)
    raw = m.group(2)
    number = raw[len(year):] if raw.startswith(year) else raw
    return year, number

def extract_article_number(title: str) -> str:
    """
    Extracts the article number from a title string, e.g., "(3)" -> "3".
    """
    if not title:
        return ""
    m = re.search(r"\(([^)]+)\)", title)
    return m.group(1) if m else ""

def build_short_preamble(text: str, max_lines: int = 4) -> str:
    """
    Builds a short preamble from the law text by capturing key introductory lines.
    Lines starting with 'نحن', 'بعد الاطلاع', 'وعلى' or containing 'الاتفاقية' are included.
    Stops at 'وبناء' or after max_lines.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result = []
    capturing = False

    for line in lines:
        if line.startswith("نحن"):
            result.append(line)
            capturing = True
            continue
        if capturing:
            if line.startswith("وبناء"):
                break
            if (
                line.startswith("بعد الاطلاع")
                or line.startswith("وعلى")
                or "الاتفاقية" in line
            ):
                result.append(line)
        if len(result) >= max_lines:
            break
    return "\n".join(result)

def split_text_into_chunks(text: str, max_lines: int = 4, overlap_lines: int = 1) -> List[str]:
    """
    Split long text into chunks of max_lines, optionally overlapping previous lines.
    Returns a list of text chunks.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i+max_lines]
        if i != 0 and overlap_lines > 0:
            chunk_lines = lines[i-overlap_lines:i] + chunk_lines
        chunks.append('\n'.join(chunk_lines))
        i += max_lines
    return chunks

def split_long_article(text: str) -> List[Tuple[str, str]]:
    """
    Split a long article into clauses using numbering patterns.
    Returns a list of tuples (clause_number, clause_text).
    """
    if not text:
        return [(None, "")]
    pattern = r"\n?([0-9٠-٩]+)\s*[–\-]\s*"
    parts = re.split(pattern, text)
    if len(parts) <= 1:
        return [(None, text.strip())]
    clauses = []
    it = iter(parts)
    first = next(it).strip()
    if first:
        clauses.append((None, first))
    for num, body in zip(it, it):
        clauses.append((num, body.strip()))
    return clauses

# -----------------------------
# Article Chunk Builder
# -----------------------------

def build_chunks_for_article(law: Dict, article: Dict, overlap_lines: int = 2) -> List[Dict]:
    """
    Build structured chunks for a single article, including:
      - Law context metadata
      - Short preamble
      - Article text broken into clauses with optional overlap
    Returns a list of dictionaries with 'text' and 'metadata'.
    """
    year, decree_number = extract_decree_info(law.get("canonical_link"))
    article_number = extract_article_number(article.get("title", ""))

    context = f"""[سياق تشريعي]
نوع الوثيقة: مرسوم سلطاني
الرقم: {decree_number}
السنة: {year}
{law.get('issue_at', '').strip()}
{law.get('publication', '').strip()}
""".strip()

    preamble = f"""[ديباجة مختصرة]
{build_short_preamble(law.get('text', ''))}
""".strip()

    clauses = split_long_article(article.get("text", ""))
    chunks = []
    previous_lines = []

    for clause_no, clause_text in clauses:
        clause_lines = clause_text.splitlines()
        if previous_lines:
            overlap_text = '\n'.join(previous_lines[-overlap_lines:])
            clause_lines = [f"[ديباجة مختصرة – استمرار الجزء السابق]"] + [overlap_text] + clause_lines
        clause_text_with_overlap = '\n'.join(clause_lines).strip()
        clause_label = f" – الفقرة ({clause_no})" if clause_no else ""
        legal_text = f"""[ النص القانونی / ماهية: مادة]

{article.get('title', '').strip()}{clause_label}:
{clause_text_with_overlap}
""".strip()
        full_text = '\n\n'.join([context, preamble, legal_text])
        chunks.append({
            "text": full_text,
            "metadata": {
                "canonical_link": law.get("canonical_link"),
                "decree_year": year,
                "decree_number": decree_number,
                "article_number": article_number,
                "clause_number": clause_no,
                "is_overlap": bool(previous_lines),
            }
        })
        previous_lines = clause_lines

    return chunks

# -----------------------------
# Text Field Chunk Builder
# -----------------------------

def build_chunks_for_text(law: Dict, max_lines: int = 4, overlap_lines: int = 1) -> List[Dict]:
    """
    Split the 'text' field of the law into chunks with optional overlap.
    Returns a list of dictionaries with 'text' and metadata.
    """
    text_field = law.get("text", "")
    if not text_field:
        return []
    text_chunks = split_text_into_chunks(text_field, max_lines=max_lines, overlap_lines=overlap_lines)
    chunks = []
    for idx, chunk in enumerate(text_chunks):
        chunks.append({
            "text": f"""[ماهیت: نص]{chunk}""",
            "metadata": {
                "canonical_link": law.get("canonical_link"),
                "chunk_index": idx,
                "is_overlap": idx > 0
            }
        })
    return chunks

# -----------------------------
# Build All Chunks
# -----------------------------

all_chunks: List[Dict] = []
for law in data:
    # 1. Article chunks
    for article in law.get("articles", []):
        all_chunks.extend(build_chunks_for_article(law, article))
    # 2. Text chunks
    all_chunks.extend(build_chunks_for_text(law))

print(f"Total chunks generated: {len(all_chunks)}")

# -------------------- Display Results --------------------
print("\n--- Top results ---")
for dist, idx in zip(D[0], I[0]):
    print(f"Distance: {dist:.4f}")
    print(f"Text preview: {all_chunks[idx]['text'][:200]}...")
    print(f"Metadata: {metadata_store[idx]}")
    print("-"*50)


# -------------------- Arabic Text Normalization --------------------
def normalize_arabic(text: str) -> str:
    """
    Standardize Arabic text for embedding:
    - Remove tatweel (ـ)
    - Normalize alef variants (إأآا -> ا)
    - Convert final ya (ى) to ي, and taa marbuta (ة) to ه
    - Collapse multiple whitespace to single space
    """
    text = text.replace("ـ", "")
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------- Prepare Chunks --------------------
for chunk in all_chunks:
    # normalize text for consistent embeddings
    chunk['text'] = normalize_arabic(chunk['text'])

    # ensure 'chunk_type' exists
    if 'chunk_type' not in chunk['metadata']:
        if '[النص القانوني]' in chunk['text'] or '[ماهية: مادة]' in chunk['text']:
            chunk['metadata']['chunk_type'] = 'مادة'
        else:
            chunk['metadata']['chunk_type'] = 'نص'

    # ensure 'is_overlap' exists
    if 'is_overlap' not in chunk['metadata']:
        chunk['metadata']['is_overlap'] = False

print(f"Total chunks ready for embedding: {len(all_chunks)}")

# -------------------- Load Embedding Model --------------------
model = SentenceTransformer("Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")

# -------------------- Prepare FAISS Index --------------------
sample_emb = model.encode([all_chunks[0]['text']], convert_to_numpy=True)
embedding_dim = sample_emb.shape[-1]

nlist = 500  # number of IVF clusters
quantizer = faiss.IndexFlatIP(embedding_dim)  # inner-product = cosine similarity
index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

# -------------------- Train FAISS --------------------
sample_size = min(5000, len(all_chunks))
sample_embs = model.encode([c['text'] for c in all_chunks[:sample_size]], convert_to_numpy=True)
sample_embs = sample_embs / np.linalg.norm(sample_embs, axis=1, keepdims=True)  # normalize embeddings
index.train(sample_embs.astype(np.float32))

# -------------------- Add Embeddings in Batches --------------------
batch_size = 500
metadata_store = []

for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i+batch_size]
    batch_texts = [c['text'] for c in batch]
    batch_embs = model.encode(batch_texts, convert_to_numpy=True)
    batch_embs = batch_embs / np.linalg.norm(batch_embs, axis=1, keepdims=True)
    index.add(batch_embs.astype(np.float32))
    metadata_store.extend([c['metadata'] for c in batch])
    print(f"Added batch {i//batch_size + 1} / {len(all_chunks)//batch_size + 1}")

# -------------------- Save FAISS Index and Metadata --------------------
faiss.write_index(index, "/content/drive/MyDrive/Project/Vector_DB/law_chunks_matryoshka.index")

with open("/content/drive/MyDrive/Project/Vector_DB/metadata_store_matryoshka.pkl", "wb") as f:
    pickle.dump(metadata_store, f)

print("FAISS index and metadata saved successfully.")



# -------------------- Parameters --------------------
top_k = 50   # number of initial FAISS candidates
top_n = 5    # number of final top results
alpha = 0.7  # weight of Cross-Encoder relative to FAISS (0-1)

# -------------------- Load Cross-Encoder Model --------------------
cross_model = CrossEncoder("Omartificial-Intelligence-Space/ARA-Reranker-V1")

# -------------------- Prepare Candidate Chunks --------------------
candidate_idxs = I[0][:top_k]
candidate_texts = [all_chunks[idx]['text'] for idx in candidate_idxs]
candidate_meta = [metadata_store[idx] for idx in candidate_idxs]
candidate_distances = D[0][:top_k]  # FAISS similarity scores

# -------------------- Normalize Candidate Texts --------------------
candidate_texts_norm = [normalize_arabic(text) for text in candidate_texts]

# -------------------- Compute Cross-Encoder Scores --------------------
pairs = [[query_norm, text] for text in candidate_texts_norm]
cross_scores = cross_model.predict(pairs, show_progress_bar=True)
cross_scores = np.array(cross_scores, dtype=float)

# -------------------- Compute FAISS Ranks --------------------
# higher distance = higher similarity, so descending order
sorted_idx = np.argsort(-candidate_distances)
faiss_ranks = np.empty_like(sorted_idx)
for rank, idx in enumerate(sorted_idx):
    faiss_ranks[idx] = rank + 1  # 1-based ranking

# Alternative tie-safe ranking:
# faiss_ranks = rankdata(-candidate_distances, method='min')

# -------------------- Compute Cross-Encoder Ranks --------------------
cross_ranks = rankdata(-cross_scores, method='min')  # higher score = better rank

# -------------------- Weighted Combined Rank --------------------
combined_rank = alpha * cross_ranks + (1 - alpha) * faiss_ranks

# -------------------- Select Top-N Results --------------------
order = np.argsort(combined_rank)
selected = order[:top_n]

# -------------------- Prepare Final Results --------------------
reranked_results = []
for rank, i in enumerate(selected, start=1):
    reranked_results.append({
        "rank": rank,
        "text": candidate_texts[i],
        "metadata": candidate_meta[i],
        "faiss_distance": candidate_distances[i],
        "cross_score": float(cross_scores[i]),
        "faiss_rank": int(faiss_ranks[i]),
        "cross_rank": int(cross_ranks[i]),
        "combined_rank": float(combined_rank[i])
    })

# -------------------- Display Reranked Results --------------------
print(f"\n===== RERANKED TOP {top_n} RESULTS (Weighted Rank, Stable FAISS) =====\n")
for item in reranked_results:
    print(f"[Rank {item['rank']}] Combined Rank: {item['combined_rank']:.2f} | "
          f"FAISS Rank: {item['faiss_rank']} | Cross Rank: {item['cross_rank']} | "
          f"Cross Score: {item['cross_score']:.4f} | FAISS Distance: {item['faiss_distance']:.4f}")
    print(f"Text Preview: {item['text'][:300].replace(chr(10), ' ')}...")
    print(f"Metadata: {item['metadata']}")
    print("-"*80)



import os
from dotenv import load_dotenv
from ollama import Client
from google.colab import drive

# -------------------- Mount Google Drive --------------------
drive.mount('/content/drive')
env_path = "/content/drive/MyDrive/Project/.env"
load_dotenv(env_path)

# -------------------- Load API credentials --------------------
API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL = os.getenv("OLLAMA_MODEL")

print("API_KEY:", API_KEY[:4], "...")  # show only first 4 chars
print("MODEL:", MODEL)

# -------------------- Initialize Ollama client --------------------
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

# -------------------- Function to query LLM --------------------
def llm_answer(question: str, reranked_results, top_k: int = 5) -> str:
    """
    Send a legal question and reranked results to the LLM.
    
    Parameters:
    - question: str, the user's legal question (Arabic)
    - reranked_results: list of dicts, output from weighted FAISS + Cross-Encoder reranking
    - top_k: number of top results to include in context
    
    Returns:
    - LLM-generated answer (str) with source citations.
    """
    # Select top-k documents
    selected_docs = reranked_results[:top_k]

    # Build context from reranked results
    context_blocks = []
    for item in selected_docs:
        block = (
            f"المرتبة: {item['rank']}\n"
            f"درجة الصلة: {item['combined_rank']:.4f}\n"
            f"المصدر: {item['metadata']}\n"
            f"النص:\n{item['text']}"
        )
        context_blocks.append(block)

    context = "\n\n======================\n\n".join(context_blocks)

    # Construct the prompt with detailed instructions
    prompt = (
        "أنت مساعد قانوني متخصص للإجابة على الأسئلة القانونية بدقة واحترافية. "
        "اتبع التعليمات التالية بدقة:\n"
        "1. اقرأ جميع الإجابات المسترجعة المرتبة حسب الصلة.\n"
        "2. حدد الإجابة التي تجيب فعلياً على السؤال، وليس فقط الأعلى تشابهًا أو درجة صلة. "
        "قبل اختيار أي إجابة، تحقق أن المعلومات فيها تجيب بشكل دقيق على السؤال، حتى لو كان لها درجة صلة أقل. "
        "إذا لم تجد أي إجابة دقيقة، أجب: 'متأسف، لم أجد أي إجابة مناسبة.'\n"
        "3. إذا كانت هناك أكثر من إجابة صحيحة، دمج المعلومات منها لصياغة رد رسمي وواضح، مدمج، وجاهز للقراءة. "
        "تجنب اختصار زائد الذي قد يحذف المعلومات الأساسية.\n"
        "4. أدرج المصادر لكل معلومة باستخدام بيانات التعريف (metadata) لكل إجابة، دائمًا بهذا التنسيق:\n"
        "   المصدر: [URL] (المرتبة: [Rank]، درجة الصلة: [Relevance Score])\n"
        "5. تجنب الهذيان أو إضافة معلومات غير موجودة في الإجابات المسترجعة.\n\n"
        "=== السياق ===\n"
        f"{context}\n\n"
        "=== السؤال ===\n"
        f"{question}\n\n"
        "=== الإجابة ===\n"
        "=== المصادر ===\n"
    )

    # Send prompt to Ollama LLM
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]



#question = "مع من وقعت سلطنة عمان الاتفاقية حول الإعفاء المتبادل من التأشيرات، وفي أي مدينة تم التوقيع؟"
#answer = llm_answer(question, reranked_results, top_k=10)
#print(answer)