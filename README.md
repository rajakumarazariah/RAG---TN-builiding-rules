# TN Building Rules RAG — Q&A System

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you ask natural language questions about the **Tamil Nadu Combined Development and Building Rules, 2019** and get concise, document-grounded answers.

---

##  Short Description

This notebook ingests the Tamil Nadu Building Rules PDF, splits it into semantically meaningful chunks, embeds them using a sentence-transformer model, and stores them in a FAISS vector index. At query time it uses **hybrid retrieval** (keyword + vector search fused via Reciprocal Rank Fusion) to find the most relevant rule excerpts, then passes them as context to a local LLM (Microsoft Phi via Ollama) to generate an answer.

---

## 🗂️ Project Structure

```
project/
├── Untitled3.ipynb          # Main notebook (this file)
├── tn_building_rules.pdf    # Source PDF (Tamil Nadu Building Rules 2019)
├── faiss_tn.bin             # Saved FAISS index (auto-generated)
└── chunks_tn.pkl            # Saved chunk list (auto-generated)
```

---

##  How It Works — Step by Step

### Step 1 — PDF Ingestion
- Uses `pypdf.PdfReader` to extract text from all 248 pages of the PDF.
- Joins all page text into a single string (~509,000 characters).

### Step 2 — Smart Chunking
The document has a known three-level hierarchy:
- **PART I / II / III …** — major sections
- **1. Short title / 2. Definitions …** — individual rules
- **(1) "Access" means … / (2) …** — definition sub-items

The chunking strategy:
1. **Skips the Table of Contents** by finding the first `RULES\nPART` marker.
2. **Splits on numbered rule headers** using a regex (`27. Requirement for site…`).
3. For **Rule 2 (Definitions)**, further splits on numbered sub-items `(1)`, `(2)`, … to keep each definition atomic.
4. For **long rules (>900 chars)**: applies a sliding window (window=900, overlap=150) to avoid exceeding the LLM context.
5. **Prefixes every chunk** with its current `[PART]` label for contextual grounding.

This produces **703 chunks** in total.

### Step 3 — Embedding & FAISS Index
- Loads `sentence-transformers/all-MiniLM-L6-v2` (lightweight 384-dim model).
- Encodes all chunks into float32 embeddings.
- L2-normalises embeddings to enable **cosine similarity** via `faiss.IndexFlatIP`.
- Saves the index (`faiss_tn.bin`) and chunks (`chunks_tn.pkl`) to disk — subsequent runs skip re-embedding automatically.

### Step 4 — Hybrid Retrieval
The `retrieve(query, k)` function uses **Reciprocal Rank Fusion (RRF)** to blend two signals:

| Signal | Method |
|---|---|
| **Keyword score** | Counts exact query-word hits per chunk, length-normalised |
| **Vector score** | Cosine similarity via FAISS inner product |

Both signals are ranked independently, then fused:

```
RRF(idx) = 1 / (60 + kw_rank[idx]) + 1 / (60 + vec_rank[idx])
```

The top-k chunks by RRF score are returned.

### Step 5 — Answer Generation
The `ask(query)` function:
1. Retrieves top-4 chunks.
2. Truncates each to 700 characters to reduce RAM pressure.
3. Builds a strict prompt instructing the LLM to answer **only from the provided context**.
4. Calls `ollama.chat(model='phi', ...)` with `temperature=0.1` (deterministic) and `num_predict=300` (concise).
5. Prints retrieved chunks (for transparency) and the final answer.

### Step 6 — Interactive Q&A Loop
Runs a `while True` input loop in the terminal / Jupyter cell. Type your question and get an answer. Type `q`, `quit`, or `exit` to stop.

---

## 🔧 Requirements

### Python Packages
```
pypdf
numpy
faiss-cpu        # or faiss-gpu
sentence-transformers
ollama
```

Install with:
```bash
pip install pypdf numpy faiss-cpu sentence-transformers ollama
```

### Local LLM via Ollama
This notebook uses **Microsoft Phi** served locally through [Ollama](https://ollama.com).

1. Install Ollama: https://ollama.com/download
2. Pull the model:
   ```bash
   ollama pull phi
   ```
3. Ensure Ollama is running before executing the notebook.

### Input File
Place the PDF at the path specified in the notebook:
```python
PDF_PATH = "tn_building_rules.pdf"
```
Make sure it is in the same directory as the notebook (or update the path).

---

## 🚀 Usage

1. Clone / copy the notebook into your working directory along with `tn_building_rules.pdf`.
2. Install all dependencies above.
3. Start Ollama and pull the `phi` model.
4. Open the notebook in Jupyter and **Run All**.
   - First run: builds embeddings and saves index (~1–2 min depending on hardware).
   - Subsequent runs: loads the saved index instantly.
5. When prompted, type a question:
   ```
   Question: what is FSI?
   Question: what are the setback requirements?
   Question: q        ← to exit
   ```

---

##  Example Q&A

**Question:** `what is fsi?`

**Retrieved chunks:** Rule 49 (Premium FSI) and Part VII FSI Tolerance Limit sections.

**Answer:**
> FSI stands for Floor Space Index. It's a measure used to determine the amount of usable space within a building relative to its total area. In Tamil Nadu, specific rules govern FSI limits based on road width, and Premium FSI charges apply when these limits are exceeded.

---

##  Design Decisions

| Decision | Reason |
|---|---|
| `all-MiniLM-L6-v2` | Small, fast, good quality for English legal text |
| `IndexFlatIP` + L2 norm | Exact cosine similarity; sufficient for 703 chunks |
| Hybrid RRF retrieval | Pure vector search misses exact legal terms; keyword search alone lacks semantic understanding |
| Phi via Ollama | Fully local — no API key, no data leaves the machine |
| `temperature=0.1` | Legal Q&A needs deterministic, factual answers |
| Chunk prefixed with PART label | Gives the LLM section context without needing the full document |

---

## ⚠️ Limitations

- Answers are limited to what is retrievable from the 703 chunks — very broad questions may miss context spread across many rules.
- The Phi model is small; for more complex legal reasoning a larger model (e.g., `llama3`, `mistral`) via Ollama would yield better results.
- Scanned/image pages in the PDF will not be extracted (text-only extraction).
- No HuggingFace token is set; rate limits may apply when downloading the embedding model for the first time.

---

## 📄 Source Document

**Tamil Nadu Combined Development and Building Rules, 2019**
Published by the Government of Tamil Nadu, Municipal Administration and Water Supply Department (G.O.(Ms) No.18, February 2019).
