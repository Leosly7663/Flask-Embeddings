# app.py
import os
import re
import time
import datetime as dt
import threading
import contextlib
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client  # pip install supabase
from dotenv import load_dotenv

# -------------------- Load env early --------------------
load_dotenv(".env.local")

# -------------------- FastAPI --------------------
app = FastAPI(title="Combine-Match API (FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Config --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

DEFAULT_MODEL = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL = os.getenv("EMBED_MODEL", DEFAULT_MODEL)

EMBED_TABLE = os.getenv("EMBED_TABLE", "rag_docs")
EMBED_ID_FIELD = os.getenv("EMBED_ID_FIELD", "id")
EMBED_TEXT_FIELD = os.getenv("EMBED_TEXT_FIELD", "text")
EMBED_HASH_FIELD = os.getenv("EMBED_HASH_FIELD", "text_hash")
EMBED_EMBED_FIELD = os.getenv("EMBED_EMBED_FIELD", "embedding")
EMBED_MODEL_FIELD = os.getenv("EMBED_MODEL_FIELD", "embed_model")
EMBED_HASH_TRACK_FIELD = os.getenv("EMBED_HASH_TRACK_FIELD", "embedding_text_hash")
EMBED_UPDATED_FIELD = os.getenv("EMBED_UPDATED_FIELD", "embed_updated_at")
EMBED_DATASET_SLUG = os.getenv("EMBED_DATASET_SLUG")  # optional filter (dataset_slug)

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
EMBED_FETCH_CAP = int(os.getenv("EMBED_FETCH_CAP", "500"))         # rows per scan page
EMBED_ENCODE_BATCH = int(os.getenv("EMBED_ENCODE_BATCH", "256"))
EMBED_INTERVAL_SEC = int(os.getenv("EMBED_INTERVAL_SEC", "3600"))  # 1 hour
EMBED_NORMALIZE = bool(int(os.getenv("EMBED_NORMALIZE", "0")))     # 1/0

# -------------------- Global clients (init in lifespan) --------------------
supabase: Optional[Client] = None

# -------------------- Model management (thread-safe, lazy) --------------------
MODEL_REGISTRY: Dict[str, SentenceTransformer] = {}
MODEL_LOCK = threading.RLock()
CURRENT_MODEL_NAME: Optional[str] = None  # lazy init


def _load_model(name: str) -> SentenceTransformer:
    t0 = time.time()
    m = SentenceTransformer(name)
    print(f"[model] loaded '{name}' in {time.time()-t0:.2f}s", flush=True)
    return m

def _ensure_and_get_model(name: Optional[str]) -> Tuple[str, SentenceTransformer]:
    global CURRENT_MODEL_NAME
    with MODEL_LOCK:
        resolved = name or CURRENT_MODEL_NAME or DEFAULT_MODEL
        if CURRENT_MODEL_NAME is None:
            CURRENT_MODEL_NAME = resolved
        if resolved in MODEL_REGISTRY:
            return resolved, MODEL_REGISTRY[resolved]
    m = _load_model(resolved)  # load outside lock
    with MODEL_LOCK:
        MODEL_REGISTRY.setdefault(resolved, m)
        return resolved, MODEL_REGISTRY[resolved]

def _set_current_model(name: str) -> Dict[str, Any]:
    global CURRENT_MODEL_NAME
    with MODEL_LOCK:
        if name in MODEL_REGISTRY:
            CURRENT_MODEL_NAME = name
            return {"current": CURRENT_MODEL_NAME, "loaded": sorted(MODEL_REGISTRY.keys())}
    m = _load_model(name)
    with MODEL_LOCK:
        MODEL_REGISTRY.setdefault(name, m)
        CURRENT_MODEL_NAME = name
        return {"current": CURRENT_MODEL_NAME, "loaded": sorted(MODEL_REGISTRY.keys())}

async def ensure_and_get_model(name: Optional[str]) -> Tuple[str, SentenceTransformer]:
    return await anyio.to_thread.run_sync(_ensure_and_get_model, name)

async def set_current_model(name: str) -> Dict[str, Any]:
    return await anyio.to_thread.run_sync(_set_current_model, name)

# -------------------- Text helpers --------------------
def split_sentences(text: str) -> List[str]:
    raw = re.split(r"[.\n]+", text)
    return [s.strip() for s in raw if s.strip()]

def combine_sentences(sentences: List[str]) -> List[str]:
    combos: List[str] = []
    for i, s in enumerate(sentences):
        combos.append(s)
        if i < len(sentences) - 1:
            combos.append(f"{s}. {sentences[i+1]}")
    return combos

# -------------------- API Schemas --------------------
class ModelRequest(BaseModel):
    name: str = Field(..., description="Hugging Face or local model path")
    makeCurrent: bool = Field(True, description="If true, set as default for future requests")

class MatchItem(BaseModel):
    sentence: str
    similarityPercent: float

class CombineMatchRequest(BaseModel):
    resume: str
    jobSentences: List[str]
    modelName: Optional[str] = Field(None, description="Optional per-request model override")
    normalize: bool = False
    topK: int = Field(1, ge=1, description="Return top-K matches per job sentence")
    maxResumeSentences: int = Field(250, ge=1, description="Guard for resume length")

class CombineMatchResponseItem(BaseModel):
    jobSentence: str
    bestMatchSentence: Optional[str] = None
    similarityPercent: Optional[float] = None
    matches: Optional[List[MatchItem]] = None
    modelUsed: str

# --- Embedding admin request/response models ---
class RestartAllRequest(BaseModel):
    datasetSlug: Optional[str] = Field(None, description="Override ENV EMBED_DATASET_SLUG")
    limit: Optional[int] = Field(None, ge=1, description="Max rows to re-embed in this call")
    pageSize: Optional[int] = Field(None, ge=1, description="Page size for scanning (defaults to EMBED_FETCH_CAP)")

class RestartOneResponse(BaseModel):
    ok: bool
    updated: int
    id: str

# --- Search request model (geo/time aware) ---
class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, description="Query text")
    topK: int = Field(10, ge=1, le=100)
    datasetSlug: Optional[str] = Field(None, description="Dataset filter")
    kind: Optional[str] = Field(None, description="Kind filter, e.g. 'feature'")
    hybrid: bool = Field(True, description="True = hybrid (semantic + lexical), False = semantic-only")
    # optional time window
    from_time: Optional[dt.datetime] = None
    to_time: Optional[dt.datetime] = None
    # optional geo circle
    lat: Optional[float] = Field(None, description="Latitude (-90..90)")
    lon: Optional[float] = Field(None, description="Longitude (-180..180)")
    radius_km: Optional[float] = Field(None, description="Search radius in km (>0)")
    # distance boost strength: 0..1 (0 = no boost; 0.5 is a good start)
    geo_weight: Optional[float] = 0.5

def _as_iso(ts: Optional[dt.datetime]) -> Optional[str]:
    return ts.isoformat() if ts else None

# -------------------- Scan helpers (no .eq/.filter needed) --------------------
def _select_csv(cols: List[str]) -> str:
    # Keep order stable
    return ",".join(cols)

def _scan_pages(columns: List[str], page_size: int, dataset_slug: Optional[str] = None):
    """
    Yields pages (lists of rows) using only .select().range().execute().
    Applies dataset_slug filter client-side for compatibility with older clients.
    """
    assert supabase is not None, "Supabase client not initialized"
    offset = 0
    csv = _select_csv(columns)
    while True:
        res = supabase.table(EMBED_TABLE).select(csv).range(offset, offset + page_size - 1).execute()
        page = res.data or []
        if dataset_slug or EMBED_DATASET_SLUG:
            ds = dataset_slug or EMBED_DATASET_SLUG
            page = [r for r in page if r.get("dataset_slug") == ds]
        if not page:
            break
        yield page
        if len(page) < page_size:
            break
        offset += len(page)

# -------------------- Embedding Worker (Supabase) --------------------
def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def _filter_needing_embed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only rows missing/outdated embeddings."""
    out = []
    for r in rows:
        emb       = r.get(EMBED_EMBED_FIELD)
        emb_model = r.get(EMBED_MODEL_FIELD)
        emb_hash  = r.get(EMBED_HASH_TRACK_FIELD)
        txt_hash  = r.get(EMBED_HASH_FIELD)
        if (emb is None) or (emb_model != EMBED_MODEL) or (emb_hash is None) or (txt_hash is None) or (emb_hash != txt_hash):
            out.append(r)
    return out

async def _fetch_batch_from_supabase() -> List[Dict[str, Any]]:
    # Scan pages and collect up to EMBED_BATCH rows that need work.
    cols = [
        EMBED_ID_FIELD, EMBED_TEXT_FIELD, EMBED_HASH_FIELD,
        EMBED_EMBED_FIELD, EMBED_MODEL_FIELD, EMBED_HASH_TRACK_FIELD,
        "dataset_slug",
        "kind",
    ]
    collected: List[Dict[str, Any]] = []
    for page in _scan_pages(cols, EMBED_FETCH_CAP, dataset_slug=None):
        need = _filter_needing_embed(page)
        if EMBED_DATASET_SLUG:
            need = [r for r in need if r.get("dataset_slug") == EMBED_DATASET_SLUG]
        collected.extend(need)
        if len(collected) >= EMBED_BATCH:
            break
    return collected[:EMBED_BATCH]

async def _embed_and_update(rows: List[Dict[str, Any]], model: SentenceTransformer):
    assert supabase is not None, "Supabase client not initialized"
    if not rows:
        return 0

    # 1) Encode texts
    texts = [ (r.get(EMBED_TEXT_FIELD) or "") for r in rows ]
    vectors: List[List[float]] = []
    for i in range(0, len(texts), EMBED_ENCODE_BATCH):
        chunk = texts[i:i+EMBED_ENCODE_BATCH]
        arr = await anyio.to_thread.run_sync(
            lambda c=chunk: model.encode(c, convert_to_numpy=True, normalize_embeddings=EMBED_NORMALIZE)
        )
        vectors.extend(arr.tolist())

    # 2) Build UPSERT payload; include required fields to satisfy INSERT
    now_iso = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

    payload = []
    for r, vec in zip(rows, vectors):
        # Pull existing values from the scanned row
        row_id       = r.get(EMBED_ID_FIELD)
        row_text     = r.get(EMBED_TEXT_FIELD) or ""           # text NOT NULL
        row_kind     = r.get("kind") or "feature"              # kind NOT NULL; safe default
        row_dataset  = r.get("dataset_slug")                   # optional; include if you have it
        row_txt_hash = r.get(EMBED_HASH_FIELD)

        item = {
            EMBED_ID_FIELD: row_id,
            # required columns for insert:
            EMBED_TEXT_FIELD: row_text,
            "kind": row_kind,
            "dataset_slug": row_dataset,

            # embedding fields we’re updating:
            EMBED_EMBED_FIELD: vec,
            EMBED_MODEL_FIELD: EMBED_MODEL,
            EMBED_HASH_TRACK_FIELD: row_txt_hash,
            EMBED_UPDATED_FIELD: now_iso,
        }
        payload.append(item)

    # 3) Upsert by primary key. If the id exists -> UPDATE; otherwise -> INSERT with required fields.
    supabase.table(EMBED_TABLE).upsert(
        payload,
        on_conflict=EMBED_ID_FIELD,
        returning="minimal",
    ).execute()

    print(f"[embed] updated {len(payload)} rows", flush=True)
    return len(payload)

async def embedding_daemon():
    """Hourly refresher loop."""
    while True:
        try:
            model_name, model = await ensure_and_get_model(EMBED_MODEL)
            total = 0
            while True:
                batch = await _fetch_batch_from_supabase()
                if not batch:
                    break
                n = await _embed_and_update(batch, model)
                total += n
            print(f"[embed] pass complete (model={model_name}) total={total}", flush=True)
        except Exception as e:
            print(f"[embed] ERROR: {e}", flush=True)
        await asyncio.sleep(EMBED_INTERVAL_SEC)

# -------------------- Admin: restart all / one --------------------
async def _fetch_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    cols = [
        EMBED_ID_FIELD, EMBED_TEXT_FIELD, EMBED_HASH_FIELD,
        EMBED_EMBED_FIELD, EMBED_MODEL_FIELD, EMBED_HASH_TRACK_FIELD,
        "dataset_slug",
    ]
    for page in _scan_pages(cols, EMBED_FETCH_CAP, dataset_slug=None):
        for r in page:
            if str(r.get(EMBED_ID_FIELD)) == str(doc_id):
                if EMBED_DATASET_SLUG and r.get("dataset_slug") != EMBED_DATASET_SLUG:
                    continue
                return r
    return None

@app.post("/embed/restart_all")
async def restart_all(req: RestartAllRequest):
    """
    Force re-embed for all rows (optionally scoped to datasetSlug).
    Streams pages to avoid large payloads.
    """
    try:
        model_name, model = await ensure_and_get_model(EMBED_MODEL)
        page_size = int(req.pageSize or EMBED_FETCH_CAP)
        max_rows  = int(req.limit) if req.limit else None

        total = 0
        t0 = time.time()

        cols = [
            EMBED_ID_FIELD, EMBED_TEXT_FIELD, EMBED_HASH_FIELD,
            EMBED_EMBED_FIELD, EMBED_MODEL_FIELD, EMBED_HASH_TRACK_FIELD,
            "dataset_slug",
        ]
        for page in _scan_pages(cols, page_size, dataset_slug=req.datasetSlug):
            # respect limit if provided
            work_page = page
            if max_rows is not None and total + len(work_page) > max_rows:
                work_page = work_page[: max_rows - total]

            n = await _embed_and_update(work_page, model)
            total += n

            if max_rows is not None and total >= max_rows:
                break

        return {
            "ok": True,
            "model": model_name,
            "updated": total,
            "elapsed_sec": round(time.time() - t0, 2),
            "dataset": req.datasetSlug or EMBED_DATASET_SLUG,
            "page_size": page_size,
            "limit": max_rows,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/embed/restart_one/{doc_id}", response_model=RestartOneResponse)
async def restart_one(doc_id: str):
    """Force re-embed a single document by ID."""
    try:
        row = await _fetch_by_id(doc_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"Row not found for {EMBED_ID_FIELD}='{doc_id}'")
        model_name, model = await ensure_and_get_model(EMBED_MODEL)
        updated = await _embed_and_update([row], model)
        return RestartOneResponse(ok=True, updated=updated, id=doc_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# -------------------- Debug: counts as pure JSON (scan-based) --------------------
@app.get("/embed/debug")
async def embed_debug(datasetSlug: Optional[str] = None, scan_page: int = Query(1000, ge=50, le=10000)):
    """
    Returns counts only:
    {
      "total": ..., "no_vec": ..., "wrong_model": ...,
      "miss_track_hash": ..., "miss_text_hash": ..., "stale_hash": ...
    }
    """
    try:
        cols = [
            EMBED_EMBED_FIELD, EMBED_MODEL_FIELD,
            EMBED_HASH_TRACK_FIELD, EMBED_HASH_FIELD,
            "dataset_slug",
        ]
        total = no_vec = wrong_model = miss_track_hash = miss_text_hash = stale_hash = 0

        for page in _scan_pages(cols, scan_page, dataset_slug=datasetSlug):
            total += len(page)
            for r in page:
                emb       = r.get(EMBED_EMBED_FIELD)
                emb_model = r.get(EMBED_MODEL_FIELD)
                emb_hash  = r.get(EMBED_HASH_TRACK_FIELD)
                txt_hash  = r.get(EMBED_HASH_FIELD)

                if emb is None: no_vec += 1
                if emb_model != EMBED_MODEL: wrong_model += 1
                if emb_hash is None: miss_track_hash += 1
                if txt_hash is None: miss_text_hash += 1
                if (emb_hash is not None) and (txt_hash is not None) and (emb_hash != txt_hash):
                    stale_hash += 1

        return {
            "supabase" : SUPABASE_URL,
            "total": total,
            "no_vec": no_vec,
            "wrong_model": wrong_model,
            "miss_track_hash": miss_track_hash,
            "miss_text_hash": miss_text_hash,
            "stale_hash": stale_hash,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# -------------------- Public endpoints: model + matching --------------------
@app.get("/model")
def get_model_status():
    with MODEL_LOCK:
        return {
            "current": CURRENT_MODEL_NAME or DEFAULT_MODEL,
            "loaded": sorted(MODEL_REGISTRY.keys()),
            "default_env": DEFAULT_MODEL,
            "embed_worker": {
                "model": EMBED_MODEL,
                "batch": EMBED_BATCH,
                "encode_batch": EMBED_ENCODE_BATCH,
                "interval_sec": EMBED_INTERVAL_SEC,
                "normalize": EMBED_NORMALIZE,
                "table": EMBED_TABLE,
            }
        }

@app.post("/model")
async def post_model(req: ModelRequest):
    try:
        if req.makeCurrent:
            info = await set_current_model(req.name)
        else:
            await anyio.to_thread.run_sync(_ensure_and_get_model, req.name)  # warm-load
            with MODEL_LOCK:
                info = {"current": CURRENT_MODEL_NAME or DEFAULT_MODEL, "loaded": sorted(MODEL_REGISTRY.keys())}
        return {"ok": True, **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load model '{req.name}': {e}") from e

@app.post("/combine-match", response_model=List[CombineMatchResponseItem])
async def combine_similarity(req: CombineMatchRequest):
    if not req.resume or not req.jobSentences:
        raise HTTPException(status_code=400, detail="resume and jobSentences are required")
    if not all(isinstance(s, str) and s.strip() for s in req.jobSentences):
        raise HTTPException(status_code=400, detail="jobSentences must be non-empty strings")

    resolved_name, model = await ensure_and_get_model(req.modelName)

    base_sentences = split_sentences(req.resume)
    if req.maxResumeSentences and len(base_sentences) > req.maxResumeSentences:
        base_sentences = base_sentences[: req.maxResumeSentences]
    combined_sentences = combine_sentences(base_sentences)
    if not combined_sentences:
        raise HTTPException(status_code=400, detail="no sentences found in resume")

    combined_embeddings = await anyio.to_thread.run_sync(
        lambda: model.encode(combined_sentences, convert_to_tensor=True, normalize_embeddings=req.normalize)
    )
    job_embeddings = await anyio.to_thread.run_sync(
        lambda: model.encode(req.jobSentences, convert_to_tensor=True, normalize_embeddings=req.normalize)
    )

    results: List[CombineMatchResponseItem] = []
    for i, job_sentence in enumerate(req.jobSentences):
        sims = util.cos_sim(job_embeddings[i], combined_embeddings)[0]
        if req.topK == 1:
            best_idx = int(sims.argmax().item())
            best_score = float(sims[best_idx].item())
            results.append(
                CombineMatchResponseItem(
                    jobSentence=job_sentence,
                    bestMatchSentence=combined_sentences[best_idx],
                    similarityPercent=round(best_score * 100, 2),
                    modelUsed=resolved_name,
                )
            )
        else:
            k = min(req.topK, len(combined_sentences))
            topk_scores, topk_idx = sims.topk(k)
            items = [
                MatchItem(sentence=combined_sentences[int(idx)], similarityPercent=round(float(score) * 100, 2))
                for score, idx in zip(topk_scores.tolist(), topk_idx.tolist())
            ]
            results.append(CombineMatchResponseItem(jobSentence=job_sentence, matches=items, modelUsed=resolved_name))
    return results

# -------------------- NEW: Geo-aware /search endpoint --------------------
@app.post("/search")
async def search(req: SearchRequest):
    if not req.q.strip():
        raise HTTPException(400, "q required")

    # 1) embed the query
    try:
        _, model = await ensure_and_get_model(EMBED_MODEL)
        q_vec = await anyio.to_thread.run_sync(
            lambda: model.encode([req.q], convert_to_numpy=True, normalize_embeddings=EMBED_NORMALIZE)
        )
        emb = q_vec[0].tolist()
    except Exception as e:
        raise HTTPException(500, f"embedding error: {e}")

    # 2) build common geo/time args (PostGIS-based SQL functions)
    common_geo: Dict[str, Any] = {
        "from_time": _as_iso(req.from_time),
        "to_time": _as_iso(req.to_time),
        "lat0": req.lat,
        "lon0": req.lon,
        "radius_km": req.radius_km,
        "geo_weight": req.geo_weight,
    }

    # 3) call RPC
    try:
        if req.hybrid:
            payload = {
                "q": req.q,
                "query_embedding": emb,
                "match_count": req.topK,
                "dataset": req.datasetSlug,
                "kind_filter": req.kind,
                **common_geo,
            }
            res = supabase.rpc("search_rag_docs_hybrid", payload).execute()
            return res.data or []
        else:
            payload = {
                "query_embedding": emb,
                "match_count": req.topK,
                "dataset": req.datasetSlug,
                "kind_filter": req.kind,
                **common_geo,
            }
            res = supabase.rpc("match_rag_docs", payload).execute()
            return res.data or []
    except Exception as e:
        raise HTTPException(500, f"search error: {e}")

# -------------------- Health --------------------
@app.get("/health")
def health():
    with MODEL_LOCK:
        return {"ok": True, "current": CURRENT_MODEL_NAME or DEFAULT_MODEL}

# -------------------- Lifespan (no deprecated on_event) --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # (optional) eager-load default runtime model:
    # await set_current_model(DEFAULT_MODEL)

    task = asyncio.create_task(embedding_daemon())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

# Wire lifespan for ASGI servers
app.router.lifespan_context = lifespan

# -------------------- Dev entry --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
