# test_search.py
# Quick exerciser for POST /search with optional geo enrichment per place mention
#
# Examples:
#   python test_search.py --base http://127.0.0.1:8000
#   python test_search.py --base http://127.0.0.1:8000 --dataset water_main_breaks --topk 8 --geo
#   python test_search.py --base http://127.0.0.1:8000 --dataset water_main_breaks --kind feature --radius 0.8 --geo-weight 0.5
#   python test_search.py --base http://127.0.0.1:8000 --from 2024-01-01T00:00:00 --to 2024-12-31T23:59:59
#
import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def trunc(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def print_results(label: str, res_json: Any, max_rows: int = 12) -> None:
    print(f"\n=== {label} ===")
    # Some servers wrap: {"data": [...]}
    if isinstance(res_json, dict) and isinstance(res_json.get("data"), list):
        rows = res_json["data"][:max_rows]
    elif isinstance(res_json, list):
        rows = res_json[:max_rows]
    else:
        print(pretty(res_json))
        return

    if not rows:
        print("(no results)")
        return

    for i, r in enumerate(rows, 1):
        score = r.get("score") or r.get("vscore") or r.get("lscore") or ""
        vscore = r.get("vscore")
        lscore = r.get("lscore")
        geo_boost = r.get("geo_boost")
        dist_km = r.get("dist_km")
        ds = r.get("dataset_slug") or ""
        kind = r.get("kind") or ""
        rid = r.get("id") or ""
        txt = trunc(r.get("text") or "")

        line = f"{i:>2}. score={score!s:>8}  kind={kind:<8}  ds={ds:<24}  id={rid}"
        if vscore is not None or lscore is not None:
            line += f"  (v={vscore!s} l={lscore!s})"
        if dist_km is not None:
            try:
                line += f"  dist={float(dist_km):.3f}km"
            except Exception:
                line += f"  dist={dist_km}"
        if geo_boost is not None:
            try:
                line += f"  gboost={float(geo_boost):.3f}"
            except Exception:
                line += f"  gboost={geo_boost}"
        print(line)
        print(f"    {txt}")


def run_search(sess: requests.Session, base: str, body: Dict[str, Any], label: str) -> None:
    url = f"{base}/search"
    t0 = time.time()
    try:
        print(body)
        resp = sess.post(url, json=body, timeout=180)
    except Exception as e:
        print(f"\nPOST {url}  --> request error: {e}")
        return
    dt = time.time() - t0
    print(f"\nPOST {url}  ({dt:.2f}s)  --> {resp.status_code}")
    try:
        js = resp.json()
    except Exception:
        print("Raw:", trunc(resp.text, 800))
        return
    if resp.ok:
        print_results(label, js)
    else:
        print(pretty(js))


# ---- Simple gazetteer for place->geo enrichment ----
# Extend these as needed. Coords are approximate; use your canonical source if available.
GAZETTEER: Dict[str, Tuple[float, float, float]] = {
    # name -> (lat, lon, default_radius_km)
    "ZELLER CRT": (43.45127, -80.42298, 0.8),   # Kitchener (approx)
    "LIBERTY AVE": (43.42464, -80.42165, 0.8),  # Kitchener (approx)
    # Add more:
    # "KING ST E": (43.45100, -80.47700, 1.0),
    # "OTTAWA ST S": (43.42130, -80.45680, 1.0),
    # "FAIRWAY RD": (43.42530, -80.43870, 1.0),
}


def _normalize(s: str) -> str:
    # Uppercase & collapse whitespace/punctuation for robust matching
    keep = []
    last_space = False
    for ch in s.upper():
        if ch.isalnum() or ch in (" ", "'", "/"):
            if ch.isspace():
                if not last_space:
                    keep.append(" ")
                last_space = True
            else:
                keep.append(ch)
                last_space = False
        else:
            if not last_space:
                keep.append(" ")
                last_space = True
    return " ".join("".join(keep).split())


def find_geo_from_query(q: str) -> Optional[Tuple[str, float, float, float]]:
    norm = _normalize(q)
    for name, (lat, lon, rkm) in GAZETTEER.items():
        if name in norm:
            return (name, lat, lon, rkm)
    return None


def maybe_inject_geo(body: Dict[str, Any], q: str, enable_geo: bool,
                     radius_override: Optional[float], geo_weight: Optional[float]) -> Optional[str]:
    """
    If enable_geo True and the query contains a known place from GAZETTEER,
    inject lat/lon/radius_km/geo_weight into body. Returns a label suffix if applied.
    """
    if not enable_geo:
        return None
    hit = find_geo_from_query(q)
    if not hit:
        return None
    name, lat, lon, rkm = hit
    body["lat"] = lat
    body["lon"] = lon
    body["radius_km"] = radius_override if (radius_override and radius_override > 0) else rkm
    if geo_weight is not None:
        body["geo_weight"] = geo_weight
    return f" +geo:{name}@({lat:.5f},{lon:.5f}) r={body['radius_km']:.2f}km"


def main():
    ap = argparse.ArgumentParser(description="Exercise /search with multiple queries (PostGIS geo-aware).")
    ap.add_argument("--base", default=os.getenv("API_BASE", "http://127.0.0.1:8000"), help="API base URL")
    ap.add_argument("--dataset", default=os.getenv("DATASET"), help="datasetSlug filter (e.g. water_main_breaks)")
    ap.add_argument("--kind", default=os.getenv("KIND", "feature"), help="kind filter (default: feature)")
    ap.add_argument("--topk", type=int, default=10, help="topK for each query")

    # Geo/time controls
    ap.add_argument("--geo", action="store_true", help="auto-inject geo for queries that mention a known place")
    ap.add_argument("--radius", type=float, default=None, help="override radius_km when --geo is used")
    ap.add_argument("--geo-weight", type=float, default=0.5, dest="geo_weight", help="distance boost weight (0..1)")
    ap.add_argument("--from", dest="from_time", type=str, default=None, help="ISO start time (e.g. 2024-01-01T00:00:00)")
    ap.add_argument("--to", dest="to_time", type=str, default=None, help="ISO end time   (e.g. 2024-12-31T23:59:59)")

    args = ap.parse_args()

    base = args.base.rstrip("/")
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})

    # ----- Define a varied set of searches -----
    tests: List[Dict[str, Any]] = []

    # 1) Hybrid: materials + cause (adds synonyms/aliases to improve recall)
    tests.append({
        "label": "Hybrid: corrosion on cast iron (CI) mains",
        "body": {
            "q": "corrosion on cast iron CI water main pipe",
            "topK": args.topk,
            "datasetSlug": args.dataset,
            "kind": args.kind,
            "hybrid": True,
            "from_time": args.from_time,
            "to_time": args.to_time,
            "geo_weight": args.geo_weight,
        }
    })

    # 2) Hybrid: exact street phrase + context (place mention -> geo auto-injected when --geo)
    tests.append({
        "label": "Hybrid: street phrase + event (ZELLER CRT)",
        "body": {
            "q": "\"ZELLER CRT\" water main break repair closure",
            "topK": args.topk,
            "datasetSlug": args.dataset,
            "kind": args.kind,
            "hybrid": True,
            "from_time": args.from_time,
            "to_time": args.to_time,
            "geo_weight": args.geo_weight,
        }
    })

    # 3) Semantic-only: pressure/failure phrasing variance near Liberty Ave (geo auto-injected when --geo)
    tests.append({
        "label": "Semantic-only: pressure-related failure near Liberty Ave",
        "body": {
            "q": "pressure related failure burst low pressure transient near Liberty Ave",
            "topK": args.topk,
            "datasetSlug": args.dataset,
            "kind": args.kind,
            "hybrid": False,
            "from_time": args.from_time,
            "to_time": args.to_time,
            "geo_weight": args.geo_weight,
        }
    })

    # 4) Hybrid: break morphology language variants
    tests.append({
        "label": "Hybrid: longitudinal / circumferential pipe break",
        "body": {
            "q": "longitudinal or circumferential break crack split along pipe",
            "topK": args.topk,
            "datasetSlug": args.dataset,
            "kind": args.kind,
            "hybrid": True,
            "from_time": args.from_time,
            "to_time": args.to_time,
            "geo_weight": args.geo_weight,
        }
    })

    # 5) Hybrid: seasonal/ground-movement cause signals (helps tease cause categories)
    tests.append({
        "label": "Hybrid: frost heave / freeze-thaw related main break",
        "body": {
            "q": "frost heave freeze thaw ground movement water main break",
            "topK": args.topk,
            "datasetSlug": args.dataset,
            "kind": args.kind,
            "hybrid": True,
            "from_time": args.from_time,
            "to_time": args.to_time,
            "geo_weight": args.geo_weight,
        }
    })

    # ----- Apply optional geo enrichment & run -----
    print(f"[i] base={base} dataset={args.dataset} kind={args.kind} topK={args.topk} "
          f"geo={'on' if args.geo else 'off'} radius={args.radius or 'default'} geo_weight={args.geo_weight} "
          f"from={args.from_time or '-'} to={args.to_time or '-'}")

    for t in tests:
        body = dict(t["body"])  # copy
        q = body.get("q", "")
        suffix = maybe_inject_geo(
            body, q,
            enable_geo=args.geo,
            radius_override=args.radius,
            geo_weight=args.geo_weight
        )
        label = t["label"] + (suffix or "")
        run_search(s, base, body, label)


if __name__ == "__main__":
    main()
