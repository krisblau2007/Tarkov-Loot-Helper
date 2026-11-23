"""Export scan results to CSV and JSON."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime

from .config import EXPORT_DIR
from .scanner import ScanResult


def export(result: ScanResult) -> tuple[str, str]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for d in result.identified:
        it = d.item
        rows.append({
            "container": d.region.grid_index,
            "col": d.region.col,
            "row": d.region.row,
            "name": it.name,
            "short_name": it.short_name,
            "size": f"{it.width}x{it.height}",
            "flea_price": it.flea_price,
            "best_trader": it.best_trader,
            "trader_price": it.best_trader_price,
            "price_per_slot": it.price_per_slot,
            "quest_needs": "; ".join(it.quest_needs),
            "hideout_needs": "; ".join(it.hideout_needs),
            "confidence": round(d.score, 3),
        })
    csv_path = EXPORT_DIR / f"scan_{stamp}.csv"
    json_path = EXPORT_DIR / f"scan_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    totals = Counter(r["name"] for r in rows)
    json_path.write_text(json.dumps({
        "scanned_at": stamp,
        "total_value": sum(r["flea_price"] or r["trader_price"] for r in rows),
        "counts": totals,
        "items": rows,
    }, indent=2), encoding="utf-8")
    return str(csv_path), str(json_path)
