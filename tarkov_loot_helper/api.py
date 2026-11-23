"""Item, price, quest and hideout data from the public tarkov.dev GraphQL API."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests

from .config import DATA_DIR

API_URL = "https://api.tarkov.dev/graphql"
CACHE_MAX_AGE = 60 * 60  # refresh prices hourly

ITEMS_QUERY = """
query Items($mode: GameMode) {
  items(gameMode: $mode) {
    id name shortName width height gridImageLink avg24hPrice basePrice
    sellFor { priceRUB vendor { name } }
  }
}"""

NEEDS_QUERY = """
query Needs($mode: GameMode) {
  tasks(gameMode: $mode) {
    name trader { name }
    objectives { ... on TaskObjectiveItem { type count foundInRaid items { id } } }
  }
  hideoutStations(gameMode: $mode) {
    name
    levels { level itemRequirements { count item { id } } }
  }
}"""


@dataclass
class Item:
    id: str
    name: str
    short_name: str
    width: int
    height: int
    grid_image: str | None
    flea_price: int
    best_trader: str
    best_trader_price: int
    quest_needs: list[str] = field(default_factory=list)
    hideout_needs: list[str] = field(default_factory=list)

    @property
    def best_price(self) -> int:
        return max(self.flea_price, self.best_trader_price)

    @property
    def price_per_slot(self) -> int:
        return self.best_price // max(1, self.width * self.height)


def _post(query: str, mode: str) -> dict:
    r = requests.post(API_URL, json={"query": query, "variables": {"mode": mode}}, timeout=60)
    try:
        payload = r.json()
    except ValueError:
        r.raise_for_status()
        raise
    if payload.get("errors") and not payload.get("data"):
        err = payload["errors"][0]
        raise RuntimeError(f"tarkov.dev API error: {err.get('message') if isinstance(err, dict) else err}")
    r.raise_for_status()
    return payload["data"]


def _cached(name: str, query: str, mode: str, force: bool) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}_{mode}.json"
    if not force and path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        data = _post(query, mode)
    except Exception:
        if path.exists():  # offline: fall back to stale cache
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def load_items(mode: str = "regular", force: bool = False) -> dict[str, Item]:
    raw = _cached("items", ITEMS_QUERY, mode, force)["items"]
    items: dict[str, Item] = {}
    for it in raw:
        traders = [s for s in it.get("sellFor") or [] if s["vendor"]["name"] != "Flea Market"]
        best = max(traders, key=lambda s: s["priceRUB"], default=None)
        items[it["id"]] = Item(
            id=it["id"],
            name=it["name"],
            short_name=it["shortName"],
            width=it["width"] or 1,
            height=it["height"] or 1,
            grid_image=it.get("gridImageLink"),
            flea_price=it.get("avg24hPrice") or 0,
            best_trader=best["vendor"]["name"] if best else "",
            best_trader_price=best["priceRUB"] if best else 0,
        )

    needs = _cached("needs", NEEDS_QUERY, mode, force)
    for task in needs.get("tasks") or []:
        for obj in task.get("objectives") or []:
            if obj.get("type") not in ("giveItem", "plantItem"):
                continue
            fir = " (FIR)" if obj.get("foundInRaid") else ""
            for ref in obj.get("items") or []:
                if ref["id"] in items:
                    items[ref["id"]].quest_needs.append(
                        f"{task['name']} [{task['trader']['name']}] x{obj['count']}{fir}")
    for station in needs.get("hideoutStations") or []:
        for lvl in station.get("levels") or []:
            for req in lvl.get("itemRequirements") or []:
                iid = req["item"]["id"]
                if iid in items:
                    items[iid].hideout_needs.append(f"{station['name']} L{lvl['level']} x{req['count']}")
    return items
