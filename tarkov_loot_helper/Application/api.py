"""Domain model: turns raw tarkov.dev data into Item objects with prices and needs."""
from __future__ import annotations

from dataclasses import dataclass, field

from tarkov_loot_helper.Application.TarkovDevAPIConnector import TarkovDevAPIConnector


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
    types: list[str] = field(default_factory=list)
    caliber: str = ""  # guns only, e.g. "762x54R"
    capacity: int = 0  # magazines only: rounds it holds (shown as "x/17" on the tile)
    has_slots: bool = False   # can hold attachments (rails, mounts, handguards...)
    armor_part: bool = False  # built-in armor component of a helmet/armor (TOP, NAPE...)

    @property
    def is_gun(self) -> bool:
        return "gun" in self.types

    @property
    def is_preset(self) -> bool:
        return "preset" in self.types

    @property
    def best_price(self) -> int:
        return max(self.flea_price, self.best_trader_price)

    @property
    def price_per_slot(self) -> int:
        return self.best_price // max(1, self.width * self.height)


def load_items(mode: str = "regular", force: bool = False, language: str = "en") -> dict[str, Item]:
    connector = TarkovDevAPIConnector(mode, language)
    traders = {tid: t.get("name", tid) for tid, t in connector.fetch_traders(force).items()}
    items: dict[str, Item] = {}
    for iid, it in connector.fetch_items(force).items():
        sells = it.get("sellToTrader") or []
        best = max(sells, key=lambda s: s.get("priceRUB") or 0, default=None)
        items[iid] = Item(
            id=iid,
            name=it["name"],
            short_name=it["shortName"],
            width=it.get("width") or 1,
            height=it.get("height") or 1,
            grid_image=it.get("gridImageLink"),
            flea_price=it.get("avg24hPrice") or 0,
            best_trader=traders.get(best["trader"], "") if best else "",
            best_trader_price=(best.get("priceRUB") or 0) if best else 0,
            types=list(it.get("types") or []),
            caliber=str((it.get("properties") or {}).get("caliber") or "").removeprefix("Caliber"),
            capacity=int((it.get("properties") or {}).get("capacity") or 0),
            has_slots=bool((it.get("properties") or {}).get("slots")),
            armor_part=(it.get("properties") or {}).get("propertiesType") == "ItemPropertiesArmorAttachment",
        )

    for task in connector.fetch_tasks(force).values():
        trader = traders.get(task.get("trader"), "")
        for obj in task.get("objectives") or []:
            if obj.get("type") not in ("giveItem", "plantItem"):
                continue
            fir = " (FIR)" if obj.get("foundInRaid") else ""
            for iid in obj.get("items") or []:
                if iid in items:
                    items[iid].quest_needs.append(f"{task['name']} [{trader}] x{obj.get('count', 1)}{fir}")

    for station in connector.fetch_hideout(force).values():
        for lvl in station.get("levels") or []:
            for req in lvl.get("itemRequirements") or []:
                iid = req.get("item")
                if iid in items and iid != "5449016a4bdc2d6f028b456f":  # skip roubles
                    fir = " (FIR)" if (req.get("attributes") or {}).get("foundInRaid") else ""
                    items[iid].hideout_needs.append(f"{station['name']} L{lvl['level']} x{req['count']}{fir}")
    return items
