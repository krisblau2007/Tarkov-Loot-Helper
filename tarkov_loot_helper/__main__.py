"""Entry point.

  python -m tarkov_loot_helper                 run the tray app (hotkey scanning)
  python -m tarkov_loot_helper setup           download item data + all icons
  python -m tarkov_loot_helper scan IMG.png    scan a saved screenshot (debug)
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="tarkov_loot_helper")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("setup", help="download item data and icons")
    sc = sub.add_parser("scan", help="scan a screenshot file and write an annotated copy")
    sc.add_argument("image")
    sc.add_argument("--ui-scale", type=float, help="override UI scale for this run")
    sc.add_argument("--export", action="store_true", help="also write CSV/JSON")
    args = parser.parse_args()

    if args.cmd == "setup":
        from . import api, icons
        from .config import Settings
        items = api.load_items(Settings.load().game_mode, force=True)
        print(f"{len(items)} items loaded, downloading icons...")
        n = icons.prefetch(list(items.values()), lambda i, t: print(f"  {i}/{t}"))
        print(f"{n} icons cached.")
        return 0

    if args.cmd == "scan":
        import cv2
        import numpy as np

        from . import api, export
        from .config import Settings
        from .identify import Identifier
        from .scanner import render_debug, scan_image
        settings = Settings.load()
        if args.ui_scale:
            settings.ui_scale = args.ui_scale
        img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {args.image}")
            return 1
        result = scan_image(img, settings, Identifier(api.load_items(settings.game_mode)))
        out = args.image.rsplit(".", 1)[0] + "_annotated.png"
        cv2.imwrite(out, render_debug(result))
        print(f"cell={settings.cell_px(img.shape[0]):.1f}px  grids={len(result.grids)}  "
              f"regions={len(result.detections)}  identified={len(result.identified)}")
        for d in result.identified:
            it = d.item
            print(f"  [{d.region.grid_index}] {it.name:<45} {d.score:.2f}  "
                  f"{it.price_per_slot:>8,}/slot  {'Q' if it.quest_needs else ' '}{'H' if it.hideout_needs else ' '}")
        print(f"Annotated image: {out}")
        if args.export:
            print("Exported:", *export.export(result))
        return 0

    from .app import App
    return App().run()


if __name__ == "__main__":
    sys.exit(main())
