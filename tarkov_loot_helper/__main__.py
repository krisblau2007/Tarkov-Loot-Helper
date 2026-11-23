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
    sc.add_argument("--out", help="where to write the annotated image")
    args = parser.parse_args()

    if args.cmd == "setup":
        from tarkov_loot_helper.Application import api
        from tarkov_loot_helper.Identification import icons
        from tarkov_loot_helper.Configuration.config import Settings
        s = Settings.load()
        s.refresh_api_options(force=True)
        print(f"Game modes: {', '.join(s.available_game_modes)} (using '{s.game_mode}')")
        items = api.load_items(s.game_mode, force=True, language=s.language)
        print(f"{len(items)} items loaded, downloading icons...")
        n = icons.prefetch(list(items.values()), lambda i, t: print(f"  {i}/{t}"))
        print(f"{n} icons cached.")
        return 0

    if args.cmd == "scan":
        import cv2
        import numpy as np

        from tarkov_loot_helper.Application import api

        from tarkov_loot_helper.Identification import export
        from tarkov_loot_helper.Configuration.config import Settings
        from tarkov_loot_helper.Identification.identify import Identifier
        from tarkov_loot_helper.Identification.scanner import render_debug, scan_image
        settings = Settings.load()
        if args.ui_scale:
            settings.ui_scale = args.ui_scale
        img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {args.image}")
            return 1
        result = scan_image(img, settings, Identifier(api.load_items(settings.game_mode, language=settings.language)))
        out = args.out or args.image.rsplit(".", 1)[0] + "_annotated.png"
        cv2.imwrite(out, render_debug(result))
        print(f"cell={settings.cell_px(img.shape[0]):.1f}px  grids={len(result.grids)}  "
              f"regions={len(result.detections)}  identified={len(result.identified)}")
        for d in result.detections:
            it = d.item or d.guess
            where = "equip" if d.kind == "equipped" else f"{d.container}:{d.col},{d.row}"
            print(f"  {'  ' if d.item else '? '}[{where:>8}] {d.width}x{d.height} {it.name[:44]:<44} "
                  f"{d.score:.2f} {d.method:<8} {d.price_per_slot:>8,}/slot  "
                  f"{'Q' if it.quest_needs else ' '}{'H' if it.hideout_needs else ' '}  ocr={d.name_text!r}")
        print(f"Annotated image: {out}")
        if args.export:
            print("Exported:", *export.export(result))
        return 0

    from tarkov_loot_helper.Application.app import App
    return App().run()


def _crash_hook(exc_type, exc, tb) -> None:
    """Unhandled errors (including inside Qt callbacks) are fatal: print and exit non-zero."""
    import traceback
    print("\nCRITICAL ERROR - Tarkov Loot Helper has stopped:\n", file=sys.stderr)
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()
    os._exit(1)


if __name__ == "__main__":
    import os
    sys.excepthook = _crash_hook
    try:
        sys.exit(main())
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:
        _crash_hook(*sys.exc_info())
