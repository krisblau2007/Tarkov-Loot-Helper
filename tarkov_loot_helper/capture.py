"""Screen capture (read-only screenshots; never touches game memory)."""
from __future__ import annotations

import mss
import numpy as np


def grab(monitor_index: int = 1) -> tuple[np.ndarray, dict]:
    """Return (BGR image, monitor geometry dict with left/top/width/height)."""
    with mss.mss() as sct:
        monitors = sct.monitors
        mon = monitors[monitor_index if monitor_index < len(monitors) else 1]
        shot = np.asarray(sct.grab(mon))[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(shot), dict(mon)
