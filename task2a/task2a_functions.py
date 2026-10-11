# ===== Paste everything below ABOVE the "DO NOT EDIT BELOW THIS LINE" marker =====
# (keep the boilerplate's own imports; these two are needed)
import math
import cv2
import numpy as np

# ---------------- constants you MUST set / tune ----------------
WHEELBASE_M = 0.30      # TODO: your Task 1A wheelbase
TRACK_M = 0.20          # TODO: your Task 1A front track width
LANE_WIDTH_M = 0.30     # TODO: real width of one lane (m); only scales the error
CRUISE = 4.0            # rad/s drive speed. Start low (2-3), raise once it holds lane
SLOW = 2.0              # speed when the lane is lost
MAX_STEER = math.radians(35)

ROI_TOP = 0.50          # ignore image above this fraction (sky / far away)
SCAN_ROWS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)  # rows scanned
NEAR_ROW, LOOK_ROW = 0.90, 0.68   # rows used for error and look-ahead
K_HEAD = 1.2            # heading gain
K_CTE = 1.5             # cross-track gain (lane-widths -> Stanley term)
K_LOOK = 1.0            # look-ahead gain
SMOOTH = 0.6            # steering low-pass (0 = none, 1 = frozen)

# HSV thresholds (same idea as Task 1C - retune on a saved frame.png)
YELLOW_LO, YELLOW_HI = (15, 90, 90), (40, 255, 255)
WHITE_LO, WHITE_HI = (0, 0, 190), (180, 50, 255)

_S = {"steer": 0.0, "width": {}, "last": None, "lost": 0}


def _runs(mask_row, min_len=2):
    """Centres of contiguous True runs in a 1-D bool array."""
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 3) + 1
    return [float(g.mean()) for g in np.split(idx, splits) if g.size >= min_len]


def detect_lane(frame):
    """Return dict(e, psi, ok) where e is the lateral offset in lane widths
    (+ = lane centre is to the LEFT of the vehicle) and psi the lane direction
    in rad (+ = lane heads to the left). ok=False when nothing usable."""
    h, w = frame.shape[:2]
    cx = w / 2.0
    small = frame[int(h * ROI_TOP):, :]
    off = int(h * ROI_TOP)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI) > 0
    whi = cv2.inRange(hsv, WHITE_LO, WHITE_HI) > 0

    pts = []  # (row_frac, centre_x, width_px)
    for rf in SCAN_ROWS:
        r = int(h * rf) - off
        if r < 0 or r >= small.shape[0]:
            continue
        # average a few rows to bridge the dashed divider
        sl = slice(max(r - 3, 0), min(r + 4, small.shape[0]))
        wr = _runs(whi[sl].any(axis=0))
        yr = _runs(yel[sl].any(axis=0))
        if not wr:
            continue
        # divider = white run closest to the image centre
        xd = min(wr, key=lambda x: abs(x - cx))
        side = 1 if xd < cx else -1          # +1: divider on my left
        # the other boundary is yellow on the opposite side of the divider
        # my lane lies on the side away from the divider's nearer edge:
        # divider left of centre -> boundary is yellow to its right, and vice versa
        cand = [y for y in yr if (y - xd) * side > 0.03 * w]
        if cand:
            xy = min(cand, key=lambda y: abs(y - xd))
            width = abs(xy - xd)
            _S["width"][rf] = 0.8 * _S["width"].get(rf, width) + 0.2 * width
            pts.append((rf, 0.5 * (xd + xy), width))
        elif rf in _S["width"]:
            # only the divider visible: offset by half the learned lane width
            pts.append((rf, xd + side * 0.5 * _S["width"][rf],
                        _S["width"][rf]))

    if len(pts) < 3:
        _S["lost"] += 1
        return {"e": 0.0, "psi": 0.0, "ok": False}

    rows = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    deg = 2 if len(pts) >= 5 else 1
    co = np.polyfit(rows, xs, deg)
    f = np.poly1d(co)
    d = f.deriv()
    wnear = _S["width"].get(min(_S["width"], key=lambda k: abs(k - NEAR_ROW)), 0.35 * w)
    wlook = _S["width"].get(min(_S["width"], key=lambda k: abs(k - LOOK_ROW)), 0.2 * w)
    e = (f(NEAR_ROW) - cx) / max(wnear, 1.0)              # lane widths, +right
    elook = (f(LOOK_ROW) - cx) / max(wlook, 1.0)
    slope = float(d(NEAR_ROW))                            # px per row-fraction
    psi = math.atan2(slope / max(h, 1), 1.0)              # small-angle, +right
    _S["lost"] = 0
    # convert to "+ = left" convention used by the steering joints
    return {"e": -float(e), "elook": -float(elook), "psi": float(psi), "ok": True}


def _ackermann(delta):
    if abs(delta) < 1e-6:
        return 0.0, 0.0
    R = WHEELBASE_M / math.tan(delta)
    left = math.atan(WHEELBASE_M / (R - TRACK_M / 2.0))
    right = math.atan(WHEELBASE_M / (R + TRACK_M / 2.0))
    return left, right


def compute_control(detection, stop_requested):
    d = detection
    if d and d.get("ok"):
        e_m = d["e"] * LANE_WIDTH_M
        # Stanley-style: heading term + cross-track term + look-ahead term
        steer = (K_HEAD * d["psi"]
                 + math.atan2(K_CTE * e_m, 1.0)
                 + K_LOOK * math.atan2(d["elook"] * LANE_WIDTH_M, 1.0))
        speed = CRUISE
    else:
        steer = _S["steer"]            # hold last command while the lane is lost
        speed = SLOW
    steer = max(-MAX_STEER, min(MAX_STEER, steer))
    steer = SMOOTH * _S["steer"] + (1 - SMOOTH) * steer
    _S["steer"] = steer
    left, right = _ackermann(steer)
    if stop_requested:
        return left, right, 0.0
    return left, right, speed
