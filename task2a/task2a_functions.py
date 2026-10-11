"""
Task 2A - the two functions, to be pasted ABOVE the 'DO NOT EDIT BELOW THIS
LINE' marker of the boilerplate task2a.py (replace whatever is above it).

Pipeline:  frame -> lane geometry (Task 1C detector) -> cross-track error e
(metres) and heading error psi (radians) -> Stanley -> Ackermann split.

*** Fill in the CONSTANTS block from your boilerplate / Task 1A before use. ***
"""

import math
from collections import deque

import cv2
import numpy as np

LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)

# ============================== CONSTANTS ===================================
# Vehicle (use the values from your Task 1A / the boilerplate)
WHEELBASE_M = 0.30          # distance between front and rear axle
TRACK_M = 0.20              # distance between the two front wheels
MAX_STEER = math.radians(40.0)   # keep a margin below the 45 deg joint limit

# Camera (read them off the scene's vision sensor; see the guide)
CAM_FOV_DEG = 60.0          # horizontal field of view of the vision sensor
HORIZON_FRAC = 0.40         # image row (fraction of height) where the road
                            # lines converge - check on a saved frame
AXIS_FRAC = 0.50            # image column (fraction of width) of the
                            # vehicle's own axis

# Road
LANE_WIDTH_M = 0.35         # real width of the lane you are driving in; the
                            # pixel error is converted to metres with it

# Controller
K_STANLEY = 1.5             # cross-track gain
K_SOFT = 0.15               # softening speed term (m/s), avoids divide by 0
CRUISE_SPEED = 6.0          # wheel speed on straights, rad/s
MIN_SPEED = 3.0             # wheel speed in the tightest bend, rad/s
STEER_SLOW = math.radians(20.0)  # steering at which speed reaches MIN_SPEED
WHEEL_RADIUS_M = 0.04       # to turn rad/s into m/s for the Stanley term
STEER_RATE = math.radians(120.0)  # max change of steering per second
HOLD_FRAMES = 15            # frames to keep steering on the last good
                            # command when detection fails
# ============================================================================

# ---------------------------------------------------------------------------
# Task 1C lane detector (the version verified on the public clips)
# ---------------------------------------------------------------------------

# Rows (fractions of frame height) at which the divider is compared with the
# image centre to decide which lane the vehicle is in.
LANE_ROWS_FRAC = (0.70, 0.76, 0.82, 0.87, 0.91)

# Rows at which the lane centre is measured, and their weights. The weighted
# mean of these rows is ~0.64*h. This is the one camera-geometry parameter that
# matters most for the centre score: tune it against the public set (see
# tune_center_rows.py) rather than per-clip.
CENTER_ROWS_FRAC = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
CENTER_ROW_WEIGHTS = (1.0, 1.2, 1.5, 1.8, 2.0, 2.0, 1.8, 1.5)

# If the divider is within this fraction of the frame width of the image
# centre, the side is too close to call, so the previous lane is kept.
LANE_DECISION_MARGIN = 0.05

# How many consecutive frames a previous answer may be re-used for when the
# current frame defeats the detector.
MAX_HOLD_FRAMES = 6

# How many frames an old divider fit may stand in while the dashed line is in
# a gap.
MAX_WHITE_CACHE_AGE = 4

# Mean absolute difference (0..255) between ~32x24 grey thumbnails of
# consecutive frames above which a new clip is assumed and the memory cleared.
SCENE_CUT_DIFF = 15.0

# Number of recent lane-width measurements kept per centre row.
WIDTH_HISTORY = 15

# Weight of the new measurement when smoothing center_x over frames.
CENTER_SMOOTHING = 0.7

# The full analysis runs on every FULL_EVERY-th frame; in between, the answer
# of the previous frame is reused (the lane moves ~1 px per frame, the
# full-credit band is 30 px). A failed or first frame is always re-analysed.
FULL_EVERY = 1        # closed loop: analyse every frame

_STATE = {}


def _lane_1c(frame):
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return _unknown()

    h, w = frame.shape[:2]
    if h < 50 or w < 50:
        return _unknown()

    state = _get_state(frame)

    last = state["last"]
    if (
        last is not None
        and state["last_age"] == 0
        and state["since_full"] + 1 < FULL_EVERY
    ):
        # Reuse the previous frame's answer.
        state["since_full"] += 1
        return {"center_x": int(last["center_x"]), "lane": last["lane"]}

    state["since_full"] = 0
    observation = _analyse_frame(frame, state)
    return _temporal_update(state, observation, w)


def _unknown():
    return {"center_x": -1, "lane": LANE_UNKNOWN}


# ---------------------------------------------------------------------------
# Memory between frames
# ---------------------------------------------------------------------------

def _get_state(frame):
    '''Return the per-sequence memory, clearing it on a clip change.

    The evaluator calls detect_lane() on every clip back to back without any
    reset, so a new clip is detected from the picture itself.
    '''
    h, w = frame.shape[:2]
    small = np.ascontiguousarray(frame[h // 48::h // 24, w // 64::w // 32])
    thumb = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    fresh = (
        _STATE.get("shape") != (h, w)
        or _scene_changed(_STATE.get("thumb"), thumb)
    )

    if fresh:
        # The lane width is a property of the camera, so it survives a clip
        # change unless the resolution changes.
        widths = _STATE.get("widths")
        if widths is None or _STATE.get("shape") != (h, w):
            widths = deque(maxlen=WIDTH_HISTORY)
        _STATE.clear()
        _STATE.update({
            "shape": (h, w),
            "widths": widths,
            "last": None,          # last returned good answer
            "last_age": 0,         # frames since it was measured
            "white": None,         # last divider model
            "white_age": 0,
            "since_full": 0,       # frames since the last full analysis
        })

    _STATE["thumb"] = thumb
    return _STATE


def _scene_changed(previous, current):
    if previous is None or previous.shape != current.shape:
        return True
    return float(np.mean(np.abs(previous - current))) > SCENE_CUT_DIFF


def _temporal_update(state, observation, w):
    last = state["last"]

    if observation is None:
        if last is not None and state["last_age"] < MAX_HOLD_FRAMES:
            state["last_age"] += 1
            return {"center_x": int(last["center_x"]), "lane": last["lane"]}
        return _unknown()

    centre = float(observation["center_x"])
    lane = observation["lane"]

    if (
        last is not None
        and last["lane"] == lane
        and state["last_age"] == 0
    ):
        centre = (
            CENTER_SMOOTHING * centre
            + (1.0 - CENTER_SMOOTHING) * float(last["center_x"])
        )

    centre = int(round(centre))
    centre = max(0, min(w - 1, centre))

    result = {"center_x": centre, "lane": lane}
    state["last"] = result
    state["last_age"] = 0
    return dict(result)


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def _analyse_frame(frame, state):
    '''Return {"center_x": float, "lane": str} or None.'''
    h, w = frame.shape[:2]
    state["rows"] = None
    yellow, white = _masks(frame)

    # --- the dashed divider ---------------------------------------------
    white_model = _fit_white_divider(white, h, w)

    if white_model is None and state["white"] is not None \
            and state["white_age"] < MAX_WHITE_CACHE_AGE:
        # The dashed line is in a gap; the vehicle has hardly moved since
        # the last frame in which it was seen.
        white_model = state["white"]
        state["white_age"] += 1
    elif white_model is not None:
        state["white"] = white_model
        state["white_age"] = 0

    yellow_models = _fit_yellow_boundaries(yellow, h, w)

    if white_model is None:
        # No dash anywhere: the divider sits midway between the two solid
        # yellow edge lines when both are visible.
        white_model = _virtual_divider(yellow_models, h, w)

    if white_model is None:
        return None

    # --- which lane -----------------------------------------------------
    lane = _decide_lane(white_model, state, h, w)
    if lane is None:
        return None

    # --- where its centre is --------------------------------------------
    centre_y = np.asarray(CENTER_ROWS_FRAC, dtype=np.float64) * h
    white_x = _eval_model(white_model, centre_y)
    if white_x is None:
        return None
    white_x = np.asarray(white_x, dtype=np.float64)

    own_name = "left" if lane == LANE_LEFT else "right"
    other_lane = LANE_RIGHT if lane == LANE_LEFT else LANE_LEFT

    own_model = _fit_guided_yellow(yellow, white_model, lane, h, w)
    if own_model is None:
        own_model = yellow_models.get(own_name)

    learned = _learned_width(state, len(centre_y))

    centre = None
    if own_model is not None:
        own_x = _eval_model(own_model, centre_y)
        pair = _centre_from_pair(white_x, own_x, lane, w)

        if pair is not None:
            centre, good, widths = pair
            state["rows"] = (centre_y.copy(),
                             (white_x + own_x) / 2.0, widths.copy(),
                             good.copy())

            # Remember the width (accepted or not, so that a wrong early value
            # is washed out by the median of later, correct ones).
            row = np.full(len(centre_y), np.nan)
            row[good] = widths[good]
            state["widths"].append(row)

            # Reject a fit to some other yellow object whose width disagrees
            # strongly with the recent lane width.
            both = good & np.isfinite(learned)
            if np.count_nonzero(both) >= 3:
                ratio = float(np.median(widths[both] / learned[both]))
                if ratio < 0.65 or ratio > 1.5:
                    centre = None

    if centre is None:
        # Own yellow edge not usable: both lanes have the same width, so the
        # yellow line on the far side of the divider gives the width too.
        other_model = _fit_guided_yellow(yellow, white_model, other_lane, h, w)
        if other_model is None:
            other_model = yellow_models.get(
                "right" if own_name == "left" else "left")
        if other_model is not None:
            other_x = _eval_model(other_model, centre_y)
            pair = _centre_from_pair(white_x, other_x, other_lane, w,
                                     require_inside=False)
            if pair is not None:
                _, good, widths = pair
                width_rows = np.where(good, widths, np.nan)
                centre = _centre_from_width(white_x, width_rows, lane, w)

    if centre is None:
        # Last resort: the lane width seen in recent frames.
        centre = _centre_from_width(white_x, learned, lane, w)

    if centre is None or not np.isfinite(centre):
        return None

    if state.get("rows") is None:
        # Centre from the lane width alone: half a width from the divider.
        sign = -1.0 if lane == LANE_LEFT else 1.0
        width = learned if np.any(np.isfinite(learned)) else None
        if width is not None:
            centres = white_x + sign * width / 2.0
            good_rows = np.isfinite(centres) & (centres >= 0) & (centres <= w - 1)
            state["rows"] = (centre_y.copy(), centres, width.copy(), good_rows)

    return {"center_x": centre, "lane": lane}


def _decide_lane(white_model, state, h, w):
    lane_y = np.asarray(LANE_ROWS_FRAC, dtype=np.float64) * h
    white_lane = _eval_model(white_model, lane_y)
    if white_lane is None:
        return None

    white_lane = np.asarray(white_lane, dtype=np.float64)
    valid = (
        np.isfinite(white_lane)
        & (white_lane >= -0.20*w)
        & (white_lane <= 1.20*w)
    )
    if np.count_nonzero(valid) < 3:
        return None

    frame_mid = w / 2.0
    right_of_vehicle = white_lane[valid] > frame_mid
    lane = (
        LANE_LEFT
        if np.count_nonzero(right_of_vehicle) >= np.count_nonzero(valid)/2.0
        else LANE_RIGHT
    )

    # Too close to call: keep the lane of the recent past instead of letting
    # noise flip it.
    offset = float(np.median(white_lane[valid])) - frame_mid
    last = state["last"]
    if (
        abs(offset) < LANE_DECISION_MARGIN * w
        and last is not None
        and state["last_age"] < MAX_HOLD_FRAMES
    ):
        lane = last["lane"]

    return lane


def _centre_from_pair(white_x, yellow_x, lane, w, require_inside=True):
    '''Centre between the divider and the yellow edge of `lane`.

    Returns (centre, good_rows_mask, widths) or None.
    '''
    if white_x is None or yellow_x is None:
        return None

    yellow_x = np.asarray(yellow_x, dtype=np.float64)
    good = np.isfinite(white_x) & np.isfinite(yellow_x)
    good &= (white_x >= 0.0) & (white_x <= float(w - 1))
    if require_inside:
        good &= (yellow_x >= 0.0) & (yellow_x <= float(w - 1))

    if lane == LANE_LEFT:
        good &= yellow_x < white_x
    else:
        good &= yellow_x > white_x

    widths = np.abs(white_x - yellow_x)
    good &= (widths >= 20.0) & (widths <= 0.95*w)

    if np.count_nonzero(good) < 3:
        return None

    centre = _weighted_centre((white_x + yellow_x) / 2.0, good)
    if centre is None:
        return None
    return centre, good, widths


def _centre_from_width(white_x, widths, lane, w):
    '''Centre half a lane width away from the divider, on the lane's side.'''
    if widths is None:
        return None
    widths = np.asarray(widths, dtype=np.float64)
    sign = -1.0 if lane == LANE_LEFT else 1.0
    centres = white_x + sign * widths / 2.0
    good = (
        np.isfinite(centres)
        & np.isfinite(white_x)
        & (white_x >= 0.0) & (white_x <= float(w - 1))
        & (centres >= 0.0) & (centres <= float(w - 1))
    )
    if np.count_nonzero(good) < 3:
        return None
    return _weighted_centre(centres, good)


def _weighted_centre(centres, good):
    weights = np.asarray(CENTER_ROW_WEIGHTS, dtype=np.float64)[good]
    if weights.size == 0 or np.sum(weights) <= 0:
        return None
    centre = float(np.sum(centres[good] * weights) / np.sum(weights))
    return centre if np.isfinite(centre) else None


def _learned_width(state, n_rows):
    learned = np.full(n_rows, np.nan)
    history = [row for row in state["widths"] if len(row) == n_rows]
    if len(history) < 3:
        return learned
    # Median of the finite values of every column (NaNs sort to the end).
    stack = np.sort(np.vstack(history), axis=0)
    count = np.count_nonzero(np.isfinite(stack), axis=0)
    enough = count >= 3
    if np.any(enough):
        cols = np.flatnonzero(enough)
        c = count[enough]
        learned[enough] = 0.5 * (stack[(c - 1) // 2, cols] + stack[c // 2, cols])
    return learned


# ---------------------------------------------------------------------------
# Colour masks
# ---------------------------------------------------------------------------

def _masks(frame):
    h, w = frame.shape[:2]
    top = int(0.45*h)

    # Nothing above the horizon is used, so only the lower part is converted.
    hsv = cv2.cvtColor(frame[top:], cv2.COLOR_BGR2HSV)

    yellow_low = cv2.inRange(
        hsv,
        np.array([18, 80, 60], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8)
    )
    yellow_low[int(0.985*h) - top:, :] = 0

    # White threshold proportional to the road brightness (illumination
    # scales paint and asphalt alike), so the mask neither vanishes in shade
    # nor floods on a bright road. 170 on a road whose median V is 100.
    road_v = hsv[int(0.60*h) - top:int(0.90*h) - top:2, ::2, 2]
    road_median = float(np.median(road_v)) if road_v.size else 100.0
    v_min = int(np.clip(1.7 * road_median, 80.0, 225.0))

    white_low = cv2.inRange(
        hsv,
        np.array([0, 0, v_min], dtype=np.uint8),
        np.array([180, 110, 255], dtype=np.uint8)
    )
    # Pale yellow paint must not be taken for the divider.
    white_low[cv2.dilate(yellow_low, np.ones((3, 3), np.uint8)) > 0] = 0
    white_low[int(0.92*h) - top:, :] = 0

    white_low = cv2.morphologyEx(
        white_low, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    white_low = cv2.morphologyEx(
        white_low, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8)
    )

    yellow = np.zeros((h, w), dtype=np.uint8)
    white = np.zeros((h, w), dtype=np.uint8)
    yellow[top:] = yellow_low
    white[top:] = white_low
    return yellow, white


# ---------------------------------------------------------------------------
# Line fitting
# ---------------------------------------------------------------------------

def _fit_guided_yellow(mask, white_model, lane, h, w):
    '''Inner edge of the yellow line on `lane`'s side of the divider.'''
    y_start = int(0.54 * h)
    y_end = int(0.93 * h)

    rows = np.arange(y_start, y_end, 2, dtype=np.float64)
    white_values = _eval_model(white_model, rows)
    if white_values is None:
        return None
    white_values = np.asarray(white_values, dtype=np.float64)

    # All rows at once: yellow pixels on this lane's side of the divider,
    # grouped into runs (gaps of up to 4 px bridged), keeping runs of at
    # least 3 pixels. The run nearest the divider is the inner edge.
    sub = mask[rows.astype(np.int32)] > 0
    cols = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    if lane == LANE_LEFT:
        sub &= cols < (white_values[:, None] - 8.0)
    else:
        sub &= cols > (white_values[:, None] + 8.0)

    r, x = np.nonzero(sub)
    if r.size == 0:
        return None

    new_run = np.r_[True, (r[1:] != r[:-1]) | (np.diff(x) > 4)]
    run_id = np.cumsum(new_run) - 1
    counts = np.bincount(run_id)
    first = np.flatnonzero(new_run)
    last = np.r_[first[1:] - 1, r.size - 1]

    keep = counts >= 3
    run_row = r[first][keep]
    if run_row.size == 0:
        return None

    if lane == LANE_LEFT:
        # x increases within a row, so the last kept run has the largest end.
        run_x = x[last][keep]
        pick = np.r_[run_row[1:] != run_row[:-1], True]
    else:
        run_x = x[first][keep]
        pick = np.r_[True, run_row[1:] != run_row[:-1]]

    if np.count_nonzero(pick) < 20:
        return None

    arr = np.column_stack((
        rows[run_row[pick]],
        run_x[pick].astype(np.float64)
    ))
    linear = _polyfit_stable(arr[:, 0], arr[:, 1], 1)
    if linear is None:
        return None

    residual = np.abs(_eval_model(linear, arr[:, 0]) - arr[:, 1])
    inliers = residual <= 12.0
    if np.count_nonzero(inliers) < 18:
        return None

    good = arr[inliers]
    if np.ptp(good[:, 0]) < 80:
        return None

    # Refit on the inliers only, so that stray runs do not tilt the line.
    model = _polyfit_stable(good[:, 0], good[:, 1], 1) or linear

    if len(good) >= 35:
        curved = _polyfit_stable(good[:, 0], good[:, 1], 2)
        if curved is not None:
            lerr = float(np.mean(np.abs(
                _eval_model(model, good[:, 0]) - good[:, 1])))
            cerr = float(np.mean(np.abs(
                _eval_model(curved, good[:, 0]) - good[:, 1])))
            if cerr + 1.0 < lerr:
                model = curved

    return model


def _fit_yellow_boundaries(mask, h, w):
    '''Centre-line of the longest yellow run per row, in each frame half.'''
    models = {"left": None, "right": None}

    y_start = int(0.52 * h)
    y_end = int(0.96 * h)

    for name, x0, x1 in (("left", 0, w // 2), ("right", w // 2, w)):
        region = mask[y_start:y_end, x0:x1] > 0
        padded = np.pad(region, ((0, 0), (1, 1)), mode="constant",
                        constant_values=False)
        changes = np.diff(padded.astype(np.int8), axis=1)

        # Transitions alternate start, end along every row, and np.nonzero
        # is row-major, so even entries are run starts and odd ones run ends.
        t_rows, t_cols = np.nonzero(changes)
        if t_rows.size == 0:
            continue
        rows, start_cols = t_rows[0::2], t_cols[0::2]
        end_cols = t_cols[1::2]

        lengths = end_cols - start_cols
        keep = lengths >= 4
        if not np.any(keep):
            continue
        rows = rows[keep]
        start_cols = start_cols[keep]
        end_cols = end_cols[keep]
        lengths = lengths[keep]

        # Longest run of every row: sort by (row, length), take the last.
        order = np.lexsort((lengths, rows))
        rows = rows[order]
        last = np.r_[rows[1:] != rows[:-1], True]
        pick = order[last]

        ys = rows[last].astype(np.float64) + y_start
        xs = (start_cols[pick] + end_cols[pick] - 1.0) * 0.5 + x0

        if ys.size < 25:
            continue

        linear = _polyfit_stable(ys, xs, 1)
        if linear is None:
            continue

        residual = np.abs(_eval_model(linear, ys) - xs)
        inliers = residual <= 8.0
        if np.count_nonzero(inliers) < 20:
            continue

        gy, gx = ys[inliers], xs[inliers]
        linear = _polyfit_stable(gy, gx, 1) or linear
        model = linear

        if gy.size >= 45 and np.ptp(gy) >= 100:
            quadratic = _polyfit_stable(gy, gx, 2)
            if quadratic is not None:
                q_err = float(np.mean(np.abs(_eval_model(quadratic, gy) - gx)))
                l_err = float(np.mean(np.abs(_eval_model(linear, gy) - gx)))
                if q_err + 2.0 < l_err:
                    model = quadratic

        models[name] = model

    return models


def _virtual_divider(yellow_models, h, w):
    '''Midline of the two yellow edge lines, if both are plausible.'''
    left, right = yellow_models.get("left"), yellow_models.get("right")
    if left is None or right is None:
        return None

    ys = np.linspace(0.55*h, 0.92*h, 12)
    lx, rx = _eval_model(left, ys), _eval_model(right, ys)
    if lx is None or rx is None:
        return None

    road = rx - lx
    good = road >= 0.40*w
    if np.count_nonzero(good) < 6:
        return None
    return _model_from_points(ys[good], (lx[good] + rx[good]) / 2.0)


def _model_from_points(ys, xs):
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if ys.size >= 6 and np.ptp(ys) >= 60:
        return _polyfit_stable(ys, xs, 2)
    return _polyfit_stable(ys, xs, 1)


def _fit_white_divider(mask, h, w):
    # The mask is empty above 0.45*h, so label only the part below it.
    top = int(0.45*h)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask[top:]), connectivity=8
    )
    components = []

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        cx, cy = centroids[i]
        cy += top

        if area < 35 or area > 3000:
            continue
        if ww < 6 or hh < 6:
            continue
        if cy < 0.45*h or cy > 0.90*h:
            continue

        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        if aspect < 1.10:
            continue

        components.append((float(cx), float(cy)))

    if len(components) >= 2:
        data = np.asarray([(c[1], c[0]) for c in components],
                          dtype=np.float64)

        best_model = None
        best_inliers = None

        # Every pair of blobs at once: the line through the pair, its inliers
        # and its score (count, vertical span, fit error).
        ii, jj = np.triu_indices(len(data), k=1)
        y1, x1 = data[ii, 0], data[ii, 1]
        y2, x2 = data[jj, 0], data[jj, 1]
        usable = np.abs(y2 - y1) >= 30
        ii, jj = ii[usable], jj[usable]
        y1, x1, y2, x2 = y1[usable], x1[usable], y2[usable], x2[usable]

        if ii.size:
            slope = (x2 - x1) / (y2 - y1)
            pred = x1[:, None] + slope[:, None] * (data[None, :, 0] - y1[:, None])
            residual = np.abs(pred - data[None, :, 1])
            inl = residual <= 18.0
            count = np.count_nonzero(inl, axis=1)

            ys = np.broadcast_to(data[None, :, 0], inl.shape)
            span = (np.where(inl, ys, -np.inf).max(axis=1)
                    - np.where(inl, ys, np.inf).min(axis=1))
            mean_error = (np.where(inl, residual, 0.0).sum(axis=1)
                          / np.maximum(count, 1))
            score = 14.0*count + 0.08*span - 1.5*mean_error
            score = np.where(span < 70, score - 15.0, score)
            score = np.where(count >= 2, score, -np.inf)

            k = int(np.argmax(score))
            if np.isfinite(score[k]):
                best_model = _polyfit_stable(
                    [y1[k], y2[k]], [x1[k], x2[k]], 1)
                best_inliers = inl[k]

        if best_model is not None and best_inliers is not None:
            good = data[best_inliers]

            # Least-squares line through every inlier, not just the pair
            # that happened to win.
            if len(good) >= 3:
                refit = _polyfit_stable(good[:, 0], good[:, 1], 1)
                if refit is not None:
                    best_model = refit

            if len(good) >= 4 and np.ptp(good[:, 0]) >= 110:
                curved = _polyfit_stable(good[:, 0], good[:, 1], 2)
                if curved is not None:
                    line_err = float(np.mean(np.abs(
                        _eval_model(best_model, good[:, 0]) - good[:, 1])))
                    curve_err = float(np.mean(np.abs(
                        _eval_model(curved, good[:, 0]) - good[:, 1])))
                    if curve_err + 1.5 < line_err:
                        best_model = curved

            if np.ptp(good[:, 0]) >= 60:
                return best_model

    return _fit_white_hough(mask, h, w)


def _fit_white_hough(mask, h, w):
    '''Fallback: the dominant straight white stroke in the lower image.

    Fits x(y) through the segment end points themselves, so that a single
    visible dash still gives its true direction.
    '''
    edges = cv2.Canny(mask, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180.0,
        threshold=12,
        minLineLength=max(12, int(0.025*w)),
        maxLineGap=max(12, int(0.05*w))
    )
    if lines is None:
        return None

    segments = []
    for item in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in item]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < max(12.0, 0.025*w) or abs(dy) < 8:
            continue
        if max(y1, y2) < 0.55*h:
            continue
        segments.append((x1, y1, x2, y2, length))

    if not segments:
        return None

    # Seed with the longest segment, gather every segment lying on its line.
    segments.sort(key=lambda s: s[4], reverse=True)
    sx1, sy1, sx2, sy2, _ = segments[0]
    seed = _polyfit_stable([sy1, sy2], [sx1, sx2], 1)
    if seed is None:
        return None

    ys, xs = [], []
    for x1, y1, x2, y2, _ in segments:
        pred = _eval_model(seed, [y1, y2])
        if pred is None:
            continue
        if abs(pred[0] - x1) <= 15.0 and abs(pred[1] - x2) <= 15.0:
            ys.extend([y1, y2])
            xs.extend([x1, x2])

    if len(ys) < 2 or np.ptp(ys) < 20:
        return None
    return _polyfit_stable(ys, xs, 1)


def _polyfit_stable(x, y, degree):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < degree + 1:
        return None

    x0 = float(np.mean(x))
    scale = float(np.max(np.abs(x - x0)))
    if scale < 1e-9:
        return None

    u = (x - x0) / scale
    if degree == 1:
        # u has zero mean, so the least-squares line is closed-form.
        denom = float(np.dot(u, u))
        if denom < 1e-12:
            return None
        slope = float(np.dot(u, y)) / denom
        return (np.array([slope, float(np.mean(y))]), x0, scale)

    try:
        coef = np.polyfit(u, y, degree)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    return (coef, x0, scale)


def _eval_model(model, y):
    if model is None:
        return None

    try:
        if isinstance(model, tuple) and len(model) == 3:
            coef, x0, scale = model
            u = (np.asarray(y, dtype=np.float64) - x0) / scale
            value = np.polyval(coef, u)
        else:
            value = np.polyval(model, np.asarray(y, dtype=np.float64))
    except (TypeError, ValueError, FloatingPointError):
        return None

    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        return None

    if value.ndim == 0:
        return float(value)
    return value



# ---------------------------------------------------------------------------
# Stage 2 - detection, extended: geometry of the lane in front of the vehicle
# ---------------------------------------------------------------------------

_CTRL = {"prev_steer": 0.0, "prev_t": None, "last": None, "lost": 0}


def detect_lane(frame):
    """Return {"e": m, "psi": rad, "lane": str} or None.

    e   cross-track error: positive when the lane centre is to the RIGHT of
        the vehicle's axis.
    psi heading error: positive when the lane runs to the RIGHT of where the
        vehicle points.
    """
    _STATE["rows"] = None
    result = _lane_1c(frame)

    rows = _STATE.get("rows")
    lane = result["lane"]
    if lane == LANE_UNKNOWN or rows is None:
        # Divider out of view (e.g. the vehicle is close to the road edge):
        # use the outer yellow edge of the lane we were last in.
        last = _STATE.get("last")
        lane = last["lane"] if last is not None else LANE_UNKNOWN
        rows = _rows_from_edge(frame, lane) if lane != LANE_UNKNOWN else None
        if rows is None:
            return None
        result = {"lane": lane}
    ys, centres, widths, good = rows
    good = np.asarray(good, dtype=bool) & np.isfinite(centres) & np.isfinite(widths)
    if np.count_nonzero(good) < 3:
        return None

    h, w = frame.shape[:2]
    cx = AXIS_FRAC * w
    f_px = (w / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)
    v0 = HORIZON_FRAC * h

    # Use the nearest usable rows: they describe where the vehicle is now.
    order = np.argsort(-ys[good])[:5]
    yy = ys[good][order]
    xx = centres[good][order]
    ww = widths[good][order]

    # Cross-track error at the nearest row, scaled to metres by the measured
    # lane width at that row.
    y_near = float(yy[0])
    x_fit = np.polyfit(yy, xx, 1) if len(yy) >= 2 and np.ptp(yy) > 1 else None
    x_near = float(np.polyval(x_fit, y_near)) if x_fit is not None else float(xx[0])
    width_near = float(np.median(ww))
    if width_near < 10.0:
        return None
    e = (x_near - cx) / width_near * LANE_WIDTH_M

    # Heading error: where the local tangent of the centre line meets the
    # horizon, compared with the camera axis. On a line the slope dx/dy only
    # carries the line's lateral offset; its vanishing point carries the
    # heading.
    slope = float(x_fit[0]) if x_fit is not None else 0.0
    x_vp = x_near - slope * (y_near - v0)
    psi = math.atan((x_vp - cx) / f_px)
    psi = max(-math.radians(45.0), min(math.radians(45.0), psi))

    return {"e": e, "psi": psi, "lane": result["lane"]}


def _rows_from_edge(frame, lane):
    """Lane centre from the outer yellow edge and the learned lane width."""
    h, w = frame.shape[:2]
    yellow, _ = _masks(frame)
    models = _fit_yellow_boundaries(yellow, h, w)
    model = models.get("right" if lane == LANE_RIGHT else "left")
    if model is None:
        return None

    ys = np.asarray(CENTER_ROWS_FRAC, dtype=np.float64) * h
    edge_x = _eval_model(model, ys)
    if edge_x is None:
        return None
    widths = _learned_width(_STATE, len(ys))
    sign = -1.0 if lane == LANE_RIGHT else 1.0
    centres = np.asarray(edge_x, dtype=np.float64) + sign * widths / 2.0
    good = np.isfinite(centres) & (centres >= -0.5 * w) & (centres <= 1.5 * w)
    if np.count_nonzero(good) < 3:
        return None
    return ys, centres, widths, good


# ---------------------------------------------------------------------------
# Stage 3 - control
# ---------------------------------------------------------------------------

def ackermann_wheel_angles(delta):
    """Split one steering angle (positive = left) into the two wheel angles.
    Replace this body with your own Task 1A function if you prefer."""
    if abs(delta) < 1e-6:
        return 0.0, 0.0
    radius = WHEELBASE_M / math.tan(abs(delta))
    inner = math.atan(WHEELBASE_M / max(radius - TRACK_M / 2.0, 1e-6))
    outer = math.atan(WHEELBASE_M / (radius + TRACK_M / 2.0))
    inner = min(inner, math.radians(45.0))
    if delta > 0:
        return inner, outer           # left wheel is the inner one
    return -outer, -inner


def compute_control(detection, stop_requested, now=None, dt=None):
    """Return (left_angle, right_angle, speed). Never blocks, never raises."""
    import time
    t = time.time() if now is None else now
    if dt is None:
        dt = 0.05 if _CTRL["prev_t"] is None else max(1e-3, min(0.5, t - _CTRL["prev_t"]))
    _CTRL["prev_t"] = t

    if detection is not None:
        _CTRL["last"] = detection
        _CTRL["lost"] = 0
    else:
        _CTRL["lost"] += 1
        detection = _CTRL["last"] if _CTRL["lost"] <= HOLD_FRAMES else None

    if detection is None:
        steer_target = 0.0
        speed = 0.0 if stop_requested else MIN_SPEED * 0.5
    else:
        speed_ms = max(CRUISE_SPEED, 1e-3) * WHEEL_RADIUS_M
        # Stanley, written for a right-positive error; steering joints are
        # positive to the LEFT, hence the minus sign.
        delta_right = detection["psi"] + math.atan2(
            K_STANLEY * detection["e"], K_SOFT + speed_ms)
        steer_target = -delta_right
        steer_target = max(-MAX_STEER, min(MAX_STEER, steer_target))
        frac = min(1.0, abs(steer_target) / STEER_SLOW)
        speed = CRUISE_SPEED - (CRUISE_SPEED - MIN_SPEED) * frac
        if stop_requested:
            speed = 0.0

    # Limit how fast the steering may move.
    step = STEER_RATE * dt
    steer = _CTRL["prev_steer"] + max(-step, min(step, steer_target - _CTRL["prev_steer"]))
    _CTRL["prev_steer"] = steer

    left, right = ackermann_wheel_angles(steer)
    return left, right, speed
