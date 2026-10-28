'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  Task 3A - Mini Theme Run.
*
*  Filename:         task3.py
*  Created:          2026
*
*  One file: lane following (Task 2A), sign / obstacle detection (Task 2B) and
*  ArUco parking (Task 2C), joined by a small state machine (class Navigator).
*
*****************************************************************************************
'''

# Team ID:          NV_3765
# Author List:      PRACHI KHUHA, PRUSHTI RANINGA, MRUNALI TRIVEDI, TAPI KARIYA
# Filename:         task3.py
# Functions:        detect_lane, compute_control, detect_signs, detect_obstacle,
#                   parking_manoeuvre, Navigator, main (+ helpers)
# Global variables: see the TASK 3 TUNING block


####################### IMPORT MODULES #######################
import math
import sys
import time
from collections import deque

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'
##############################################################


################### TASK 3 TUNING (calibrate in the Easy arena) ###################
DEBUG = True               # print a trace once per simulated second; set False to submit

SPEED_MAX        = 3.0     # rules: limit everywhere
SPEED_ZONE       = 1.1     # rules: limit is 1.2 between the two Speed 40 signs
SPEED_NEAR_SIGN  = 1.0     # slow creep while closing in on the pickup point
SPEED_LANECHANGE = 1.0     # while changing lane
SPEED_PARK_APPR  = 1.1     # between the P sign and the ArUco hand-over

# Speed 40 sighting: a sign counts after this many consecutive frames, and a
# *second* sighting must follow an absence of ZONE_GAP_S sim-seconds and come at
# least ZONE_MIN_S sim-seconds after the first (so one sign flickering is not two).
SIGN_CONFIRM_FRAMES = 2
ZONE_GAP_S = 1.0
ZONE_MIN_S = 3.0

# Hospital pickup: sign area (px, from detect_signs) at which to slow down / halt.
# Print the areas with DEBUG on, drive to the pickup point once, and set
# HOSP_HALT_AREA to the area you read when the vehicle is level with the point.
HOSP_SLOW_AREA = 900
HOSP_HALT_AREA = 2200
HALT_SECONDS   = 5.0
HALT_MARGIN_S  = 0.3       # extra standing time, so the 5 s is never short
REST_DIFF      = 0.8       # mean grey change between frames that counts as "standing"
REST_FRAMES    = 4
REST_FALLBACK_S = 2.5      # assume at rest this long after the stop command

# Obstacle: change lane when an obstacle in OUR lane has its base at/below this
# image row (fraction of height) - roughly 0.35 m ahead of the camera with the
# 30 deg pitch - or is simply this big.
OBS_TRIGGER_Y    = 0.45
OBS_TRIGGER_AREA = 6000
LC_SHIFT      = 1.0        # lane widths to move over
LC_RAMP_S     = 1.0        # sim-seconds to ramp the offset in
LC_TIMEOUT_S  = 6.0        # give up if the lane label never flips
LC_SETTLE_S   = 1.5        # sim-seconds in the new lane before obstacles count again
LC_COOLDOWN_S = 1.0

# Parking
PARK_SIGN_FRAMES = 3
PARK_MIN_MARKER_PX = 14.0  # marker must look at least this wide to hand over
##############################################################################


# ============ Task 2B: sign and obstacle detection (unchanged) ============
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    '''
    Purpose:
    ---
    Detect every road sign visible in a single image, and report what each
    one is and how large it appears. An image can hold zero, one, or more
    than one sign at once.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `detections` :  [ list of dict ]
        Zero, one, or many entries - one per sign found in the image:
        [
            {
                "label" : str,   one of SIGN_LABELS
                "area"  : int,   that sign's area in pixels
            },
            ...
        ]

    Example call:
    ---
    detections = detect_signs(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    detections = []

    #################### ADD YOUR CODE HERE ####################
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img, w_img = image.shape[:2]
    obs_mask = _obstacle_mask(hsv)

    # ---- blue signs: Parking / Hospital ----
    blue = cv2.inRange(hsv, (95, 120, 50), (130, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for hull, x, y, w, h in _blobs(blue, MIN_BLUE_AREA):
        if _touches_border(x, y, w, h, w_img, h_img):
            continue
        if not 0.6 <= w / float(h) <= 1.7:
            continue
        label = _classify_blue(hsv, hull, blue, x, y, w, h)
        if label is not None:
            detections.append({"label": label, "area": int(cv2.contourArea(hull))})

    # ---- red ring signs: School / Speed40 ----
    red = _red_mask(hsv)
    red[cv2.dilate(obs_mask, np.ones((5, 5), np.uint8)) > 0] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for hull, x, y, w, h in _blobs(red, MIN_RING_AREA):
        if _touches_border(x, y, w, h, w_img, h_img):
            continue
        if not 0.5 <= w / float(h) <= 1.6 or max(w, h) < 20:
            continue
        hull_mask = np.zeros(red.shape, np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        hull_area = float(cv2.countNonZero(hull_mask))
        red_frac = cv2.countNonZero(cv2.bitwise_and(red, hull_mask)) / hull_area
        white = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
        white_frac = cv2.countNonZero(cv2.bitwise_and(white, hull_mask)) / hull_area
        if red_frac > 0.6 or white_frac < 0.25:
            continue                      # solid blob or no white inside -> not a sign
        _, r = cv2.minEnclosingCircle(hull)
        circle_fill = hull_area / (np.pi * r * r)
        label = "Speed40" if circle_fill > 0.75 else "School"
        detections.append({"label": label, "area": int(hull_area)})
    ############################################################

    return detections


def detect_obstacle(image):
    '''
    Purpose:
    ---
    Detect whether the obstacle is visible in a single image, and how large
    it appears.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "present" : bool,   True if the obstacle is visible
            "area"    : int,    its area in pixels, 0 if not present
        }

    Example call:
    ---
    result = detect_obstacle(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    present = False
    area = 0

    #################### ADD YOUR CODE HERE ####################
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _grow_obstacle(hsv, _obstacle_mask(hsv))
    best = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > best:
            best = int(st[i, cv2.CC_STAT_AREA])
    if best >= MIN_OBSTACLE_AREA:
        present = True
        area = best
    ############################################################

    return {"present": present, "area": area}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_signs() or detect_obstacle() - the
# evaluation script only ever calls those two functions. List them in
# the file header too.
# ------------------------------------------------------------------

MIN_BLUE_AREA = 300
MIN_RING_AREA = 150
MIN_OBSTACLE_AREA = 300


def _red_mask(hsv):
    lo = cv2.inRange(hsv, (0, 120, 70), (9, 255, 255))
    hi = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    return cv2.bitwise_or(lo, hi)


def _obstacle_mask(hsv):
    # solid red object: survives a 7x7 opening, thin sign rings and small dots do not
    red = _red_mask(hsv)
    solid = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    out = np.zeros_like(solid)
    n, lab, st, _ = cv2.connectedComponentsWithStats(solid)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < MIN_OBSTACLE_AREA or max(w, h) <= 20:
            continue
        out[lab == i] = 255
    return out


def _grow_obstacle(hsv, solid):
    # win back the edge pixels the opening removed (stays within the red mask, so
    # thin rings further away are not pulled in), then fill any holes
    grown = cv2.bitwise_and(cv2.dilate(solid, np.ones((9, 9), np.uint8)), _red_mask(hsv))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(grown)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def _blobs(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(hull)
        yield hull, x, y, w, h


def _touches_border(x, y, w, h, W, H):
    return x <= 0 or y <= 0 or x + w >= W or y + h >= H


def _classify_blue(hsv, hull, blue, x, y, w, h):
    hull_mask = np.zeros(blue.shape, np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    hull_area = float(cv2.countNonZero(hull_mask))
    k = 2 * max(2, int(0.15 * min(w, h))) + 1
    inner = cv2.erode(hull_mask, np.ones((k, k), np.uint8))
    letter = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 110), (180, 110, 255)), inner)
    frac = cv2.countNonZero(letter) / hull_area
    if not 0.06 <= frac <= 0.55:
        return None                       # no white letter inside -> not a sign
    hue = float(np.median(hsv[..., 0][cv2.bitwise_and(blue, hull_mask) > 0]))
    ys, xs = np.nonzero(letter)
    lh = ys.max() - ys.min() + 1
    if lh >= 8:
        runs = []
        for row in range(int(ys.min() + 0.78 * lh), int(ys.min() + 0.95 * lh) + 1):
            r = letter[row, xs.min():xs.max() + 1] > 0
            runs.append(int(np.count_nonzero(np.diff(r.astype(np.int8)) == 1) + int(r[0])))
        n_runs = int(np.median(runs))
        if n_runs >= 2:
            return "Hospital"         # H: two bars at the bottom
        if n_runs == 1:
            return "Parking"          # P: one stem at the bottom
    return "Parking" if hue < 108 else "Hospital"



# ============ Task 2A: lane detection and control (unchanged) ============

LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)

# ============================== CONSTANTS ===================================
# Vehicle (from Task 1A)
WHEELBASE_M = 0.120         # distance between front and rear axle
TRACK_M = 0.110             # distance between the two front wheels
MAX_STEER = math.radians(35.0)   # keep a margin below the 45 deg joint limit

# Camera - run calib_2a.py with the scene open to read these off the scene
CAM_FOV_DEG = 95.0          # horizontal field of view of the vision sensor
CAM_PITCH_DEG = 30.0        # how far the camera looks down
HORIZON_FRAC = 0.15         # image row (fraction of height) where the road
                            # lines converge
AXIS_FRAC = 0.50            # image column (fraction of width) of the
                            # vehicle's own axis

# Controller (the cross-track error is measured in LANE WIDTHS, so the real
# lane width in metres is not needed)
K_CT = 2.4                  # cross-track gain, per lane width
C_PSI = 1.0                 # look-ahead correction: lane widths of cross-track
                            # error per radian of heading estimate that belong
                            # to the bend, not to the vehicle's position
K_H = 1.0                   # weight of the heading error. On a bend the
                            # heading estimate also contains the curve of
                            # the road, so a value below 1 stops the vehicle
                            # steering too hard into it
CRUISE_SPEED = 2.0          # wheel speed on straights, rad/s
MIN_SPEED = 1.0             # wheel speed in the tightest bend, rad/s
STEER_SLOW = math.radians(20.0)  # steering at which speed reaches MIN_SPEED
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
    """Return {"e": lane widths, "psi": rad, "lane": str} or None.

    e   cross-track error in lane widths: positive when the lane centre is to
        the RIGHT of the vehicle's axis.
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

    # Cross-track error at the nearest row, as a fraction of the measured lane
    # width at that row.
    y_near = float(yy[0])
    x_fit = np.polyfit(yy, xx, 1) if len(yy) >= 2 and np.ptp(yy) > 1 else None
    x_near = float(np.polyval(x_fit, y_near)) if x_fit is not None else float(xx[0])
    width_near = float(np.median(ww))
    if width_near < 10.0:
        return None
    e = (x_near - cx) / width_near

    # Heading error: where the local tangent of the centre line meets the
    # horizon, compared with the camera axis. On a line the slope dx/dy only
    # carries the line's lateral offset; its vanishing point carries the
    # heading.
    slope = float(x_fit[0]) if x_fit is not None else 0.0
    x_vp = x_near - slope * (y_near - v0)
    # A camera pitched down by CAM_PITCH_DEG sees a sideways shift of the
    # vanishing point stretched by 1 / cos(pitch).
    psi = math.atan((x_vp - cx) * math.cos(math.radians(CAM_PITCH_DEG)) / f_px)
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
        # Stanley-style law, written for a right-positive error; the
        # steering joints are positive to the LEFT, hence the minus sign.
        e_corrected = detection["e"] - C_PSI * detection["psi"]
        delta_right = K_H * detection["psi"] + math.atan(K_CT * e_corrected)
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

# ---------------------------------------------------------------------------
# Task 2C - parking (unchanged apart from names: WHEELBASE -> WHEELBASE_P)
# ---------------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
BAY_MOUTH_FORWARD  = 0.37
BAY_MIDDLE_FORWARD = 0.19
BAY_WIDTH          = 0.22

WHEELBASE_P    = 0.120
TRACK_WIDTH    = 0.110
WHEEL_OFFSET   = 0.0275
STEER_LIMIT    = np.deg2rad(45.0)

IMG_W, IMG_H  = 640, 480
FOCAL_PX      = (IMG_W / 2) / np.tan(np.deg2rad(95.0) / 2)
CAMERA_MATRIX = np.array([[FOCAL_PX, 0, IMG_W / 2],
                          [0, FOCAL_PX, IMG_H / 2],
                          [0, 0, 1]])
CAM_PITCH     = np.deg2rad(30.0)
CAM_ABOVE_MK  = 0.176 - 0.110
CAM_AHEAD     = 0.047
MARKER_SIDE   = 0.096
_H = MARKER_SIDE / 2
MARKER_CORNERS_3D = np.array([[-_H, _H, 0], [_H, _H, 0],
                              [_H, -_H, 0], [-_H, -_H, 0]], np.float32)
YAW_BLEND       = 0.12
YAW_GATE        = np.deg2rad(15.0)
AMBIGUITY_RATIO = 1.5

# ---- tuning knobs ----
# Speeds are drive-motor velocities in rad/s. The path the vehicle takes does not
# depend on them; they only set how fast it gets there. Slow and steady works best:
# the pose estimate is only reliable if the vehicle is not changing direction too fast.
SLOW_DOWN_AT = 0.15       # m, start slowing this far before the bay mouth

# Steering: the vehicle aims at a heading (not at the marker) that brings it onto the
# bay's centre line, then a second loop steers to that heading.
#   aim       : the most it will ever aim across the road (rad)
#   look_k    : lookahead along the line = look_k * (forward - middle) ...
#   look_min / look_max : ... but never below / above these (m)
#   brake     : 1/m, how hard it allows itself to straighten out near the line
#   gain      : 1/(m*rad), steering from heading error
#   max_steer : steering limit used by this function (rad)
#   max_step  : the most the steering may change in one frame (rad)
#   cruise / creep : drive speeds, open road / last stretch (rad/s)
#   stop_early: call it parked this much before the middle of the bay (m)
#   lat_bias / head_bias: small corrections to the lateral offset (m) and heading (rad) it steers by
NEAR_BAY = dict(aim=0.72, look_k=0.20, look_min=0.05, look_max=0.56, brake=4.44,
                gain=7.68, max_steer=0.764, max_step=0.20,
                cruise=1.34, creep=1.40, stop_early=0.020, lat_bias=0.0, head_bias=0.0)
# bay 1 needs twice the sideways crossing, so it gets its own, gentler settings.
# lat_bias : the surveyed centre line of bay 1 sits about 2.6 mm to the right of the line the
#            marker gives (measured from a recording of a real run), so it aims that much right
# head_bias: the pose reads this much straighter than the vehicle really is after the long
#            crossing, so it steers to the heading that comes out parallel
FAR_BAY  = dict(aim=0.72, look_k=0.159, look_min=0.069, look_max=0.707, brake=4.40,
                gain=5.59, max_steer=0.621, max_step=0.19,
                cruise=0.94, creep=1.157, stop_early=-0.010, lat_bias=-0.0026, head_bias=-0.035)
FAR_BAY_LATERAL = 0.34    # m, a first lateral offset beyond this means the far bay

LOST_LIMIT = 15           # frames without the marker before it holds still

# memory between frames
_last_steering = 0.0
_lost_frames   = 0
_last_forward  = None
_params        = None


def parking_manoeuvre(pose):
    '''
    Purpose:
    ---
    Decide, every frame, how to steer and how fast to drive so that the
    vehicle ends up inside the target bay - on its centre line, parallel to
    it, stopped in the middle.

    Input Arguments:
    ---
    `pose` :    [ dict or None ]
        None on a frame where the target bay's marker was not seen.
        Otherwise 'lateral' (m, + = left of the centre line), 'heading'
        (rad, + = turned left), 'forward' (m to the marker line) and
        'bearing' (rad, + = marker to the left).

    Returns:
    ---
    `steering` :    [ float ]   radians, + = left
    `speed` :       [ float ]   rad/s
    `parked` :      [ bool ]    True once parked
    '''
    global _last_steering, _lost_frames, _last_forward, _params

    # ---- which bay? the vehicle starts level, so the first offset says ----
    if _params is None:
        if pose is None:
            return 0.0, 0.0, False
        _params = FAR_BAY if pose['lateral'] > FAR_BAY_LATERAL else NEAR_BAY
    p = _params

    # ---- no marker this frame ----
    if pose is None:
        _lost_frames += 1
        # close to the bay and the marker is gone: it is in the bay, stop here
        if _last_forward is not None and _last_forward < BAY_MIDDLE_FORWARD + 0.12 \
                and _lost_frames > 3:
            return 0.0, 0.0, True
        if _lost_frames > LOST_LIMIT:
            return 0.0, 0.0, False          # hold still until the marker is back
        return _last_steering, p['creep'], False   # one lost frame: carry on slowly

    _lost_frames = 0
    lateral = pose['lateral'] - p['lat_bias']
    heading = pose['heading'] - p['head_bias']
    forward = pose['forward']
    _last_forward = forward

    # ---- in the middle of the bay: stop ----
    if forward <= BAY_MIDDLE_FORWARD + p['stop_early']:
        return 0.0, 0.0, True

    # ---- aim: a heading that takes the vehicle onto the centre line ----
    lookahead = min(max(p['look_k'] * (forward - BAY_MIDDLE_FORWARD),
                        p['look_min']), p['look_max'])
    aim = -float(np.clip(np.arctan2(lateral, lookahead), -p['aim'], p['aim']))
    # do not aim steeper than it can still straighten out of before the line
    limit = np.sqrt(2.0 * p['brake'] * abs(lateral))
    aim = float(np.clip(aim, -limit, limit))

    # ---- steer to that heading ----
    curvature = p['gain'] * (aim - heading)
    steering = float(np.clip(np.arctan(WHEELBASE_P * curvature),
                             -p['max_steer'], p['max_steer']))
    steering = float(np.clip(steering, _last_steering - p['max_step'],
                             _last_steering + p['max_step']))
    _last_steering = steering

    # ---- speed: cruise, then creep for the last stretch ----
    if forward < BAY_MOUTH_FORWARD:
        speed = p['creep']
    elif forward > BAY_MOUTH_FORWARD + SLOW_DOWN_AT:
        speed = p['cruise']
    else:
        t = (forward - BAY_MIDDLE_FORWARD) / (BAY_MOUTH_FORWARD + SLOW_DOWN_AT - BAY_MIDDLE_FORWARD)
        speed = max(p['creep'], p['cruise'] * t)

    return steering, float(speed), False



class PoseEstimator:
    '''Where the vehicle is relative to the target bay, from the marker.

    Position comes straight from the marker each frame. Heading is harder: a
    small marker far away gives a noisy reading, and a square seen nearly
    face-on has two mirror-image poses that fit its corners almost equally
    well. So heading is predicted from the steering actually commanded, and
    each clear marker reading pulls that prediction a little toward itself.
    The vehicle is spawned parallel to the bays, which is where it starts.
    '''

    def __init__(self):
        self.heading = 0.0      # rad, + = turned RIGHT (internal convention)
        self.rho = None         # smoothed horizontal range to the marker, m

    @staticmethod
    def _bearing(corners):
        '''Horizontal bearing to the marker centre, rad, + = to the RIGHT.'''
        cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
        xc = (cx - IMG_W / 2) / FOCAL_PX
        yc = (IMG_H / 2 - cy) / FOCAL_PX
        return float(np.arctan2(xc, np.cos(CAM_PITCH) + yc * np.sin(CAM_PITCH)))

    @staticmethod
    def _pnp(corners):
        '''Heading (rad, + = RIGHT, or None if ambiguous) and horizontal range (m).'''
        _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            MARKER_CORNERS_3D, corners.astype(np.float32), CAMERA_MATRIX,
            np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = []
        for rv, tv, e in zip(rvecs, tvecs, np.array(errs).flatten()):
            R, _ = cv2.Rodrigues(rv)
            fwd = R.T @ np.array([0.0, 0.0, 1.0])   # camera forward, in the marker frame
            sols.append((float(e), float(np.arctan2(fwd[0], -fwd[2])), tv.flatten()))
        sols.sort(key=lambda s: s[0])
        t = sols[0][2]
        rho = float(np.sqrt(max(t @ t - CAM_ABOVE_MK ** 2, 1e-6)))
        clear = len(sols) < 2 or sols[1][0] > AMBIGUITY_RATIO * max(sols[0][0], 1e-6)
        return (sols[0][1] if clear else None), rho

    def update(self, corners, last_steering):
        '''The pose handed to parking_manoeuvre(), or None if no marker.'''
        if corners is None:
            return None
        bearing = self._bearing(corners)
        yaw_seen, rho_now = self._pnp(corners)
        if self.rho is None:
            self.rho = rho_now
        rho_prev, self.rho = self.rho, 0.6 * self.rho + 0.4 * rho_now
        # Distance covered since the last frame: range shrinks at cos(bearing)
        # per metre travelled.
        ds = min(max((rho_prev - self.rho) / max(np.cos(bearing), 0.3), 0.0), 0.01)
        # Steering LEFT (+) turns the heading LEFT, NEGATIVE in this convention.
        self.heading -= np.tan(last_steering) / WHEELBASE_P * ds
        if yaw_seen is not None and abs(yaw_seen - self.heading) < YAW_GATE:
            self.heading += YAW_BLEND * (yaw_seen - self.heading)

        los = bearing + self.heading                # line of sight, from the bay axis
        lateral_cam = -self.rho * np.sin(los)       # + = RIGHT of the centre line
        forward_cam = self.rho * np.cos(los)
        lateral_front = lateral_cam + (WHEELBASE_P / 2 - CAM_AHEAD) * np.sin(self.heading)
        forward = forward_cam + CAM_AHEAD * np.cos(self.heading)
        # Handed over in the Task 1 convention: POSITIVE = LEFT throughout.
        return {
            'lateral': float(-lateral_front),
            'heading': float(-self.heading),
            'forward': float(forward),
            'bearing': float(-bearing),
        }



# ===========================================================================
# Task 3 - what is new: the run logic
# ===========================================================================

def park_wheel_angles(delta):
    '''Task 1A / 2C kingpin Ackermann split, used for the parking manoeuvre.'''
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    l = WHEELBASE_P
    w = TRACK_WIDTH - 2 * WHEEL_OFFSET
    s, c = np.sin(delta), np.cos(delta)
    return (float(np.arctan2(2 * l * s, 2 * l * c - w * s)),
            float(np.arctan2(2 * l * s, 2 * l * c + w * s)))


def find_marker(detector, frame):
    '''Corners of the biggest ArUco marker in view (TL, TR, BR, BL), or None.'''
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(corners) == 0:
        return None
    best, best_side = None, 0.0
    for quad in corners:
        quad = quad.reshape(4, 2)
        side = float(np.linalg.norm(quad[0] - quad[1]))
        if side > best_side:
            best, best_side = quad, side
    return best if best_side >= PARK_MIN_MARKER_PX else None


def obstacle_info(frame):
    '''Largest obstacle: present, area, centroid x and the image row of its base.'''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _grow_obstacle(hsv, _obstacle_mask(hsv))
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area >= MIN_OBSTACLE_AREA and (best is None or area > best["area"]):
            best = {"present": True, "area": area, "cx": float(cen[i][0]),
                    "bottom": int(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT])}
    return best or {"present": False, "area": 0, "cx": 0.0, "bottom": 0}


def obstacle_in_my_lane(obs, lane, shape):
    '''True if the obstacle sits on our side of the white divider.'''
    h, w = shape[:2]
    model = _STATE.get("white")
    x_div = None
    if model is not None:
        y = float(np.clip(obs["bottom"], 0.45 * h, 0.92 * h))
        x_div = _eval_model(model, y)
    if x_div is None:
        # no divider known: the obstacle is ours if it is near the image axis
        return abs(obs["cx"] - w / 2.0) < 0.25 * w
    return obs["cx"] < x_div if lane == LANE_LEFT else obs["cx"] > x_div


def _thumb(frame):
    small = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


class Navigator:
    '''Keeps track of what the run has done and decides what to do next.

    Modes:  DRIVE -> HALT -> DRIVE ... -> PARK_APPROACH -> PARK -> DONE
    A lane change (self.lc) and the Speed 40 zone (self.zone) run alongside.
    '''

    def __init__(self):
        self.mode = "DRIVE"
        self.steer = 0.0                # last steering commanded, rad, + = left
        self.t = 0.0
        self.shape = (480, 640)

        # Speed 40 zone: 0 = not yet, 1 = inside, 2 = left again
        self.zone = 0
        self.sz_streak, self.sz_last_seen, self.sz_gap = 0, -1e9, 0.0
        self.zone_t0 = 0.0

        # Hospital halt
        self.hospital_done = False
        self.hosp_streak = 0
        self.halt_cmd_t = None
        self.rest_t = None
        self.rest_count = 0
        self.prev_thumb = None

        # Lane change
        self.lc = None
        self.lc_cooldown_until = 0.0
        self.lc_count = 0

        # Parking
        self.park_streak = 0
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv2.aruco.DetectorParameters())
        self.estimator = PoseEstimator()
        self.last_log = -1e9

    # ------------------------------------------------------------------
    def step(self, frame, t):
        '''One frame in, (left_angle, right_angle, speed) out.'''
        self.t, self.shape = t, frame.shape
        if self.mode == "PARK":
            return self._park(frame)
        if self.mode == "DONE":
            return 0.0, 0.0, 0.0

        signs = detect_signs(frame)
        best = {}
        for s in signs:
            best[s["label"]] = max(best.get(s["label"], 0), s["area"])
        det = detect_lane(frame)
        obs = obstacle_info(frame)

        self._update_zone(best)
        cap = SPEED_ZONE if self.zone == 1 else SPEED_MAX
        stop = False

        if self.mode == "HALT":
            stop = True
            self._update_halt(frame)
        elif self.mode in ("DRIVE", "PARK_APPROACH"):
            if self.mode == "DRIVE":
                cap = min(cap, self._hospital_check(best))
                self._parking_check(best, obs, det)
            if self.mode == "PARK_APPROACH":
                cap = min(cap, SPEED_PARK_APPR)
                corners = find_marker(self.detector, frame)
                if corners is not None and self.lc is None:
                    self.mode = "PARK"
                    print("[%.2f] hand over to parking" % self.t)
                    return self._park(frame, corners)

        bias, lc_cap = self._lane_change(det, obs)
        cap = min(cap, lc_cap)

        if det is not None and bias:
            det = dict(det, e=det["e"] + bias)
        left, right, speed = compute_control(det, stop, now=t)
        speed = min(speed, cap)
        self.steer = _CTRL["prev_steer"]
        self._log(best, obs, det, speed)
        return left, right, speed

    # ------------------------------------------------------------------
    def _update_zone(self, best):
        seen = "Speed40" in best
        t = self.t
        if seen:
            self.sz_streak += 1
            if self.sz_streak == 1:
                self.sz_gap = t - self.sz_last_seen
            if self.sz_streak >= SIGN_CONFIRM_FRAMES:
                if self.zone == 0:
                    self.zone, self.zone_t0 = 1, t
                    print("[%.2f] Speed 40 zone START" % t)
                elif (self.zone == 1 and self.sz_gap >= ZONE_GAP_S
                        and t - self.zone_t0 >= ZONE_MIN_S):
                    self.zone = 2
                    print("[%.2f] Speed 40 zone END" % t)
            self.sz_last_seen = t
        else:
            self.sz_streak = 0

    def _hospital_check(self, best):
        '''Returns a speed cap; starts the halt when the sign is big enough.'''
        if self.hospital_done:
            return SPEED_MAX
        area = best.get("Hospital", 0)
        self.hosp_streak = self.hosp_streak + 1 if area >= HOSP_HALT_AREA else 0
        if self.hosp_streak >= 2:
            self.mode = "HALT"
            self.halt_cmd_t, self.rest_t, self.rest_count = self.t, None, 0
            self.prev_thumb = None
            print("[%.2f] Hospital: stop commanded (area %d)" % (self.t, area))
            return 0.0
        return SPEED_NEAR_SIGN if area >= HOSP_SLOW_AREA else SPEED_MAX

    def _update_halt(self, frame):
        t = self.t
        if self.rest_t is None:
            th = _thumb(frame)
            if self.prev_thumb is not None and \
                    float(np.mean(np.abs(th - self.prev_thumb))) < REST_DIFF:
                self.rest_count += 1
            else:
                self.rest_count = 0
            self.prev_thumb = th
            if self.rest_count >= REST_FRAMES or t - self.halt_cmd_t >= REST_FALLBACK_S:
                self.rest_t = t
                print("[%.2f] Hospital: vehicle at rest" % t)
        elif t - self.rest_t >= HALT_SECONDS + HALT_MARGIN_S:
            self.hospital_done = True
            self.mode = "DRIVE"
            print("[%.2f] Hospital: halt finished (%.2f s at rest), moving off"
                  % (t, t - self.rest_t))

    def _parking_check(self, best, obs, det):
        '''Only after the pickup and the Speed 40 zone are both finished.'''
        ready = self.hospital_done and self.zone == 2 and self.lc is None
        if ready and "Parking" in best:
            self.park_streak += 1
        else:
            self.park_streak = 0
        if self.park_streak >= PARK_SIGN_FRAMES:
            self.mode = "PARK_APPROACH"
            print("[%.2f] Parking sign: looking for the ArUco marker" % self.t)

    def _lane_change(self, det, obs):
        '''Returns (offset in lane widths for e, speed cap).'''
        t = self.t
        lc = self.lc
        if lc is None:
            if (det is None or det["lane"] == LANE_UNKNOWN or not obs["present"]
                    or t < self.lc_cooldown_until):
                return 0.0, SPEED_MAX
            h = self.shape[0]
            near = (obs["bottom"] >= OBS_TRIGGER_Y * h
                    or obs["area"] >= OBS_TRIGGER_AREA)
            if near and obstacle_in_my_lane(obs, det["lane"], self.shape):
                self.lc = {"t0": t, "from": det["lane"], "flipped": None,
                           "sgn": 1.0 if det["lane"] == LANE_LEFT else -1.0}
                self.lc_count += 1
                print("[%.2f] Obstacle #%d in my (%s) lane (area %d) - changing lane"
                      % (t, self.lc_count, det["lane"], obs["area"]))
                return 0.0, SPEED_LANECHANGE
            return 0.0, SPEED_MAX

        if lc["flipped"] is None:
            if det is not None and det["lane"] not in (lc["from"], LANE_UNKNOWN):
                lc["flipped"] = t
                print("[%.2f] now in the %s lane" % (t, det["lane"]))
            elif t - lc["t0"] > LC_TIMEOUT_S:
                print("[%.2f] lane change timed out, giving up" % t)
                self.lc, self.lc_cooldown_until = None, t + LC_COOLDOWN_S
                return 0.0, SPEED_LANECHANGE
        if lc["flipped"] is not None:
            if t - lc["flipped"] >= LC_SETTLE_S:
                self.lc, self.lc_cooldown_until = None, t + LC_COOLDOWN_S
                return 0.0, SPEED_LANECHANGE
            return 0.0, SPEED_LANECHANGE
        ramp = min(1.0, (t - lc["t0"]) / LC_RAMP_S)
        return lc["sgn"] * LC_SHIFT * ramp, SPEED_LANECHANGE

    def _park(self, frame, corners=None):
        if corners is None:
            corners = find_marker(self.detector, frame)
        pose = self.estimator.update(corners, self.steer)
        steering, speed, parked = parking_manoeuvre(pose)
        if parked:
            self.mode = "DONE"
            print("[%.2f] PARKED" % self.t)
            return 0.0, 0.0, 0.0
        steering = float(np.clip(steering, -STEER_LIMIT, STEER_LIMIT))
        self.steer = steering
        left, right = park_wheel_angles(steering)
        return left, right, float(speed)

    def _log(self, best, obs, det, speed):
        if not DEBUG or self.t - self.last_log < 1.0:
            return
        self.last_log = self.t
        print("[%.1f] %s zone=%d lane=%s v=%.2f signs=%s obs=%d(base %d) lc=%s"
              % (self.t, self.mode, self.zone, det["lane"] if det else None, speed,
                 best, obs["area"], obs["bottom"], self.lc is not None))


##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


def connect_and_locate():
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with a Task 3 scene open?\n'
            f'  Simulator said: {err}')
    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        handles[name] = sim.getObject(path)
    return sim, handles


def read_frame(sim, handles):
    '''Camera frame as BGR: handleVisionSensor, flip rows, RGB -> BGR.'''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()
    nav = Navigator()

    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1
            left, right, speed = nav.step(frame, sim.getSimulationTime())
            drive(sim, handles, left, right, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass
    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
