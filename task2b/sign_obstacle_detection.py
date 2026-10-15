'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          NV_3765
# Author List:      PRACHI KHUHA, PRUSHTI RANINGA, MRUNALI TRIVEDI, TAPI KARIYA
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle, _red_mask, _obstacle_mask, _blobs,
#                   _touches_border, _classify_blue
# Global variables: MIN_BLUE_AREA, MIN_RING_AREA, MIN_OBSTACLE_AREA


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
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
    mask = _obstacle_mask(hsv)
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


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area
