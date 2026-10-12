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

# Team ID:          3765
# Author List:      Prachi Anilkumar Khuha
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle, _red_mask, _blue_mask,
#                   _components, _filled, _light, _letter_is_h
# Global variables: _K3, _K5


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)


def _red_mask(image, hsv):
    h, s, v = cv2.split(hsv)
    red = ((h <= 10) | (h >= 170)) & (s > 110) & (v > 80)
    # the overlay drawn on the captures has pure (0,0,255) marker dots - drop them
    b, g, r = cv2.split(image)
    dots = (b < 12) & (g < 12) & (r > 240)
    dots = cv2.dilate(dots.astype(np.uint8), _K3) > 0
    return (red & ~dots).astype(np.uint8)


def _blue_mask(hsv):
    h, s, v = cv2.split(hsv)
    return ((h >= 95) & (h <= 125) & (s > 150) & (v > 100)).astype(np.uint8)


def _components(mask, min_side, kernel=_K3):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = (int(t) for t in st[i])
        if w >= min_side and h >= min_side:
            yield i, lab, x, y, w, h, a


def _filled(lab, i, x, y, w, h):
    comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(comp)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    return comp, out


def _light(hsv_sub):
    return (hsv_sub[..., 1] < 90) & (hsv_sub[..., 2] > 140)


def _letter_is_h(letter):
    # bottom band of the letter: H has two legs, P has one
    ys, xs = np.nonzero(letter)
    if ys.size < 12:
        return None
    box = letter[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = box.shape
    band = box[int(hh * 0.72):, :]
    left = float(band[:, :ww // 2].sum())
    right = float(band[:, ww // 2:].sum())
    if left + right == 0:
        return None
    return min(left, right) / max(left, right) > 0.45


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

    # red-bordered signs: a hollow red ring with a light inside
    for i, lab, x, y, w, h, a in _components(_red_mask(image, hsv), 22):
        ring, filled = _filled(lab, i, x, y, w, h)
        box = float(w * h)
        fa = float(filled.sum())
        if a / box > 0.6 or not 0.6 < w / float(h) < 1.5:
            continue
        inside = (filled > 0) & (ring == 0)
        if inside.sum() < 0.35 * fa:
            continue
        if float(_light(hsv[y:y + h, x:x + w])[inside].mean()) < 0.5:
            continue
        label = "Speed40" if fa / box > 0.66 else "School"
        detections.append({"label": label, "area": int(fa)})

    # blue square signs with a light letter: Parking (P) or Hospital (H)
    for i, lab, x, y, w, h, a in _components(_blue_mask(hsv), 18):
        ring, filled = _filled(lab, i, x, y, w, h)
        box = float(w * h)
        if filled.sum() / box < 0.62 or not 0.6 < w / float(h) < 1.5:
            continue
        inside = (filled > 0) & (ring == 0)
        if inside.sum() < 0.08 * filled.sum():
            continue
        letter = _light(hsv[y:y + h, x:x + w]) & inside
        if letter.sum() < 0.06 * filled.sum():
            continue
        is_h = _letter_is_h(letter)
        if is_h is None:
            continue
        detections.append({"label": "Hospital" if is_h else "Parking",
                           "area": int(filled.sum())})
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
    best = 0
    for i, lab, x, y, w, h, a in _components(_red_mask(image, hsv), 24, _K5):
        ring, filled = _filled(lab, i, x, y, w, h)
        fa = float(filled.sum())
        if a / float(w * h) < 0.5 or a < 0.7 * fa:   # a solid block, not a ring
            continue
        best = max(best, int(fa))
    present = best > 0
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
