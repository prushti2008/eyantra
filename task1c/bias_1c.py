"""
Development helper - NOT part of the submission.

Measures the SIGNED centre error (center_x minus the true centre) of
lane_detection.py on the public clips, using the official evaluator's own
functions. A consistent offset shows up here as a non-zero median that has the
same sign on most clips; a one-off error shows up as a spread on one clip.

It also scores what a fixed sideways shift would earn, so you can see whether
there is anything to gain before changing any code. Run from task1c/:

    python bias_1c.py                       (uses lane_detection.py)
    python bias_1c.py lane_detection_v1.py
"""

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

ev = importlib.import_module("eyantra_autoeval.year.y2026.NV.task1c.evaluator")


def collect(detect_lane, clip_path, rows):
    cap = cv2.VideoCapture(str(clip_path))
    signed, lane_wrong, scored = [], 0, 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(rows):
            break
        res = detect_lane(frame.copy())
        gt_cx, code, amb = rows[i]
        i += 1
        if amb:
            continue
        scored += 1
        if res["lane"] != ev.LANE_WORDS[code] or res["center_x"] < 0:
            lane_wrong += 1
            continue
        signed.append(res["center_x"] - gt_cx)
    cap.release()
    return np.asarray(signed, dtype=np.float64), lane_wrong, scored


def credit(err):
    err = np.abs(err)
    return np.where(err <= 30, 1.0, np.where(err <= 60, 0.5, 0.0))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "lane_detection.py"
    truth = ev.load_ground_truth()
    clips = sorted(truth["clips"])
    _, detect_lane = ev.load_detect_lane(Path(name).resolve())
    dataset = ev.find_dataset(clips)
    if dataset is None:
        return

    print("\n{:<12}{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        "clip", "n", "median", "p10", "p90", "<=30px", "lane wrong"))
    everything, total_scored = [], 0
    for c in clips:
        e, wrong, scored = collect(detect_lane, dataset / c, truth["clips"][c])
        total_scored += scored
        if e.size:
            everything.append(e)
            print("{:<12}{:>7}{:>+9.1f}{:>+9.1f}{:>+9.1f}{:>8.0f}%{:>9}".format(
                c, e.size, np.median(e), np.percentile(e, 10),
                np.percentile(e, 90), 100.0 * np.mean(np.abs(e) <= 30), wrong))
        else:
            print("{:<12}{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
                c, 0, "-", "-", "-", "-", wrong))

    if not everything:
        return
    e = np.concatenate(everything)
    print("\nAll lane-correct frames: median {:+.1f} px, mean {:+.1f} px".format(
        np.median(e), np.mean(e)))
    print("\nCentre credit if every centre_x were shifted by s px "
          "(share of all scored frames):")
    best = (None, -1.0)
    for s in range(-30, 31, 5):
        share = float(np.sum(credit(e - s))) / total_scored
        if share > best[1]:
            best = (s, share)
        print("  shift {:+3d}: {:6.2f}%  -> {:5.2f} / 30".format(
            s, 100 * share, 30 * share))
    base = float(np.sum(credit(e))) / total_scored
    print("\nNo shift: {:.2f} / 30.  Best shift {:+d}: {:.2f} / 30  (gain {:+.2f} marks)".format(
        30 * base, best[0], 30 * best[1], 30 * (best[1] - base)))


if __name__ == "__main__":
    main()
