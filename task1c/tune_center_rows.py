"""
Development helper - NOT part of the submission.

Scores lane_detection.py on the public clips with the official evaluator's own
functions (eyantra-autoeval), once for each candidate set of centre rows, so
the one camera-geometry parameter (which image rows the lane centre is read
at) can be chosen from evidence instead of guessed.

Run from task1c/ (the folder holding lane_detection.py and public/):

    python tune_center_rows.py

Pick the profile with the best centre marks, preferring one in the middle of a
flat region over a narrow peak, and copy its rows/weights into
CENTER_ROWS_FRAC / CENTER_ROW_WEIGHTS in lane_detection.py.
"""

import importlib
from pathlib import Path

ev = importlib.import_module("eyantra_autoeval.year.y2026.NV.task1c.evaluator")


def band(centre):
    rows = tuple(round(centre + d, 3) for d in (-0.06, -0.03, 0.0, 0.03, 0.06))
    return rows, (1.0, 2.0, 3.0, 2.0, 1.0)


def main():
    truth = ev.load_ground_truth()
    clips = sorted(truth["clips"])
    module, detect_lane = ev.load_detect_lane(Path("lane_detection.py").resolve())
    dataset = ev.find_dataset(clips)
    if dataset is None:
        return

    profiles = {"current": (module.CENTER_ROWS_FRAC, module.CENTER_ROW_WEIGHTS)}
    for c in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        profiles["band@{:.2f}".format(c)] = band(c)

    chance = max(ev.LANE_CHANCE_MIN, ev.majority_share(truth))
    print("{:<12}{:>8}{:>9}{:>8}{:>10}{:>9}".format(
        "profile", "lane", "centre", "total", "mean err", "unknown"))
    for name, (rows, weights) in profiles.items():
        module.CENTER_ROWS_FRAC = rows
        module.CENTER_ROW_WEIGHTS = weights
        module._STATE.clear()
        per_clip = {c: ev.score_clip(detect_lane, dataset / c, truth["clips"][c])
                    for c in clips}
        s = ev.aggregate(per_clip, chance)
        print("{:<12}{:>8.2f}{:>9.2f}{:>8.2f}{:>8.1f}px{:>9}".format(
            name, s["marks_lane"], s["marks_center"], s["marks_total"],
            s["mean_center_error_px"] or -1, s["unknown_returned"]))
        print("            rows={} weights={}".format(rows, weights))


if __name__ == "__main__":
    main()
