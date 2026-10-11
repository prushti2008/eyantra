"""
Development helper - NOT part of the submission.

Scores several versions of the detector on the public clips with the official
evaluator's own functions, and prints one line per file, so the better file
can be chosen on evidence. Run from task1c/ (needs eyantra-autoeval):

    python compare_1c.py lane_detection_old.py lane_detection_new.py
"""

import importlib
import sys
import time
from pathlib import Path

ev = importlib.import_module("eyantra_autoeval.year.y2026.NV.task1c.evaluator")


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return

    truth = ev.load_ground_truth()
    clips = sorted(truth["clips"])
    dataset = ev.find_dataset(clips)
    if dataset is None:
        return
    chance = max(ev.LANE_CHANCE_MIN, ev.majority_share(truth))

    print("\nPUBLIC = marks on the public clips (lane chance-corrected).")
    print("PRIVATE-EST = 20*lane accuracy + 30*centre score: what the same "
          "answers earn on a single-lane set, where the lane marks are not "
          "chance-corrected.\n")
    print("{:<26}{:>7}{:>8}{:>8}{:>9}{:>9}{:>9}{:>12}".format(
        "file", "lane", "centre", "PUBLIC", "PRIV-EST", "mean err",
        "unknown", "detect time"))
    for name in files:
        _, detect_lane = ev.load_detect_lane(Path(name).resolve())
        t0 = time.perf_counter()
        per_clip = {c: ev.score_clip(detect_lane, dataset / c, truth["clips"][c])
                    for c in clips}
        wall = time.perf_counter() - t0
        s = ev.aggregate(per_clip, chance)
        private_est = (ev.MARKS_LANE * s["lane_accuracy"]
                       + ev.MARKS_CENTER * s["center_accuracy"])
        print("{:<26}{:>7.2f}{:>8.2f}{:>8.2f}{:>9.2f}{:>7.1f}px{:>9}{:>10.1f} s".format(
            Path(name).name[:25], s["marks_lane"], s["marks_center"],
            s["marks_total"], private_est, s["mean_center_error_px"] or -1,
            s["unknown_returned"], s["detect_seconds"]))
        bad = ["{} {}/{}".format(c, t["lane_correct"], t["scored"])
               for c, t in sorted(per_clip.items())
               if t["lane_correct"] < t["scored"]]
        print("    lane errors in: " + (", ".join(bad) if bad else "none")
              + "   (total run {:.0f} s)".format(wall))


if __name__ == "__main__":
    main()
