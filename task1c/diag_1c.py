"""
Development helper - NOT part of the submission.

Shows WHY lane_detection.py picks the wrong lane on some public clips.
Run from task1c/ (needs eyantra-autoeval installed):

    python diag_1c.py clip_11.mp4 clip_18.mp4 clip_19.mp4 clip_20.mp4

Prints, for every stretch of wrong-lane frames, what the detector saw, and
writes diag_1c.png: one annotated wrong frame per clip
    red    = fitted dashed divider       yellow = fitted yellow lines
    blue   = your center_x               green  = true centre
    grey   = image centre
"""

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

ev = importlib.import_module("eyantra_autoeval.year.y2026.NV.task1c.evaluator")


def draw_model(img, ld, model, colour):
    if model is None:
        return
    h = img.shape[0]
    ys = np.arange(int(0.45*h), h, 4, dtype=np.float64)
    xs = ld._eval_model(model, ys)
    if xs is None:
        return
    pts = np.column_stack((xs, ys)).astype(np.int32)
    cv2.polylines(img, [pts], False, colour, 2)


def main():
    names = sys.argv[1:] or ["clip_11.mp4", "clip_18.mp4",
                             "clip_19.mp4", "clip_20.mp4"]
    truth = ev.load_ground_truth()
    ld, detect_lane = ev.load_detect_lane(Path("lane_detection.py").resolve())
    dataset = ev.find_dataset(sorted(truth["clips"]))
    tiles = []

    for name in names:
        rows = truth["clips"][name]
        ld._STATE.clear()
        cap = cv2.VideoCapture(str(dataset / name))
        wrong, tile, i = [], None, 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(rows):
                break
            res = detect_lane(frame.copy())
            gt_cx, code, amb = rows[i]
            gt_lane = ev.LANE_WORDS[code]
            if not amb and res["lane"] != gt_lane:
                h, w = frame.shape[:2]
                yellow, white = ld._masks(frame)
                wm = ld._fit_white_divider(white, h, w)
                ym = ld._fit_yellow_boundaries(yellow, h, w)
                y8 = 0.8*h
                fmt = lambda m: ("{:4.0f}".format(ld._eval_model(m, y8))
                                 if m is not None else "  - ")
                wrong.append((i, gt_lane, res["lane"], res["center_x"], gt_cx,
                              fmt(wm), fmt(ym["left"]), fmt(ym["right"]),
                              int(np.count_nonzero(white)),
                              int(np.count_nonzero(yellow))))
                if tile is None and len(wrong) >= 5:
                    tile = frame.copy()
                    draw_model(tile, ld, wm, (0, 0, 255))
                    draw_model(tile, ld, ym["left"], (0, 255, 255))
                    draw_model(tile, ld, ym["right"], (0, 255, 255))
                    cv2.line(tile, (w//2, h), (w//2, h - 60), (128, 128, 128), 2)
                    cv2.line(tile, (res["center_x"], h), (res["center_x"], h//2),
                             (255, 0, 0), 2)
                    cv2.line(tile, (int(gt_cx), h), (int(gt_cx), h//2),
                             (0, 255, 0), 2)
                    cv2.putText(tile, "{} f{} true={} got={}".format(
                        name, i, gt_lane, res["lane"]), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    mask_view = np.zeros_like(frame)
                    mask_view[white > 0] = (255, 255, 255)
                    mask_view[yellow > 0] = (0, 255, 255)
                    tiles.append(np.hstack((tile, mask_view)))
            i += 1
        cap.release()

        print("\n=== {}: {} wrong-lane frames ===".format(name, len(wrong)))
        print(" frame true  got   cx  gtcx | x@0.8h: divider yel_L yel_R"
              " | white_px yellow_px")
        for k, r in enumerate(wrong):
            if k % 5 == 0 or k == len(wrong) - 1:
                print(" {:5d} {:5} {:5} {:4d} {:5.0f} |        {}  {}  {}"
                      " | {:8d} {:9d}".format(*r))

    if tiles:
        sheet = np.vstack(tiles)
        sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
        cv2.imwrite("diag_1c.png", sheet)
        print("\nwrote diag_1c.png")


if __name__ == "__main__":
    main()
