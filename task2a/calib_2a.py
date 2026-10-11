"""
Reads the camera constants of the scene so you do not have to guess them.

1. Open CoppeliaSim with Task_2A.ttt (do NOT press Play).
2. Run:   python calib_2a.py
It saves frame.png and prints the values for the CONSTANTS block of task2a.py.
It only READS from the scene; it changes nothing.
"""
import math

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'


def rotate(q, v):
    x, y, z, w = q
    t = 2.0 * np.cross([x, y, z], v)
    return np.asarray(v) + w * t + np.cross([x, y, z], t)


def main():
    sim = RemoteAPIClient().getObject('sim')
    cam = sim.getObject(VISION_SENSOR)

    res = sim.getVisionSensorResolution(cam)
    w, h = int(res[0]), int(res[1])
    print("image size            : {} x {}".format(w, h))

    try:
        fov = sim.getObjectFloat(cam, sim.visionfloatparam_perspective_angle) \
            if hasattr(sim, "getObjectFloat") else \
            sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle)
        # The angle belongs to the larger image side.
        f_px = (max(w, h) / 2.0) / math.tan(fov / 2.0)
        hfov = 2.0 * math.atan((w / 2.0) / f_px)
        print("CAM_FOV_DEG           = {:.1f}".format(math.degrees(hfov)))
    except Exception as err:
        print("could not read the field of view:", err)
        f_px, hfov = None, None

    try:
        pose = sim.getObjectPose(cam, sim.handle_world)
        axis = rotate(pose[3:7], [0.0, 0.0, 1.0])      # viewing direction
        pitch_down = -math.asin(max(-1.0, min(1.0, axis[2])))
        print("camera height         : {:.3f} m".format(pose[2]))
        print("camera pitch down     : {:.1f} deg".format(math.degrees(pitch_down)))
        if f_px:
            horizon_row = h / 2.0 - f_px * math.tan(pitch_down)
            print("HORIZON_FRAC          = {:.2f}   (horizon at row {:.0f} of {})"
                  .format(horizon_row / h, horizon_row, h))
    except Exception as err:
        print("could not read the camera pose:", err)

    try:
        sim.handleVisionSensor(cam)
        img, r = sim.getVisionSensorImg(cam, 0)
        if len(img):
            frame = np.frombuffer(img, dtype=np.uint8).reshape((r[1], r[0], 3))
            frame = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_RGB2BGR)
            cv2.imwrite('frame.png', frame)
            print("saved frame.png - open it and check the road looks right")
    except Exception as err:
        print("could not save a frame:", err)


if __name__ == "__main__":
    main()
