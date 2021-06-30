import cv2 as cv
import numpy as np
import math
import handtracker.hand_tracking_module as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeController:
    def __init__(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.vol_range = self.volume.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]


def main():
    vc = VolumeController()
    tracker = htm.HandTracker()
    cap = cv.VideoCapture(0)
    cap.set(3, 600)
    cap.set(4, 600)
    set_vol = -50
    vol_per = 0
    vol_bar = 350
    while True:
        ret, img = cap.read()
        img = tracker.find_hands(img)
        position = tracker.hand_position(img, 0)
        if len(position) > 0:
            x1, y1 = position[4][1], position[4][2]
            x2, y2 = position[8][1], position[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)
            length = math.hypot(x1 - x2, y1 - y2)
            # Hand range 20 - 300
            # volume range -50 - 5
            set_vol = np.interp(length, [20, 300], [-50, 5])
            vc.volume.SetMasterVolumeLevel(set_vol, None)
            vol_bar = np.interp(set_vol, [-50, 5], [350, 50])
            vol_per = np.interp(set_vol, [-50, 5], [0, 100])

        cv.rectangle(img, (50, 50), (80, 350), (0, 255, 0), 3)
        cv.rectangle(img, (50, int(vol_bar)), (80, 350), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(vol_per) + ' %', (50, 25), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.imshow("Live image", img)
        if cv.waitKey(1) == ord('q'):
            cap.release()
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
