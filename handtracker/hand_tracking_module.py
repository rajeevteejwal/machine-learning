import cv2 as cv
import mediapipe as mp


class HandTracker:
    def __init__(self, mode=False, maxnumhands=2, detectionconfidence=0.5, trackingconfidence=0.5):
        self.mode = mode
        self.maxNumHands = maxnumhands
        self.detectionConfidence = detectionconfidence
        self.trackingConfidence = trackingconfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.handDraw = mp.solutions.drawing_utils
        self.result = None

    def find_hands(self, img, is_draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        if self.result.multi_hand_landmarks:
            for handsLMS in self.result.multi_hand_landmarks:
                if is_draw:
                    self.handDraw.draw_landmarks(img, handsLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def hand_position(self, img, hand_num=0):
        lm_list = []
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[0]
            for p_id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([p_id, cx, cy])
        return lm_list


def main():
    cap = cv.VideoCapture(0)
    ht = HandTracker()
    while True:
        ret, img = cap.read()
        img = ht.find_hands(img)
        hand_positions = ht.hand_position(img)
        for point in hand_positions:
            cv.circle(img, (point[1], point[2]), 5, (255, 0, 255), cv.FILLED)

        cv.imshow("Live Image", img)
        if cv.waitKey(1) == ord('q'):
            cap.release()
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
