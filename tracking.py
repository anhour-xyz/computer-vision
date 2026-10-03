import numpy as np
import cv2
import time


lk_params = dict(winSize=(15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not open camera 0. On macOS, allow camera access for the app "
        "running Python in System Settings > Privacy & Security > Camera, "
        "then restart that app."
    )

while True:

    start = time.time()
    suc,frame = cap.read()
    if not suc or frame is None:
        cap.release()
        raise RuntimeError("Camera opened but did not return a frame.")

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in trajectories]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []
        for tr, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > trajectory_len:
                del tr[0]
            new_trajectories.append(tr)
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        trajectories = new_trajectories

        cv2.polylines(img, [np.int32(tr) for tr in trajectories], False, (0, 255, 0))
        cv2.putText(img, "track count: %d" % len(trajectories), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        for x,y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x,y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

        frame_idx += 1
        prev_gray = frame_gray

        end = time.time()
        fps = 1 / (end-start)

        cv2.putText(img, "FPS: %.2f" % fps, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Optical Flow", img)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
