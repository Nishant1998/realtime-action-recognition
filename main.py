from ultralytics import YOLO
import numpy as np
import time
import cv2
import pandas as pd
from utils.meter import AverageMeter, Counter, Timer
from model.pose.video_pose_extractor import PoseExtractor
# Laptop gpu
# DET FPS : 2.69
# POSE FPS : 37.99
# FPS : 2.22

# GPU P100
# DET FPS : 61.21
# POSE FPS : 80.23
# FPS : 29.89

# Laptop
# fp32
# FPS :: yolo:2.92, tracker:489.82, pose:27.32
# FPS : 2.62
# v5n 3.1 v5nu 3.7


def main():
    # 'data/view-HC3.mp4'
    cap = cv2.VideoCapture('data/view-HC3.mp4')
    fps_meter = AverageMeter()
    timer = Timer()
    frame_count = Counter()
    pose_extractor = PoseExtractor(is_half=True)
    pose_extractor.eval()
    # pose_extractor = pose_extractor.half()

    colormap = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
            frame = cv2.resize(frame, (960, 544))

            timer.start()
            # frame = frame.half()
            track_id, boxes_xyxy, track_class, track_conf, track_pose_last_n, current_pose = pose_extractor(frame, frame_count.count)
            timer.stop()
            fps_meter.update(timer.get_duration())

            for i, (xt, yt, xr, yr) in enumerate(boxes_xyxy):
                tid = track_id[i]
                color = tuple(map(int, colormap[tid % len(colormap)]))
                cv2.rectangle(frame, (int(xt), int(yt)), (int(xr), int(yr)), color, 2)
                cv2.putText(frame, str(tid), (xt, yt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('frame', frame)
            # cv2.waitKey(0)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif frame_count.count > 128:
                break
        else:
            print(ret)
            break
        frame_count.increment()

    cap.release()
    cv2.destroyAllWindows()
    print(f"FPS : {1/fps_meter.avg:.2f}")


if __name__ == "__main__":
    main()


# import cv2
# import numpy as np
#
# # Load the image
# img = cv2.imread('image.jpg')
#
# # Define a list of colors to choose from
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#
# # Loop through each bounding box and draw it on the image
# for i, box in enumerate(boxes_xyxy):
#     # Extract the coordinates of the bounding box
#     x1, y1, x2, y2 = box
#
#     # Determine a random color for the bounding box based on the person ID
#     color = tuple(np.random.randint(0, 256, 3))
#
#     # Draw the bounding box on the image
#     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
# # Show the image with bounding boxes
# cv2.imshow('image with bounding boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

