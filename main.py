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


def main():
    cap = cv2.VideoCapture('data/view-HC3.mp4')
    fps_meter = AverageMeter()
    timer = Timer()
    frame_count = Counter()
    pose_extractor = PoseExtractor(is_half=True)
    pose_extractor.eval()
    # pose_extractor = pose_extractor.half()

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))

            timer.start()
            # frame = frame.half()
            track_id, boxes_xyxy, track_class, track_conf, track_pose_last_n, current_pose = pose_extractor(frame, frame_count.count)
            timer.stop()
            fps_meter.update(timer.get_duration())

            for xt, yt, xr, yr in boxes_xyxy:
                cv2.rectangle(frame, (int(xt), int(yt)), (int(xr), int(yr)), (255, 0, 255), 2)

            cv2.imshow('frame', frame)
            # cv2.waitKey(0)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif frame_count.count > 64:
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
