import torch
import torch.nn as nn
# from ultralytics import YOLO
from model.pose.pose_resnet import get_pose_net
from utils.preprocessing_utils import yolo_postprocess, crop_bounding_box, PoseStorage
from model.trackers import get_tracker
import numpy as np
from utils.meter import AverageMeter, Counter, Timer
from model.ObjectDetector.yolo import YoloV8Wrapper as YoloV8


class PoseExtractor(nn.Module):
    def __init__(self, is_half):
        super().__init__()
        # weights/yolov8n.pt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_half = True if self.device == 'cuda' and is_half else False
        yolo_overrides = {'device': self.device, 'half': True, 'conf': 0.25, 'classes': [0]}
        self.yolo = YoloV8(model_name="weights/yolov8n.pt", overrides=yolo_overrides)
        # self.yolo.fuse()
        # self.yolo.info(verbose=False)
        self.yolo.to(self.device)
        self.yolo_classes = [0]  # 0 - person 1 - head

        self.pose_detector = get_pose_net()
        self.pose_detector.init_weights("weights/pose_resnet_50_256x192.pth.tar")
        self.pose_detector.to(self.device)
        self.pose_storage = PoseStorage()

        self.tracker = get_tracker()

        self.timer = Timer()
        self.fps_yolo = AverageMeter()
        self.fps_tracker = AverageMeter()
        self.fps_pose = AverageMeter()

    def forward(self, frame, frame_number):
        self.timer.start()
        yolo_result = self.yolo(frame)
        boxes_xyxy_, _, _, combined_result = yolo_postprocess(yolo_result, self.yolo.scale)
        self.timer.stop()
        self.fps_yolo.update(self.timer.get_duration())

        self.timer.start()
        tracker_result = self.tracker.update(combined_result.cpu(), frame)
        tracker_result = np.array(tracker_result)
        boxes_xyxy = tracker_result[:, :4].astype(int)
        track_id = tracker_result[:, 4].astype(int)
        track_class = tracker_result[:, 5].astype(int)
        track_conf = tracker_result[:, 6].astype(int)
        self.timer.stop()
        self.fps_tracker.update(self.timer.get_duration())

        self.timer.start()
        cropped_images = crop_bounding_box(boxes_xyxy, frame)
        cropped_images = cropped_images.to(self.device)
        pose_point = self.pose_detector(cropped_images / 255)
        self.pose_storage.update(track_id, pose_point)
        # last n pose of active tracks
        pose_ids, track_pose_last_n, current_pose = self.pose_storage.get_pose(track_id)
        self.timer.stop()
        self.fps_pose.update(self.timer.get_duration())

        try:
            print(
                f"FPS :: yolo:{1 / self.fps_yolo.avg:.2f}, tracker:{1 / self.fps_tracker.avg:.2f}, pose:{1 / self.fps_pose.avg:.2f}")
        except:
            pass

        # track_id, active_track_bbox, track_class, track_conf, track_pose_last_n
        return track_id, boxes_xyxy, track_class, track_conf, track_pose_last_n, current_pose
