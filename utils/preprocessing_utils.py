from collections import defaultdict

import cv2
import numpy as np
import torch


def yolo_postprocess(result, scale, conf=0.0):
    boxes = result.boxes
    boxes_xyxy = boxes.xyxy * scale
    boxes_conf = boxes.conf
    boxes_cls = boxes.cls

    idx = torch.nonzero(boxes_conf > conf)
    if len(idx) == 1:
        idx = idx[0]
    else:
        idx = idx.squeeze()

    boxes_xyxy = boxes_xyxy[idx]
    boxes_cls = boxes_cls[idx]
    boxes_conf = boxes_conf[idx]
    combined_result = torch.cat((boxes_xyxy, boxes_conf.unsqueeze(1), boxes_cls.unsqueeze(1)), dim=1)
    boxes_xyxy = boxes_xyxy.detach().cpu().numpy().astype(np.int32)

    return boxes_xyxy, boxes_conf, boxes_cls, combined_result


def crop_bounding_box(bounding_boxes, image):
    cropped_images = []
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        # print((x1, y1, x2, y2))
        cropped_image = image[y1:y2, x1:x2]

        # Resize the cropped image to the desired shape (256, 128)
        resized_image = cv2.resize(cropped_image, (128, 256))
        cropped_images.append(torch.from_numpy(resized_image.transpose((2, 0, 1))))
    cropped_images = torch.stack(cropped_images)
    return cropped_images


class PoseStorage:
    def __init__(self, n=100):
        self.pose_sequence_data = defaultdict(list)
        self.n = n

    def update(self, track_id, track_pose):
        for i, id in enumerate(track_id):
            self.pose_sequence_data[id].append(track_pose[i])

    def get_pose(self, track_id):
        pose_sequence_data = []
        return_pose_ids = []
        current_pose = []
        for id in track_id:
            track_pose = self.pose_sequence_data[id][-1]
            current_pose.append(track_pose)

        for id in track_id:
            track_pose = self.pose_sequence_data[id]
            if len(track_pose) >= self.n:
                pose_sequence_data.append(torch.stack(track_pose[-self.n:]))
                return_pose_ids.append(id)

        if len(return_pose_ids) == 0:
            return None, None, torch.stack(current_pose)
        else:
            return return_pose_ids, torch.stack(pose_sequence_data), torch.stack(current_pose)