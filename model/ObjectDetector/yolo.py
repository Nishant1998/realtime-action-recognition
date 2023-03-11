import torch
import numpy as np
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops, DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
import cv2
import time
import torchvision.transforms as T
from torch import nn
from utils.meter import AverageMeter, Timer


class YoloV8Wrapper(nn.Module):
    def __init__(self, model_name="weights/yolov8n.pt", cfg=DEFAULT_CFG, overrides=None):
        super().__init__()
        # (960, 544) (224, 224)
        self.reshape_shape = (544, 544)
        self.model = YOLO(model_name).model

        self.args = get_cfg(cfg, overrides)
        self.scale = None

        # if self.args.device == 'cuda' and self.args.half==True:
        #     self.model = self.model.to(self.args.device).half()
        # elif self.args.device == 'cuda':
        #     self.model = self.model.to(self.args.device)
        #
        if self.args.device == 'cuda':
            self.model = self.model.to(self.args.device)
            if self.args.half:
                self.model = self.model.half()

    def forward(self, x):
        x, org_shape = self.image_preprocess(x)
        # x = self.model(x)

        if self.args.half:
            with torch.cuda.amp.autocast():
                x = self.model(x)
            x = (x[0].float(), [x[1][0].float(), x[1][1].float(), x[1][2].float()])
        else:
            x = self.model(x)

        x = ops.non_max_suppression(x,
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    classes=self.args.classes)[0]
        x = Results(boxes=x, orig_shape=org_shape)
        return x

    def image_preprocess(self, img):
        # numpy image
        org_shape = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.reshape_shape, interpolation=cv2.INTER_LINEAR)
        img_tensor = T.ToTensor()(img).unsqueeze(0)

        self.scale = (torch.tensor([[org_shape[1], org_shape[0], org_shape[1], org_shape[0]]]) / torch.tensor([[self.reshape_shape[1],self.reshape_shape[0],self.reshape_shape[1],self.reshape_shape[0]]]))

        # if self.args.device == 'cuda' and self.args.half == True:
        #     img_tensor = img_tensor.to(self.args.device).half()
        #     self.scale = self.scale.to(self.args.device)
        # elif self.args.device == 'cuda':
        #     img_tensor = img_tensor.to(self.args.device)
        #     self.scale = self.scale.to(self.args.device)

        if self.args.device == 'cuda':
            img_tensor = img_tensor.to(self.args.device)
            self.scale = self.scale.to(self.args.device)
            if self.args.half:
                img_tensor = img_tensor.half()

        return img_tensor, org_shape