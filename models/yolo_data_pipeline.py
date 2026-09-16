"""Controlled data-only ablations for the pinned Ultralytics 8.4.152 pipeline."""
from copy import copy
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics.data.augment import Format
from ultralytics.models.yolo.segment import SegmentationTrainer


class CocoFormat(Format):
    """Rasterize augmented instance polygons with the evaluation mask convention."""
    def _format_segments(self, instances, cls, w, h):
        if self.mask_overlap:
            raise ValueError('COCO experiment requires separate instance masks')
        masks=[]
        for polygon in instances.segments:
            rle=mu.merge(mu.frPyObjects([polygon.reshape(-1).astype(float).tolist()],h,w))
            mask=mu.decode(rle)
            if self.mask_ratio != 1:
                # Keep the baseline resizing operation to isolate rasterization.
                mask=cv2.resize(mask,(w//self.mask_ratio,h//self.mask_ratio))
            masks.append(mask)
        return np.asarray(masks,dtype=np.uint8),instances,cls


class CocoTrainer(SegmentationTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        dataset=super().build_dataset(img_path,mode,batch)
        dataset.format_class=CocoFormat
        dataset.transforms=dataset.build_transforms(hyp=copy(self.args))
        return dataset
