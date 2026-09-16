"""Controlled data-only ablations for the pinned Ultralytics 8.4.152 pipeline."""
from copy import copy
from functools import lru_cache
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics.data.augment import Format
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator, SegmentationPredictor


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


def disk_stats(image):
    """Anthony's median / (p84-p16), measured on the original image before augmentation."""
    if image.ndim==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=image.shape
    if (h,w)!=(2048,2048):
        raise ValueError('Disk normalization expects original 2048x2048 images')
    yy,xx=np.ogrid[:h,:w]
    pixels=image[(yy-1024)**2+(xx-1024)**2<=900**2]
    cdf=np.cumsum(np.bincount(pixels,minlength=256))/pixels.size
    lo,mid,hi=np.searchsorted(cdf,[.16,.5,.84])
    return float(mid),float(max(hi-lo,1))


@lru_cache(maxsize=2048)
def file_disk_stats(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise FileNotFoundError(path)
    return disk_stats(image)


def normalize_tensor(tensor,stats):
    # Standard YOLO preprocessing has already divided image values by255.
    values=torch.as_tensor(stats,device=tensor.device,dtype=torch.float32)
    result=(tensor.float()*255-values[:,0,None,None,None])/values[:,1,None,None,None]
    return result.to(tensor.dtype)


class CocoTrainer(SegmentationTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        dataset=super().build_dataset(img_path,mode,batch)
        dataset.format_class=CocoFormat
        dataset.transforms=dataset.build_transforms(hyp=copy(self.args))
        return dataset


class DiskValidator(SegmentationValidator):
    def preprocess(self,batch):
        batch=super().preprocess(batch)
        batch['img']=normalize_tensor(batch['img'],[file_disk_stats(p) for p in batch['im_file']])
        return batch


class DiskTrainer(SegmentationTrainer):
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        batch['img']=normalize_tensor(batch['img'],[file_disk_stats(p) for p in batch['im_file']])
        return batch

    def get_validator(self):
        return DiskValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)


class DiskPredictor(SegmentationPredictor):
    def preprocess(self,images):
        if isinstance(images,torch.Tensor):
            raise ValueError('Disk predictor requires original image arrays, not preprocessed tensors')
        stats=[disk_stats(image) for image in images]
        return normalize_tensor(super().preprocess(images),stats)
