import os
os.environ.setdefault('YOLO_OFFLINE','true')
os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/solar-yolo-tests')
os.environ.setdefault('MPLCONFIGDIR','/tmp/solar-yolo-tests/mpl')
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
import cv2
from pycocotools import mask as mu
from ultralytics.utils.instance import Instances
from ultralytics.models.yolo.segment import SegmentationTrainer,SegmentationValidator,SegmentationPredictor
from models.yolo_data_pipeline import CocoFormat,DiskTrainer,DiskValidator,DiskPredictor,disk_stats


class DataPipelineTests(unittest.TestCase):
    def test_coco_masks_keep_instances_and_match_evaluator(self):
        polygons=np.array([[[3.2,4.7],[25.8,4.7],[17.3,24.1]],[[10.2,8.1],[28.4,8.1],[20.3,29.6]]],np.float32)
        instances=Instances(np.array([[3,4,26,25],[10,8,29,30]],np.float32),segments=polygons,bbox_format='xyxy',normalized=False)
        classes=np.zeros((2,1),np.float32)
        expected=np.stack([mu.decode(mu.merge(mu.frPyObjects([p.reshape(-1).tolist()],32,32))) for p in polygons])
        masks,out,cls=CocoFormat(mask_ratio=1,mask_overlap=False)._format_segments(instances,classes,32,32)
        np.testing.assert_array_equal(masks,expected)
        self.assertIs(out,instances);self.assertIs(cls,classes)
        self.assertTrue((masks[0]&masks[1]).any()) # identities preserved, not unioned
        half,_,_=CocoFormat(mask_ratio=2,mask_overlap=False)._format_segments(instances,classes,32,32)
        np.testing.assert_array_equal(half,np.stack([cv2.resize(m,(16,16)) for m in expected]))
        with self.assertRaises(ValueError):CocoFormat(mask_overlap=True)._format_segments(instances,classes,32,32)

    def test_train_validation_inference_normalization_agree(self):
        image=np.broadcast_to((np.arange(2048)%61+100).astype(np.uint8),(2048,2048)).copy()
        yy,xx=np.ogrid[:2048,:2048];disk=(yy-1024)**2+(xx-1024)**2<=900**2
        pixels=image[disk];hist=np.bincount(pixels,minlength=256);cdf=np.cumsum(hist)/pixels.size
        lo,mid,hi=np.searchsorted(cdf,[.16,.5,.84]);stats=disk_stats(image)
        self.assertEqual(stats,(float(mid),float(hi-lo)))
        rgb=np.repeat(image[:,:,None],3,axis=2);raw=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()/255
        batch={'img':raw,'im_file':['photo.jpeg']}
        with patch('models.yolo_data_pipeline.file_disk_stats',return_value=stats),patch.object(SegmentationTrainer,'preprocess_batch',side_effect=lambda b:dict(b)),patch.object(SegmentationValidator,'preprocess',side_effect=lambda b:dict(b)),patch.object(SegmentationPredictor,'preprocess',return_value=raw):
            train=DiskTrainer.preprocess_batch(object.__new__(DiskTrainer),batch)['img']
            val=DiskValidator.preprocess(object.__new__(DiskValidator),batch)['img']
            pred=DiskPredictor.preprocess(object.__new__(DiskPredictor),[rgb])
        torch.testing.assert_close(train,val);torch.testing.assert_close(train,pred)
        expected=(torch.from_numpy(image).float()-mid)/(hi-lo)
        torch.testing.assert_close(train[0,0],expected)
        self.assertEqual(disk_stats(np.full((2048,2048),128,np.uint8)),(128.,1.))
        with self.assertRaises(ValueError):disk_stats(np.zeros((512,512),np.uint8))
