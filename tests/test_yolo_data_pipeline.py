import os
os.environ.setdefault('YOLO_OFFLINE','true')
os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/solar-yolo-tests')
os.environ.setdefault('MPLCONFIGDIR','/tmp/solar-yolo-tests/mpl')
import unittest
import numpy as np
import cv2
from pycocotools import mask as mu
from ultralytics.utils.instance import Instances
from models.yolo_data_pipeline import CocoFormat


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
