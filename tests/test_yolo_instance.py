import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pycocotools import mask as mu
from models.yolo_instance import polygon_line,exclusive_rles,split_records,retain_periodic_checkpoint

class InstanceTests(unittest.TestCase):
    def test_periodic_checkpoint_preserves_completed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);last=directory/'last.pt';last.write_bytes(b'epoch4')
            trainer=SimpleNamespace(epoch=3,last=last,wdir=directory)
            retain_periodic_checkpoint(trainer)
            self.assertEqual(list(directory.glob('pq_epoch*.pt')),[])
            trainer.epoch=4;last.write_bytes(b'epoch5')
            retain_periodic_checkpoint(trainer)
            last.write_bytes(b'epoch6');trainer.epoch=5
            retain_periodic_checkpoint(trainer)
            self.assertEqual((directory/'pq_epoch005.pt').read_bytes(),b'epoch5')

    def test_polygon_coordinates_and_single_class(self):
        self.assertEqual(polygon_line({'segmentation':[[0,0,20,0,20,10]]},20,10),'0 0.000000000 0.000000000 1.000000000 0.000000000 1.000000000 1.000000000')
        with self.assertRaises(ValueError):polygon_line({'segmentation':[[0,0,1,0,1,1],[2,2,3,2,3,3]]},20,10)

    def test_confidence_order_overlap_and_area_after_exclusion(self):
        masks=np.zeros((3,10,10),bool)
        masks[0,1:6,1:6]=True;masks[1,3:8,3:8]=True;masks[2,0,0]=True
        pred=exclusive_rles(masks,[.5,.9,.99],.2,2,min_area=4)
        self.assertEqual(len(pred),2)
        a,b=[mu.decode(p).astype(bool) for p in pred]
        np.testing.assert_array_equal(a,masks[1])
        self.assertFalse((a&b).any())
        self.assertEqual(int(b.sum()),16)
        self.assertEqual(len(exclusive_rles(masks,[.5,.9,.99],.6,4,min_area=4)),1)

    def test_annotators_of_same_photo_stay_together(self):
        images=[dict(id=str(i),file_name=f'2020{m:02d}01000000Ch.jpeg',date_captured=f'2020-{m:02d}-01') for m in range(1,11) for i in range(m*2,m*2+2)]
        train,val=split_records({'images':images})
        self.assertFalse({r['file_name'] for r in train}&{r['file_name'] for r in val})
        self.assertEqual(len(val),4)
