import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pycocotools import mask as mu
from models import patch_unet as model


class PatchTrainingTests(unittest.TestCase):
    def test_foreground_coordinates_stay_paired_and_epochs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'train.jsonl'
            manifest.write_text('{}\n')
            image = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
            mask = np.zeros((2048, 2048), dtype=bool)
            mask[600, 600] = mask[1400, 1400] = True
            ds = model.PatchDataset(manifest, patch=32, samples_per_image=1)
            with patch.object(model, 'load_row', return_value=(image, mask)):
                snapshots = []
                for epoch in range(12):
                    ds.set_epoch(epoch)
                    x, y = ds[0]
                    rng = np.random.default_rng(np.random.SeedSequence([42, epoch, 0, 0]))
                    if rng.random() < .75:
                        self.assertEqual(y.sum().item(), 1)
                    snapshots.append(x.clone())
                self.assertTrue(any(not torch.equal(snapshots[0], x) for x in snapshots[1:]))
                ds.set_epoch(0)
                self.assertTrue(torch.equal(ds[0][0], snapshots[0]))

    def test_threshold_selection_scores_every_annotator_and_empty_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            prob = np.zeros((16, 16), dtype=np.float32)
            prob[2:6, 2:6] = .8
            prob[10:14, 10:14] = .4  # False positive removed by higher threshold.
            gtmask = np.zeros((16, 16), dtype=np.uint8)
            gtmask[2:6, 2:6] = 1
            rle = mu.encode(np.asfortranarray(gtmask))
            gt = {'a': [[rle], []], 'b': [[]]}
            args = SimpleNamespace(stride=512, infer_batch=1, min_area=1,
                                   max_candidates=20, selection_confidence=.2,
                                   selection_max_instances=10)
            rows = [{'physical_id': s, 'path': s} for s in ['a', 'b']]
            with patch.object(model, 'predict_image', side_effect=[prob, np.zeros_like(prob)]):
                metrics = model.validate(torch.nn.Identity(), rows, gt, 'cpu', args,
                                         [.3, .6], Path(tmp))
            self.assertAlmostEqual(metrics[0]['pq'], (2/3 + 0 + 1)/3)
            self.assertAlmostEqual(metrics[1]['pq'], 2/3)
            np.testing.assert_array_equal(np.load(Path(tmp)/'a.npy'), prob)

    def test_final_inference_restores_best_epoch_and_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'val.jsonl'
            manifest.write_text(json.dumps({'physical_id':'a'})+'\n')
            net = torch.nn.Conv2d(1, 1, 1)
            selected_weights = []
            def validate(net, rows, gt, device, args, thresholds, directory):
                directory.mkdir(parents=True, exist_ok=True)
                n = len(selected_weights)
                selected_weights.append(net.weight.detach().clone())
                np.save(directory/'a.npy', np.array([n]))
                return [{'threshold':.3, 'pq':.8 if n == 0 else .1}]
            ds = [(torch.ones(1,1,4,4), torch.ones(1,1,4,4))]
            dataset = unittest.mock.MagicMock()
            dataset.__len__.return_value = 1
            cli = ['patch_unet', '--train', str(manifest), '--val', str(manifest),
                   '--test', str(manifest), '--ground-truth', str(manifest),
                   '--raw-val', str(root/'raw_val'), '--raw-test', str(root/'raw_test'),
                   '--run-dir', str(root), '--epochs', '2', '--validate-every', '1', '--device', 'cpu']
            with patch('sys.argv', cli), patch.object(model, 'validation_ground_truth', return_value={}), \
                 patch.object(model, 'PatchDataset', return_value=dataset), \
                 patch.object(model, 'DataLoader', return_value=ds), \
                 patch.object(model, 'UNet', return_value=net), \
                 patch.object(model, 'validate', side_effect=validate), \
                 patch.object(model, 'write_predictions') as write:
                model.main()
            self.assertTrue(torch.equal(net.weight, selected_weights[0]))
            self.assertEqual(json.loads((root/'selected_model.json').read_text())['epoch'], 1)
            self.assertEqual(np.load(root/'probabilities/val/a.npy').tolist(), [0])
            self.assertEqual(write.call_count, 2)
            for call in write.call_args_list:
                self.assertEqual(call.args[5], .3)


if __name__ == '__main__':
    unittest.main()
