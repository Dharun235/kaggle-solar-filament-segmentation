"""Grouped-fold YOLOv8 instance segmentation; independent of the U-Net pipeline."""
from __future__ import annotations
import argparse,csv,json,random,sys,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pipeline_lib import physical_stem
from scripts.postprocess import pq_score_rles


def split_records(data,fold=0):
    months=sorted({im.get('date_captured',physical_stem(im['file_name'])[:6])[:7] for im in data['images']})
    random.Random(42).shuffle(months)
    folds={month:i%5 for i,month in enumerate(months)}
    train=[];val=[]
    for im in data['images']:
        month=im.get('date_captured',physical_stem(im['file_name'])[:6])[:7]
        (val if folds[month]==fold else train).append(im)
    assert not ({physical_stem(i['file_name']) for i in train}&{physical_stem(i['file_name']) for i in val})
    return train,val


def polygon_line(ann,width,height):
    polygons=ann['segmentation']
    if not isinstance(polygons,list) or len(polygons)!=1:
        raise ValueError('Expected one polygon per filament; refusing to split an instance')
    pts=np.asarray(polygons[0],dtype=float).reshape(-1,2)
    if len(pts)<3 or not np.isfinite(pts).all():raise ValueError('Invalid polygon')
    pts=pts/np.array([width,height])
    if (pts<0).any() or (pts>1).any():raise ValueError('Polygon outside image bounds')
    return '0 '+' '.join(f'{v:.9f}' for v in pts.ravel())


def prepare(data_root,out):
    data=json.loads((data_root/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    train,val=split_records(data)
    anns=defaultdict(list)
    for ann in data['annotations']:anns[ann['image_id']].append(ann)
    val_unique=list({physical_stem(im['file_name']):im for im in val}.values())
    dataset=out/'dataset'
    for split,records in [('train',train),('val',val_unique)]:
        for kind in ['images','labels']:(dataset/kind/split).mkdir(parents=True,exist_ok=True)
        for im in records:
            name=str(im['id'])
            if Path(name).name!=name:raise ValueError('Unexpected annotation ID')
            source=(data_root/'train/train_images'/im['file_name']).resolve()
            if not source.is_file():raise FileNotFoundError(source)
            target=dataset/'images'/split/f'{name}.jpeg'
            if not target.exists():target.symlink_to(source)
            lines=[polygon_line(a,im['width'],im['height']) for a in anns[im['id']]]
            (dataset/'labels'/split/f'{name}.txt').write_text('\n'.join(lines)+'\n')
    # JSON is valid YAML; Ultralytics requires a .yaml extension.
    config=dataset/'data.yaml'
    config.write_text(json.dumps(dict(path=str(dataset.resolve()),train='images/train',val='images/val',names={0:'filament'}),indent=2))
    info={'train_annotation_records':len(train),'train_photos':len({im['file_name'] for im in train}),'val_annotation_records':len(val),'val_photos':len(val_unique),'val_physical_ids':sorted(physical_stem(im['file_name']) for im in val_unique),'seed':42,'fold':0,'folds':5,'grouping':'physical image/month, same algorithm as scripts/prepare_data.py','annotation_policy':'Each training annotator is a separate sample. YOLO mAP uses one record per validation photo; final PQ uses every validation annotator.'}
    (out/'split.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k!='val_physical_ids'},indent=2),flush=True)
    return config,val_unique,{physical_stem(im['file_name']):[record for record in val if im['file_name']==record['file_name']] for im in val_unique},anns


def exclusive_rles(masks,scores,confidence,cap,min_area=80):
    occupied=np.zeros(masks.shape[1:],bool);result=[]
    for i in np.argsort(-np.asarray(scores),kind='stable'):
        if scores[i]<confidence:continue
        if len(result)>=cap:break
        mask=np.asarray(masks[i],bool)&~occupied
        if mask.sum()<min_area:continue
        occupied|=mask
        result.append(mu.encode(np.asfortranarray(mask,dtype=np.uint8)))
    return result


def prediction(model,path):
    r=model.predict(str(path),imgsz=2048,conf=.1,iou=.7,max_det=100,retina_masks=True,device=0,verbose=False)[0]
    if r.masks is None:return np.zeros((0,*r.orig_shape),bool),np.zeros(0)
    masks=r.masks.data.cpu().numpy()>.5
    if masks.shape[1:]!=tuple(r.orig_shape):raise ValueError('Expected native-resolution masks')
    return masks,r.boxes.conf.cpu().numpy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
    status('preparing')
    dataset,val,records,anns=prepare(args.data_root,out)
    if args.prepare_only:return
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available(),'GPU required'
    cfg=dict(model='yolov8s-seg.pt',epochs=30,imgsz=2048,batch=1,device=0,workers=2,seed=42,deterministic=True,optimizer='AdamW',lr0=.001,cos_lr=True,patience=30,amp=True,mask_ratio=2,overlap_mask=False,mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=.1,fliplr=.5,flipud=.5,scale=.1,translate=.05,plots=False,cache=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    status('training',epochs=cfg['epochs'])
    model=YOLO(cfg.pop('model'))
    model.train(data=str(dataset),project=str(out),name='train',exist_ok=False,**cfg)
    del model
    torch.cuda.empty_cache()
    candidates=[out/'train/weights/best.pt',out/'train/weights/last.pt']
    grid=[(c,k) for c in [.1,.2,.3,.4,.5] for k in [4,8,16,100]]
    trials=[]
    for checkpoint in candidates:
        model=YOLO(str(checkpoint));values={key:[] for key in grid}
        for n,im in enumerate(val,1):
            stem=physical_stem(im['file_name'])
            masks,scores=prediction(model,args.data_root/'train/train_images'/im['file_name'])
            truth=[[mu.merge(mu.frPyObjects(a['segmentation'],2048,2048)) for a in anns[rec['id']]] for rec in records[stem]]
            for conf,cap in grid:
                pred=exclusive_rles(masks,scores,conf,cap)
                values[(conf,cap)].extend(pq_score_rles(gs,pred) for gs in truth)
            status('validation',checkpoint=checkpoint.name,images_done=n,images_total=len(val))
            print(f'{checkpoint.name} validation {n}/{len(val)}',flush=True)
        for (conf,cap),pq in values.items():trials.append(dict(checkpoint=str(checkpoint),confidence=conf,max_instances=cap,pq=float(np.mean(pq))))
        del model
        torch.cuda.empty_cache()
    (out/'validation_grid.json').write_text(json.dumps(trials,indent=2))
    selected=max(trials,key=lambda r:r['pq']);selected['checkpoint_sha256']=hashlib.sha256(Path(selected['checkpoint']).read_bytes()).hexdigest()
    (out/'selected.json').write_text(json.dumps(selected,indent=2));print('SELECTED',selected,flush=True)
    model=YOLO(selected['checkpoint'])
    images=sorted((args.data_root/'test/test_images').glob('*.jpeg'))
    if len(images)!=180:raise ValueError(f'Expected180 test images, got{len(images)}')
    seen=0;rows=0
    with (out/'submission.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for n,path in enumerate(images,1):
            masks,scores=prediction(model,path)
            pred=exclusive_rles(masks,scores,selected['confidence'],selected['max_instances'])
            occ=np.zeros((2048,2048),bool)
            for i,rle in enumerate(pred,1):
                decoded=mu.decode(rle).astype(bool)
                assert decoded.any() and not (occ&decoded).any()
                occ|=decoded
                writer.writerow([f'{path.stem}_{i}',rle['counts'].decode('ascii')]);rows+=1
            seen+=bool(pred)
            status('test_inference',images_done=n,images_total=len(images))
    status('complete',validation_pq=selected['pq'],submission_rows=rows,images_with_predictions=seen,test_images=len(images),overlap_pixels=0)
    print('DONE',out/'submission.csv',flush=True)

if __name__=='__main__':main()
