import torch
import os
from utils.datasets import create_dataloader
import argparse
from utils.general import (check_img_size, colorstr)
from pathlib import Path
import sys


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def test(opt):
    # Trainloader
    train_path = "../dataset/MIO-TCD/data/images/train/"
    gs = 32
    batch_size = 16
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    single_cls = False
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                            hyp=opt.hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                            rect=opt.rect, rank=LOCAL_RANK, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad,
                                            prefix=colorstr('train: '), shuffle=True)
    nb = len(train_loader)  # number of batches
    print("number of batches: ", nb)
    idx = 0
    while True:
        rt_imgs, rt_targets, _, _ = next(train_loader.iterator)
        if not rt_imgs:
            break
        idx += 1
        print(idx)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low-ct.yaml', help='hyperparameters path')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    test(opt)
