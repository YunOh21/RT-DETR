"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
import cv2
from PIL import Image

from src.core import YAMLConfig

import torchvision.ops as ops

def draw(images, labels, boxes, scores, names, thrh=0.6):
    for i, im in enumerate(images):
        np_im = np.array(im.convert("RGB"))
        np_im = np_im[:, :, ::-1].copy()
        h_img, w_img, _ = np_im.shape

        scr = scores[i]
        lab = labels[i]
        box = boxes[i]

        mask = scr > thrh
        scr = scr[mask]
        lab = lab[mask]
        box = box[mask]

        if box.numel() == 0:
            out_im = Image.fromarray(np_im[:, :, ::-1])
            out_im.save(f"results_{i}.jpg")
            continue

        keep = ops.nms(box, scr, iou_threshold=0.5)
        scr = scr[keep]
        lab = lab[keep]
        box = box[keep]

        # x1 기준으로 정렬
        x1s = box[:, 0]
        order = torch.argsort(x1s)
        box = box[order]
        scr = scr[order]
        lab = lab[order]

        used_text_boxes = []

        def overlaps(r1, r2):
            x10, y10, x11, y11 = r1
            x20, y20, x21, y21 = r2
            return not (x11 <= x20 or x21 <= x10 or y11 <= y20 or y21 <= y10)

        num_box = box.size(0)

        if num_box == 1:
            b = box[0]
            cls_id = int(lab[0].item())
            score = float(scr[0].item())
            text = f"{names[cls_id]} {score:.2f}"

            x1, y1, x2, y2 = map(int, b.tolist())
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)

            cv2.rectangle(np_im, (x1, y1), (x2, y2), color, 2)

            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )

            tx = x1
            if tx + tw > w_img:
                tx = max(0, w_img - tw - 5)

            ty = y1 - 5
            if ty - th < 0:
                ty = y2 + th + 5
                if ty + baseline > h_img:
                    ty = min(max(th + 5, y1), h_img - baseline - 1)

            cv2.putText(
                np_im,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            out_im = Image.fromarray(np_im[:, :, ::-1])
            out_im.save(f"results_{i}.jpg")
            continue

        for idx, b in enumerate(box):
            cls_id = int(lab[idx].item())
            score = float(scr[idx].item())
            text = f"{names[cls_id]} {score:.2f}"

            x1, y1, x2, y2 = map(int, b.tolist())
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)

            cv2.rectangle(np_im, (x1, y1), (x2, y2), color, 2)

            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )

            tx = x1
            if tx + tw > w_img:
                tx = max(0, w_img - tw - 5)

            primary_is_below = (idx % 2 == 0)

            candidates = []

            ty_below = y2 + th + 5
            if ty_below + baseline <= h_img:
                tb_below = [tx, ty_below - th, tx + tw, ty_below + baseline]
                if primary_is_below:
                    candidates.append(("below", ty_below, tb_below))

            ty_above = y1 - 5
            if ty_above - th >= 0:
                tb_above = [tx, ty_above - th, tx + tw, ty_above + baseline]
                if not primary_is_below:
                    candidates.insert(0, ("above", ty_above, tb_above))
                else:
                    candidates.append(("above", ty_above, tb_above))

            if not candidates:
                if primary_is_below and ty_above - th >= 0:
                    candidates.append(("above", ty_above, tb_above))
                elif (not primary_is_below) and ty_below + baseline <= h_img:
                    candidates.append(("below", ty_below, tb_below))

            if not candidates:
                ty = min(max(th + 5, y1), h_img - baseline - 1)
                tb = [tx, ty - th, tx + tw, ty + baseline]
                candidates.append(("fallback", ty, tb))

            chosen_ty, chosen_tb = candidates[0][1], candidates[0][2]
            for _, cand_ty, cand_tb in candidates:
                if not any(overlaps(cand_tb, prev) for prev in used_text_boxes):
                    chosen_ty, chosen_tb = cand_ty, cand_tb
                    break

            y0, y1_box = chosen_tb[1], chosen_tb[3]
            if y0 < 0:
                shift = -y0
                chosen_ty += shift
                chosen_tb[1] += shift
                chosen_tb[3] += shift
            if y1_box > h_img:
                shift = y1_box - h_img
                chosen_ty -= shift
                chosen_tb[1] -= shift
                chosen_tb[3] -= shift

            used_text_boxes.append(chosen_tb)

            cv2.putText(
                np_im,
                text,
                (tx, chosen_ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        out_im = Image.fromarray(np_im[:, :, ::-1])
        out_im.save(f"results_{i}.jpg")
        

def main(args,):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self,) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    names = ["normal", "anomaly"]

    draw([im_pil], labels, boxes, scores, names)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,)
    parser.add_argument('-r', '--resume', type=str,)
    parser.add_argument('-f', '--im-file', type=str,)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
