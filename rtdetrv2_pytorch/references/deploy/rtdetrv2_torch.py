"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw, ImageFont

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, names, font, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        # 텍스트 위치 변경 원하면 b[0], b[1] 수정 필요, 현재 박스 안 좌측 상단에 그림
        for j,b in enumerate(box):
            if lab[j].item() == 0:
                draw.rectangle(list(b), outline='green', width=3)
                draw.text((b[0], b[1]), text=f"{names[lab[j].item()]} {round(scrs[j].item(),2)}", fill='green', font=font)
            else:
                draw.rectangle(list(b), outline='red', width=3)
                draw.text((b[0], b[1]), text=f"{names[lab[j].item()]} {round(scrs[j].item(),2)}", fill='red', font=font)

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
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
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)

    draw([im_pil], labels, boxes, scores, names, font)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
