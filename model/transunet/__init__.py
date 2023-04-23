from .unet import UNet
# from .resunet import ResUNet
from .transunet import TransUnet
# from .dialunet import DialUnet
import torch.nn as nn

# from dpt.models import DPTDepthModel
# from .dpt.blocks import Interpolate

def get_model(args):
    if not hasattr(args, "channel"):
        channel = 32
    else:
       channel =  args.channel
    if "rgb" in args.exp_name:
        channel = 3
    if args.model == 'unet':
        model = UNet(channel, 3, dim=args.dim, norm=args.norm)
    if args.model == 'resunet':
        model = ResUNet(channel, 3, dim=args.dim, residual_num=args.residual_num, norm=args.norm)

    if args.model == 'transunet':
        model = TransUnet(channel, 3, dim=args.dim, residual_num=args.residual_num, norm=args.norm)
    if args.model == "dialunet":
        model = DialUnet(channel, 3, dim=args.dim, residual_num=args.residual_num, norm=args.norm,dropout=args.dropout)
    if args.model == 'dpt':
        features = 256
        non_negative = True
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        model = DPT(head)
    return model