import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.googlenet import GoogLeNet
from utils import box


class Identifier(nn.Module):
    def __init__(self, parts: int = 8):
        super(Identifier, self).__init__()
        self.parts = parts

        self.epoch, self.lr = 0, .01

        self.backbone = GoogLeNet()
        self.bridge = nn.Conv2d(832, 512, 1)
        self.branch = nn.Conv2d(512, self.parts, 1)

        self.shape = (80, 160)

        # Network
        for i in range(self.parts):
            setattr(self, 'linear{}'.format(i + 1), nn.Linear(512, 64))

        # TODO: Check CUDA available
        self.cuda()
        self.eval()

    @staticmethod
    def transform(image: np.ndarray) \
            -> np.ndarray:
        image = image.astype(np.float32)
        image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
        image = image.transpose((2, 0, 1))
        return image

    @staticmethod
    def forward_handler(feature: torch.Tensor, weights: torch.Tensor):
        mask = feature * torch.unsqueeze(weights, 1)
        pool = F.avg_pool2d(mask, mask.size()[2:4])
        return pool.view(pool.size(0), -1)

    def forward(self, x: torch.Tensor):
        feature = self.backbone(x)
        feature = self.bridge(feature)

        weights = torch.sigmoid(self.branch(feature))

        features = list(
            map(lambda i:
                getattr(self, 'linear{}'.format(i + 1))(
                    self.forward_handler(feature, weights[:, i])
                ), range(self.parts))
        )

        concat = torch.cat(features, 1)
        normed = concat / torch.clamp(
            torch.norm(concat, 2, 1, keepdim=True), min=1e-6
        )

        return normed

    def extract(self, image: np.ndarray, boxes: np.ndarray) \
            -> torch.Tensor:
        if not boxes.size:
            return torch.FloatTensor()

        boxes = box.clip(np.round(boxes).astype(np.int), image.shape)

        patches = map(lambda b: image[b[1]:b[3], b[0]:b[2]], boxes)
        patches = map(lambda p: self.transform(cv2.resize(p, self.shape)), patches)

        with torch.no_grad():
            # TODO: Check CUDA available
            var = torch.autograd.Variable(
                torch.from_numpy(np.asarray(list(patches), dtype=np.float32))
            ).cuda()
            result = self(var)

        return result.data.cpu().numpy()

    def load(self, weights: str = 'data/googlenet_part8_all_xavier_ckpt_56.h5'):

        # TODO: Remove this part after change weights
        def _wrapper_(key: str) \
                -> str:
            return key.replace('backbone', 'feat_conv')\
                .replace('bridge', 'conv_input_feat')\
                .replace('branch', 'conv_att')\
                .replace('linear', 'linear_feature')

        import h5py
        with h5py.File(weights, mode='r') as file:

            for k, v in filter(lambda kv: _wrapper_(kv[0]) in file, self.state_dict().items()):
                param = torch.from_numpy(np.asarray(file[_wrapper_(k)]))
                if v.size() == param.size():
                    v.copy_(param)

            epoch = file.attrs['epoch'] if 'epoch' in file.attrs else -1

            lr = file.attrs.get('lr', -1)
            lr = np.asarray([lr] if lr > 0 else [], dtype=np.float)

            self.epoch, self.lr = epoch, lr

            return self
