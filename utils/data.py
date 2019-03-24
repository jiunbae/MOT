from typing import Type, Tuple
from pathlib import Path
from configparser import ConfigParser

import numpy as np
import pandas as pd
from torch.utils import data
import skimage

class BaseLoader:
    def __init__(self, root_dir: str):
        """
        Arguments:
            root_dir    (string): Root directory of dataset
        """
        self.root = root_dir

    def __len__(self) \
            -> int:
        """
        :return: Size of dataset
        """
        return 0

    def __getitem__(self, idx: int)\
            -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: Tuple include follows:
            - Image file path       (string)
            - Ground truths boxes   (np.ndarray)
            - Ground truths ids     (np.ndarray)
            - Detection boxes       (np.ndarray)
            - Detection scores      (np.ndarray)
        """
        pass


class MOT(BaseLoader):
    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self.root = Path(root_dir)

        parser = ConfigParser()

        with open(str(self.root.joinpath('seqinfo.ini'))) as file:
            parser.read_file(file)

        self.sequence = parser['Sequence']
        img_dir = self.sequence.get('imDir', 'img1')
        img_ext = self.sequence.get('imExt', '.jpg')

        self.images = list(self.root.joinpath(img_dir).glob('*' + img_ext))

        self.gt = pd.read_csv(str(self.root.joinpath('gt/gt.txt')), header=None).values[:, :-3]
        self.det = pd.read_csv(str(self.root.joinpath('det/det.txt')), header=None).values[:, :-3]

    def __len__(self):
        return self.sequence.get('seqLength', 0)

    def __getitem__(self, idx: int) \
            -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image = self.images[idx]
        index = int(image.stem)

        gt = self.gt[self.gt[:, 0] == index]
        det = self.det[self.det[:, 0] == index]

        return str(image), gt[:, 2:], gt[:, 1], det[:, 2:6], det[:, 6]


class Dataset(data.Dataset):
    def __init__(self, root_dir: str, loader: Type[BaseLoader]):
        """
        Arguments:
            root_dir    (string): Root directory of dataset
            loader      (string): Dataset Loader, see also BaseLoader and implement.
        """
        assert issubclass(loader, BaseLoader), \
            "Loader must be inherited from BaseLoader"

        self.root = Path(root_dir)
        self.loader = loader(root_dir)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        image, *data = self.loader[idx]

        return (skimage.io.imread(image), *data)
