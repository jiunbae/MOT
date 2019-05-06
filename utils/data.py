from typing import Type, Tuple
from pathlib import Path
from configparser import ConfigParser

import numpy as np
import pandas as pd
from torch.utils import data
import skimage


class BaseLoader:
    def __init__(self, root_dir: str,
                 *args, **kwargs):
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
            - Detection boxes       (np.ndarray)
            - Detection scores      (np.ndarray)
            - Ground truths boxes   (np.ndarray) (Optional)
            - Ground truths ids     (np.ndarray) (Optional)
        """
        pass

    @classmethod
    def get(cls, loader: str):
        return {
            klass.__name__.lower(): klass for klass in cls.__subclasses__()
        }[loader.lower()]


class MOT(BaseLoader):
    GT = 'gt/gt.txt'
    DET = 'det/det.txt'

    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self.root = Path(root_dir)

        parser = ConfigParser()

        with open(str(self.root.joinpath('seqinfo.ini'))) as file:
            parser.read_file(file)

        self.sequence = parser['Sequence']
        img_dir = self.sequence.get('imDir', 'img1')
        img_ext = self.sequence.get('imExt', '.jpg')

        self.images = list(sorted(self.root.joinpath(img_dir).glob('*' + img_ext)))

        self.det, self.gt = None, None

        if self.root.joinpath(self.GT).is_file():
            self.gt = pd.read_csv(str(self.root.joinpath(self.GT)), header=None).values

        if self.root.joinpath(self.DET).is_file():
            self.det = pd.read_csv(str(self.root.joinpath(self.DET)), header=None).values

    def __len__(self):
        return int(self.sequence.get('seqLength', 0))

    def __getitem__(self, idx: int) \
            -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image = self.images[idx]
        index = int(image.stem)

        gt, det = np.zeros((1, 6)), np.zeros((1, 7))

        if self.gt is not None:
            gt = self.gt[self.gt[:, 0] == index]

        if self.det is not None:
            det = self.det[self.det[:, 0] == index]

        return str(image), det[:, 2:6], det[:, 6], gt[:, 2:6], gt[:, 1],


class DETECTION(BaseLoader):
    """Detection Dataset

    Default format for detection results

    Each sequence contains detection result by each image.
    Each detection result contains:
        - image_id (auto generate)
        - class_id
        - score
        - *box (x, y, w, h)
        - (Optional)

    This loader prepend image id and merge all detections,
    for handler all gt in sequence.
    """
    GT = '../../detections/{}'

    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self.root = Path(root_dir)
        self.sequence = self.root.stem

        self.images = list(sorted(self.root.glob('*.png')))

        self.gt = np.empty((0, 7))

        for idx, gt_file in enumerate(sorted(
                self.root.joinpath(self.GT.format(self.sequence)).glob('*.txt')
        ), 1):
            values = pd.read_csv(str(gt_file), header=None, sep=',').values
            indices = np.empty(np.size(values, 0)); indices.fill(idx)
            self.gt = np.concatenate((self.gt, np.column_stack((indices, values))))

    def __len__(self):
        return np.size(self.gt, 0)

    def __getitem__(self, idx: int) \
            -> Tuple[str, np.ndarray, np.ndarray]:
        image = self.images[idx]
        index = int(image.stem)

        gt = np.zeros((1, 6))

        if self.gt is not None:
            gt = self.gt[self.gt[:, 0] == index]

        return str(image), gt[:, 3:7], gt[:, 2]


class KITTI(BaseLoader):
    GT = '../../label_02/{}.txt'

    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self.root = Path(root_dir)
        self.sequence = self.root.stem

        self.images = list(sorted(self.root.glob('*.png')))

        self.gt = None

        gt_file = self.root.joinpath(self.GT.format(self.sequence))

        if gt_file.is_file():
            df = pd.read_csv(str(gt_file), header=None, sep=' ')
            sub = df[(df[2] == 'Van') | (df[2] == 'Car') | (df[2] == 'Truck')]
            self.gt = np.column_stack([
                sub.iloc[:, :2].values,
                sub.iloc[:, 6:10].values,
            ])
            # Define all detection score 1.
            self.gt[:, 1] = 1.

    def __len__(self):
        return np.size(self.gt, 0)

    def __getitem__(self, idx: int) \
            -> Tuple[str, np.ndarray, np.ndarray]:
        image = self.images[idx]
        index = int(image.stem)

        gt = np.zeros((1, 6))

        if self.gt is not None:
            gt = self.gt[self.gt[:, 0] == index]

        return str(image), gt[:, 2:6], gt[:, 1]


class Dataset(data.Dataset):
    def __init__(self, root_dir: str,
                 loader: Type[BaseLoader],
                 *args, **kwargs):
        """
        Arguments:
            root_dir    (string): Root directory of dataset
            loader      (string): Dataset Loader, see also BaseLoader and implement.
        """
        assert issubclass(loader, BaseLoader), \
            "Loader must be inherited from BaseLoader"

        self.root = Path(root_dir)
        self.loader = loader(root_dir, *args, **kwargs)
        self.index = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self[self.index]
            self.index += 1
            return result
        except IndexError:
            self.index = 0
            raise StopIteration

    def __getitem__(self, idx):
        image, *data = self.loader[idx]

        return (skimage.io.imread(image)[:, :, ::-1], *data)


get = BaseLoader.get
