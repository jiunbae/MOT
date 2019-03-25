import argparse
from pathlib import Path

from tqdm import tqdm

from lib.tracker import Tracker
from utils.data import Dataset, MOT


def main(args: argparse.Namespace):
    dataset = Path(args.dataset)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(list(dataset.iterdir()))) as t:
        for sequence in dataset.iterdir():

            t.set_description(str(sequence))

            tracker = Tracker()
            loader = Dataset(str(sequence), MOT)

            with open(str(dest.joinpath('{}.txt'.format(sequence.stem))), 'w') as file:
                for frame, (image, gt_boxes, gt_ids, det_boxes, det_scores) in enumerate(tqdm(loader)):
                    for target in tracker.update(image, det_boxes, det_scores):
                        file.write(', '.join(map(str, [frame, target.id, *target.to_tlwh, 1, -1, -1, -1])) + '\n')

            t.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi Object Tracking')
    parser.add_argument("--dataset", dest='dataset', type=str, default='../datasets/MOT17/train',
                        help="train data path")
    parser.add_argument("--dest", dest='dest', type=str, default='./results',
                        help="result destination")

    arguments = parser.parse_args()

    main(arguments)
