import argparse
from pathlib import Path

from lib.tracker import Tracker
from utils.data import Dataset, MOT


def main(args: argparse.Namespace):
    dataset = Path(args.dataset)
    dest = Path(args.dest)

    for sequence in dataset.iterdir():
        tracker = Tracker()
        loader = Dataset(str(sequence), MOT)

        with open(str(dest.joinpath('{}.txt'.format(sequence.stem))), 'w') as file:
            for frame, gt_boxes, gt_ids, det_boxes, det_scores in loader:
                for target in tracker.update(frame, det_boxes, det_scores):
                    file.write(', '.join(map(str, [frame, target.track_id, *target.tlwh, 1, -1, -1, -1])) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi Object Tracking')
    parser.add_argument("--dataset", dest='dataset', type=str, default='../datasets/MOT17/train',
                        help="train data path")
    parser.add_argument("--dest", dest='dest', type=str, default='./results',
                        help="result destination")

    arguments = parser.parse_args()

    main(arguments)
