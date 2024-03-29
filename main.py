import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

from lib.tracker import Tracker
from utils import init, data


def main(args: argparse.Namespace):
    dataset = Path(args.dataset)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(list(dataset.iterdir()))) as task:
        for sequence in sorted(dataset.iterdir() if not args.no_sequence else [dataset]):

            task.set_description(sequence.stem)

            tracker = Tracker()

            loader = data.Dataset(str(sequence), data.get(args.type), args.jump)

            with open(str(dest.joinpath('{}.txt'.format(sequence.stem))), 'w') as file:
                for frame, (image, boxes, scores, image_name) in enumerate(tqdm(loader)):

                    # Mask R-CNN Detection Support
                    if args.support is not None:
                        try:
                            support = Path(args.support).joinpath('{}.txt'.format(image_name))
                            support = pd.read_csv(str(support), header=None).values

                            support_boxes = support[:, 2:6]
                            support_boxes[:, 2:] -= support_boxes[:, :2]
                            support_scores = support[:, 1]

                            if args.support_only:
                                boxes = support_boxes
                                scores = support_scores
                            else:
                                boxes = np.concatenate([
                                    boxes,
                                    support_boxes,
                                ])
                                scores = np.concatenate([
                                    scores,
                                    support_scores,
                                ])
                        except pd.errors.EmptyDataError:
                            boxes = np.zeros((0, 4))
                            scores = np.zeros(0)

                    file.writelines(
                        map(lambda t: ', '.join(map(
                            str, [frame, t.id, *t.to_tlwh, 1, -1, -1, -1, -1]
                        )) + '\n', tracker.update(image, boxes, scores))
                    )

            task.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi Object Tracking')
    parser.add_argument("--dataset", type=str, default='../datasets/MOT17/train',
                        help="train data path")
    parser.add_argument("--no-sequence", action='store_true', default=False,
                        help="Run for only one sequence")
    parser.add_argument("--dest", type=str, default='./results',
                        help="result destination")
    parser.add_argument("--support", type=str, default=None,
                        help="Support detection")
    parser.add_argument("--support-only", action='store_true', default=False,
                        help="Support detection only")
    parser.add_argument("--type", type=str, default='MOT',
                        help="Dataset type, default=MOT")

    parser.add_argument("--jump", type=int, default=10,
                        help="Jump")

    parser.add_argument("--cache", action='store_true', default=False,
                        help="Use previous results for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Manual seed")

    arguments = parser.parse_args()

    init(arguments.seed)

    main(arguments)
