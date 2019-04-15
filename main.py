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
        for sequence in dataset.iterdir():

            task.set_description(sequence.stem)

            tracker = Tracker(min_score=.0)
            loader = data.Dataset(str(sequence), data.MOT)

            # Mask R-CNN Detection Support
            if args.support is not None:
                support = Path(args.support).joinpath('{}.txt'.format(sequence.stem))
                support = pd.read_csv(str(support), header=None).values
            else:
                support = None

            with open(str(dest.joinpath('{}.txt'.format(sequence.stem))), 'w') as file:
                for frame, (image, boxes, scores, *_) in enumerate(tqdm(loader)):

                    # Mask R-CNN Detection Support
                    if support is not None:
                        selected = support[support[:, 0] == loader.index - 1]
                        support_boxes = selected[:, 2:6]
                        support_scores = selected[:, 6]

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
    parser.add_argument("--dest", type=str, default='./results',
                        help="result destination")
    parser.add_argument("--cache", action='store_true', default=False,
                        help="Use previous results for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Manual seed")

    parser.add_argument("--support", type=str, default=None,
                        help="Support detection")
    parser.add_argument("--support-only", action='store_true', default=False,
                        help="Support detection only")

    arguments = parser.parse_args()

    init(arguments.seed)

    main(arguments)
