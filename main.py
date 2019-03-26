import argparse
from pathlib import Path

from tqdm import tqdm

from lib.tracker import Tracker
from utils import init, data


def main(args: argparse.Namespace):
    dataset = Path(args.dataset)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(list(dataset.iterdir()))) as task:
        for sequence in dataset.iterdir():

            task.set_description(sequence.stem)

            tracker = Tracker()
            loader = data.Dataset(str(sequence), data.MOT)

            with open(str(dest.joinpath('{}.txt'.format(sequence.stem))), 'w') as file:
                for frame, targets in enumerate(map(lambda f: tracker.update(f[0], f[3], f[4]), tqdm(loader))):
                    file.writelines(
                        map(lambda t: ', '.join(map(
                            str, [frame, t.id, *t.to_tlwh, 1, -1, -1, -1, -1]
                        )) + '\n', targets)
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

    arguments = parser.parse_args()

    init(arguments.seed)

    main(arguments)
