from os import path, listdir
import argparse

import torch
import numpy as np
from tqdm import tqdm

from tracker.mot_tracker import OnlineTracker
from datasets.mot_seq import get_loader


def seed(v: int = 42):
    np.random.seed(v)
    torch.manual_seed(v)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(v)


def eval_seq(dataloader, name=''):
    tracker = OnlineTracker()

    for fid, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=name):
        frame, det_rect, det_scores, gt_rect, gt_id = batch

        for target in tracker.update(frame, det_rect, det_scores):
            yield fid+1, target.track_id, target.tlwh


def main(data_root, seqs, dest):
    if seqs[0] == 'all':
        seqs = listdir(data_root)
    for seq in seqs:
        with open(path.join(dest, '{}.txt'.format(seq)), 'w') as f:
            for fid, tid, rect in eval_seq(get_loader(data_root, seq), seq):
                if tid < 0:
                    continue
                f.write(', '.join(map(str, [fid, tid, *rect]))+', 1, -1, -1, -1\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="train data path", dest='path', type=str, default='../datasets/MOT16/train')
    parser.add_argument("--seqs", help="sequences on dataset, split by comma", dest='seqs', type=str, default='all')
    parser.add_argument("--dest", help="result destination", dest='dest', type=str, default='./results')
    args = parser.parse_args()

    main(data_root=args.path, seqs=args.seqs.split(','), dest=args.dest)
