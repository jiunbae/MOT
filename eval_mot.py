from os import path

from tracker.mot_tracker import OnlineTracker
from datasets.mot_seq import get_loader

def eval_seq(dataloader):
    tracker = OnlineTracker()

    for fid, batch in enumerate(dataloader):
        frame, det_tlwhs, det_scores, gt_tlwhs, gt_id = batch

        for target in tracker.update(frame, det_tlwhs, None):
            yield fid+1, target.track_id, target.tlwh

def main(data_root, seqs):
    for seq in seqs:
        with open(path.join(data_root, seq, 'result.csv'), 'w') as f:
            for fid, tid, tlwh in eval_seq(get_loader(data_root, seq)):
                if tid < 0: continue
                x, y, w, h = tlwh
                f.write(','.join(map(str, [fid, tid, *tlwh]))+',1,-1,-1,-1\n')

if __name__ == '__main__':
    main(data_root = '../datasets/MOT16/train', seqs=['MOT16-02'])
