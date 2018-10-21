from os import path, listdir
import argparse

from pymot.pymot import MOTEvaluation

def load_result(data_path='../dataset/MOT16/train/MOT16-02'):
    len_seq = 0
    with open(path.join(data_path, "seqinfo.ini")) as f:
        len_seq = int(f.readlines()[4].strip().split("=")[1])

    gt = [[] for i in range(len_seq + 1)]
    result = [[] for i in range(len_seq + 1)]
    with open(path.join(data_path, 'gt', 'gt.txt')) as f:
        for seq, *data in [line.strip().split(',') for line in f.readlines()]:
            gt[int(seq)].append(data)
    
    with open(path.join(data_path, 'result.csv')) as f:
        for seq, *data in [line.strip().split(',') for line in f.readlines()]:
            result[int(seq)].append(data)

    return gt, result

def result_to_json(gt, result):
    hypotheses = {}
    groundtruth = {}
    num = 0
    hypotheses['frames'] = []
    hypotheses['class'] = "video"
    groundtruth['frames'] = []
    groundtruth['class'] = "video"
    for g in gt:
        groundtruth["frames"].append({"timestamp": num})
        groundtruth["frames"][num]["num"] = num
        groundtruth["frames"][num]["class"] = "frame"
        groundtruth["frames"][num]["annotations"] = []
        for obj in g:
            idx, x, y, w, h = obj[0], float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])
            groundtruth["frames"][num]["annotations"].append({"dco": False, "height": h, "width": w, "id": idx, "y": y, "x":x})
        num += 1
        
    num = 0
    for r in result:
        hypotheses["frames"].append({"timestamp": num})
        hypotheses["frames"][num]["num"] = num
        hypotheses["frames"][num]["class"] = "frame"
        hypotheses["frames"][num]["hypotheses"] = []
        for obj in r:
            idx, x, y, w, h = obj[0], float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])
            hypotheses["frames"][num]["hypotheses"].append({"height": h, "width": w, "id": idx, "y": y, "x":x})
        num+=1
    return groundtruth, hypotheses

def mot(groundtruth, hypotheses):
    evaluator = MOTEvaluation(groundtruth, hypotheses, 0.2)
    evaluator.evaluate()
    return evaluator.getMOTA(), evaluator.getMOTP()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="train data path", dest='path', type=str, default='../datasets/MOT16/train')
    parser.add_argument("--seqs", help="sequences on dataset, split by comma", dest='seqs', type=str, default='all')
    args = parser.parse_args()

    ds, seqs = args.path, args.seqs.split(',')

    if seqs[0] == 'all':
        seqs = listdir(ds)

    motas, motps = 0, 0
    for seq in seqs:
        mota, motp = mot(*result_to_json(*load_result(path.join(ds, seq))))
        print ("{}: MOTA: {}, MOTP: {}".format(seq, mota, motp))
        motas += mota
        motps += motp

    print ("{}: MOTA: {}, MOTP: {}".format('total', motas/len(seqs), motps/len(seqs)))
