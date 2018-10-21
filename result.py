import argparse
from os import listdir, path
from shutil import copy2

def main(ds, dest):
	for seq in listdir(ds):
		if path.exists(path.join(ds, seq, 'result.csv')):
			copy2(path.join(ds, seq, 'result.csv'), path.join(dest, seq+'.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="train data path", dest='path', type=str, default='../datasets/MOT16/train')
    parser.add_argument("--dest", help="result file destination", dest='dest', type=str, default='./results')
    args = parser.parse_args()

    main(args.path, args.dest)
