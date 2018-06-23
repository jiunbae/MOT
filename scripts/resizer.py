import argparse
import multiprocessing as mp
from configparser import ConfigParser
from glob import glob
from pathlib import Path
from os import path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image

def task(file, to, size):
    Image.open(file)\
         .resize(size, Image.ANTIALIAS)\
         .save(path.join(to, path.basename(file)))

def main(input_dir, output_dir, ratio):
    config = ConfigParser()
    config.read(path.join(input_dir, 'seqinfo.ini'))

    if not path.isdir(input_dir): return

    size = np.array([int(config.get('Sequence', 'imWidth')),
                     int(config.get('Sequence', 'imHeight'))]) * ratio
    imDir = config.get('Sequence', 'imDir')
    imExt = config.get('Sequence', 'imExt')

    Path(path.join(output_dir, imDir)).mkdir(parents=True, exist_ok=True)
    Path(path.join(output_dir, 'gt')).mkdir(parents=True, exist_ok=True)

    # resize gt
    df = pd.read_csv(path.join(input_dir, 'gt', 'gt.txt'), header=None)
    df.values[:, 2:6] *= ratio
    df.to_csv(path.join(output_dir, 'gt', 'gt.txt'), header=None, index=None)

    # resize image
    pool = mp.Pool(mp.cpu_count())
    r = pool.map_async(partial(task,
                               to=path.join(output_dir, imDir),
                               size=size.astype(np.int)),
                       glob(path.join(input_dir, imDir, '*'+imExt)))
    pool.close()
    pool.join()
    r.get()

    config.set('Sequence', 'imWidth', str(int(size[0])))
    config.set('Sequence', 'imHeight', str(int(size[1])))

    with open(path.join(output_dir, 'seqinfo.ini'), 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input dir name", type=str)
    parser.add_argument("output", help="output dir name", type=str)
    parser.add_argument("--size", help="resize ratio", type=float, default=.5)

    args = parser.parse_args()
    main(args.input, args.output, args.size)
