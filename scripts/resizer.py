import argparse
from glob import glob
from os import path
from pathlib import Path
from configparser import ConfigParser
import multiprocessing as mp
from functools import partial

from PIL import Image

SIZE = (960, 540)

def task(output, directory):
    def parse():
        config = ConfigParser()
        config.read(path.join(directory, 'seqinfo.ini'))
        return config.get('Sequence', 'imDir'), config.get('Sequence', 'imExt')

    if not path.isdir(directory): return
    imgs, ext = parse()
    for file in glob(path.join(directory, imgs, '*' + ext)):
        p = path.join(output, path.basename(directory), imgs, path.basename(file))
        Path(path.dirname(p)).mkdir(parents=True, exist_ok=True)
        Image.open(file).resize(SIZE, Image.ANTIALIAS).save(p)

def main(input_dir, output_dir):
    pool = mp.Pool(mp.cpu_count())
    r = pool.map_async(partial(task, output_dir), glob(path.join(input_dir, '*')))
    pool.close()
    pool.join()
    r.get()

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input dir name", type=str)
    parser.add_argument("output", help="output dir name", type=str)

    args = parser.parse_args()
    main(args.input, args.output)
