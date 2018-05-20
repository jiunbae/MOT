import argparse
from glob import glob
from os import path
from pathlib import Path
from configparser import ConfigParser

from PIL import Image

SIZE = (960, 540)

def main(input_dir, output_dir):
    def parse(directory):
        config = ConfigParser()
        config.read(path.join(directory, 'seqinfo.ini'))
        return config.get('Sequence', 'imDir'), config.get('Sequence', 'imExt')

    for directory in glob(path.join(input_dir, '*')):
        if not path.isdir(directory): continue
        imgs, ext = parse(directory)
        for file in glob(path.join(input_dir, directory, imgs, '*' + ext)):
            p = path.join(output_dir, directory, imgs, file)
            Path(p).mkdir(parents=True, exist_ok=True)
            Image.open(file).resize(SIZE, Image.ANTIALIAS).save(p)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input dir name", type=str)
    parser.add_argument("output", help="output dir name", type=str)

    args = parser.parse_args()
    main(args.input, args.output)
