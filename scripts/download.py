"""
script for download OTB-100

aurthor: Bae Jiun, Maybe
"""

import argparse
from pathlib import Path
from urllib import request
from zipfile import ZipFile
from os import remove
from functools import partial
from multiprocessing import Pool

from bs4 import BeautifulSoup as bs


def process(des, prefix, target):
    """process each dataset

    download, extract and remove
    """
    def download(url, obj):
        print('Downloading {} ...'.format(url))
        request.urlretrieve(url, obj)

    def extract(target, obj):
        print('Extracting {} ...'.format(target))
        with ZipFile(target) as z:
            z.extractall(obj)

    path = Path(des) / Path(target).name

    download(prefix + target, str(path))
    extract(str(path), des)
    remove(str(path))
    return str(path), True


def complete(obj, result):
    print (obj, result)


def main(des, core, prefix='http://cvlab.hanyang.ac.kr/tracker_benchmark/'):
    def targets(url):
        response = request.urlopen(url)
        soup = bs(response.read(), 'html.parser')

        for table in soup.find_all('table', 'seqtable'):
            for tag in table.find_all('a'):
                yield tag.get('href')

    pool = Pool(core)
    pool.map_async(partial(process, des, prefix), targets(prefix+'datasets.html'), callback=complete)
    pool.close()
    pool.join()

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="dataset download destination", type=str, default='./train')
    parser.add_argument("--core", help="multiprocessing core", dest='core', type=int, default=4)

    args = parser.parse_args()

    main(args.dir, args.core)
