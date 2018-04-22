import argparse
from pathlib import Path
from urllib import request
from zipfile import ZipFile
from os import remove
from functools import partial
from multiprocessing import Pool

from bs4 import BeautifulSoup as bs


def process(des, prefix, target):
    def download(url, to):
        print ('Downloading {} ...'.format(url))
        request.urlretrieve(url, to)

    def extract(target, to):
        print ('Extracting {} ...'.format(target))
        with ZipFile(target) as z:
            z.extractall(to)

    path = Path(des) / Path(target).name
    
    download(prefix + target, str(path))
    extract(str(path), des)
    remove(str(path))
    return True


def main(des, core, prefix='http://cvlab.hanyang.ac.kr/tracker_benchmark/'):
    def targets(url):
        response = request.urlopen(url)
        soup = bs(response.read(), 'html.parser')

        for table in soup.find_all('table', 'seqtable'):
            for tag in table.find_all('a'):
                yield tag.get('href')



    pool = Pool()
    pool.map(partial(process, des, prefix), targets(prefix+'datasets.html'))
    pool.close()
    pool.join()
    # result = sum(map(process, targets(prefix+'datasets.html')))
    # print ('Total {} done', result)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="dataset download destination", type=str, default='./train')
    parser.add_argument("--core", help="multiprocessing core", dest='core', type=int, default=4)

    args = parser.parse_args()

    main(args.dir, args.core)
