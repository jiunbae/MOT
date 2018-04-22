import argparse
from pathlib import Path
from urllib import request
from zipfile import ZipFile
from os import remove

from bs4 import BeautifulSoup as bs


def main(des, prefix='http://cvlab.hanyang.ac.kr/tracker_benchmark/'):
    def targets(url):
        response = request.urlopen(url)
        soup = bs(response.read(), 'html.parser')

        for table in soup.find_all('table', 'seqtable'):
            for tag in table.find_all('a'):
                yield tag.get('href')

    def download(url, to):
        print ('Downloading {} ...'.format(url))
        request.urlretrieve(url, to)

    def extract(target, to):
        print ('Extracting {} ...'.format(target))
        with ZipFile(target) as z:
            z.extractall(to)

    for target in targets(prefix + 'datasets.html'):
        path = Path(des) / Path(target).name
        download(prefix + target, str(path))
        extract(str(path), des)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="dataset download destination", type=str, default='./train')

    args = parser.parse_args()

    main(args.dir)
