from PIL import Image

from configparser import ConfigParser
from os import path, listdir
from collections import defaultdict

def configparse(file):
    parser = ConfigParser()
    with open(file) as f:
        parser.read_string(f.read())
    return parser

def configtruths(file):
    with open(file) as f:
        sequence = defaultdict(dict)
        objects = defaultdict(dict)
        for line in f.readlines():
            seq, obj, *data = line.strip().split(',')
            sequence[int(seq)][int(obj)] = objects[int(obj)][int(seq)] = list(map(float, data))
        return sequence, objects

def configimages(images, ext='.jpg'):
    for img in listdir(images):
        if img.endswith(ext):
            yield Image.open(path.join(images, img)).convert('RGB')

def parse(directory):
    config = configparse(path.join(directory, 'seqinfo.ini'))
    assert 'Sequence' in config
    seq = config['Sequence']
    sequence, objects = configtruths(path.join(directory, 'gt', 'gt.txt'))
    return {
        'width': int(seq['imWidth']),
        'height': int(seq['imHeight']),
        'name': seq['name'],
        'length': int(seq['seqLength']),
        'truths': sequence,
        'objects': objects,
        'images': configimages(path.join(directory, seq['imDir']), seq['imExt'])
    }
