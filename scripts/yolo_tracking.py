from os import listdir
from os.path import isfile, join
import cv2
import re
import math

import numpy as np
from darkflow.net.build import TFNet

options = {
    "model": "cfg/v1/yolo-full.cfg", 
    "load": "bin/yolo-full.weights",
    "threshold": 0.1
}

data = '../train/BlurCar1/'
tfnet = TFNet(options)

# Utils
# - center position of rectangle
center = lambda tar: ((tar[0] + tar[2]) / 2, (tar[1] + tar[3]) / 2)
# - check overlap rectangle and position
overlap = lambda tar, obj: tar[0] < obj[0] and obj[0] < tar[2] and tar[1] < obj[1] and obj[1] < tar[3]
# - distance two position
distance = lambda tar, obj: math.sqrt((tar[0] - obj[0])**2 + (tar[1] - obj[1])**2)
# - tuplize rectangle
tuplize = lambda tar: (tar[0], tar[1], tar[2], tar[3])

with open(data + 'groundtruth_rect.txt') as f:
    a = list(map(lambda line: list(map(int, re.split(';|,| |\t', line.replace('\n', '')))), f.readlines()))
    box = (a[0][0], a[0][1], a[0][0] + a[0][2], a[0][1] + a[0][3])
# Set target object
objects = [{
    'class': None,
    'position': box,
    'follow': True,
    'begin': None,
    'skip': [],
    'end': None,
}]

rects = []

# Process
for b, file in zip(a, sorted(listdir(data + "img"))):
    if not isfile(join(data + "img", file)): continue
    img = cv2.imread(join(data + "img", file))
    print (file)

    for result in tfnet.return_predict(img):
        rect = (result['topleft']['x'],result['topleft']['y'],result['bottomright']['x'],result['bottomright']['y'])
        update = False
        for i, obj in enumerate(objects):
            if overlap(rect, center(obj['position'])) and (result['label'] == obj['class'] or not obj['class']):
                if not obj['class']:
                    obj['class'] = result['label']
                    obj['begin'] = file
                obj['position'] = rect
                update = True
        if not update:
            objects.append({
                'class': result['label'],
                'position': rect,
                'follow': False,
                'begin': file,
                'skip': [],
                'end': None,
            })

    for obj in objects:
        if not update:
            obj['skip'].append(file)

    rects.append(tuplize(objects[0]['position']))
    # # Showcase
    # print (file, '=======================')
    # print (objects)
    # cv2.rectangle(img, *tuplize(objects[0]['position']), (0,121, 255), 3)
    # cv2.imwrite(join(data + 'out', file), img)

with open(data + 'result_rect.txt', 'w') as f:
    f.write('\n'.join([','.join(map(str, r)) for r in rects]))
