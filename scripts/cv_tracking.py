import cv2
import re
import numpy as np
import imutils
from os import listdir
from os.path import isfile, join

data = 'train/BlurCar1/'

scale = 600

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']

tracker_create = lambda type: {
    'BOOSTING': cv2.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'TLD': cv2.TrackerTLD_create,
    'MEDIANFLOW': cv2.TrackerMedianFlow_create,
    'GOTURN': cv2.TrackerGOTURN_create,
}[type]()

with open(data + 'groundtruth_rect.txt') as f:
    a = list(map(lambda line: list(map(int, re.split(';|,| |\t', line.replace('\n', '')))), f.readlines()))

for tracker_type in tracker_types:
    tracker = tracker_create(tracker_type)
    print ('processing', tracker_type)

    tracker_init = False
    with open(data + 'opencv_rect_{}.txt'.format(tracker_type), 'w') as f:
        for file in sorted(listdir(data + "img")):
            if not isfile(join(data + "img", file)): continue
            img = cv2.imread(join(data + "img", file))
            # img = imutils.resize(img, width=scale)
            # img = cv2.GaussianBlur(img, (5, 5), 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            if not tracker_init:
                rect = np.array(a[0])# / (len(img) / scale)
                tracker_init = tracker.init(img, tuple(rect.astype(int)))
            else:
                _, rect = tracker.update(img)
    
            rect = np.array(rect)# * (len(img) / scale)
            f.write(','.join(map(str, rect.astype(int))) + '\n')
