import numpy as np
import torch
from scipy.misc import imresize
from sklearn.linear_model import Ridge

size = lambda obj: obj[2] * obj[3]
xclash = lambda tar, obj: 0 if tar[0] + tar[2] < obj[0] or obj[0] + obj[2] < tar[0] else max(tar[0] + tar[2] - obj[0], obj[0] + obj[2] - tar[0])
yclash = lambda tar, obj: 0 if tar[1] + tar[3] < obj[1] or obj[1] + obj[3] < tar[1] else max(tar[1] + tar[3] - obj[1], obj[3] + obj[3] - tar[1])
clash = lambda tar, obj: xclash(tar, obj) * yclash(tar, obj) / size(obj)
overlap = lambda tar, obj, rat: clash(tar, obj) >= rat
performonce = lambda t, o: sum([overlap(t, o, r) for r in x]) / len(x)
performance = lambda tar, obj, rat: [sum([overlap(t, o, r) for t, o in zip(tar, obj)])/len(tar) for r in rat]
perframe = lambda tar, obj: [sum([overlap(t, o, r) for r in x]) / len(x) for t, o in zip(tar, obj)]
scale = lambda box, ratio=0.01: [box[0] - box[2] * ratio, box[1] - box[3] * ratio, box[2] + box[2] * ratio * 2, box[3] + box[3] * ratio * 2]

def AUC(results, truths, x = np.arange(0.001, 1.001, 0.001)):
    return performance(results, truths, x)

def AUC_frame(results, truths):
    return perframe(results, truths)

def overlap_ratio(rect1, rect2):
    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def crop_image(img, bbox, img_size=107, padding=16, valid=False):
    
    x,y,w,h = np.array(bbox,dtype='float32')

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w/img_size
        pad_h = padding * h/img_size
        half_w += pad_w
        half_h += pad_h
        
    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)
    
    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        
        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    
    scaled = imresize(cropped, (img_size, img_size))
    return scaled

class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0,3,1,2).astype('float32')
        regions = regions - 128.
        return regions

class BBRegressor():
    def __init__(self, img_size, alpha=1000, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = Ridge(alpha=self.alpha)

    def train(self, X, bbox, gt):
        X = X.cpu().numpy()
        bbox = np.copy(bbox)
        gt = np.copy(gt)
        
        if gt.ndim==1:
            gt = gt[None,:]

        r = overlap_ratio(bbox, gt)
        s = np.prod(bbox[:,2:], axis=1) / np.prod(gt[0,2:])
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])

        X = X[idx]
        bbox = bbox[idx]

        Y = self.get_examples(bbox, gt)
        
        self.model.fit(X, Y)

    def predict(self, X, bbox):
        X = X.cpu().numpy()
        bbox_ = np.copy(bbox)

        Y = self.model.predict(X)
    
        bbox_[:,:2] = bbox_[:,:2] + bbox_[:,2:]/2
        bbox_[:,:2] = Y[:,:2] * bbox_[:,2:] + bbox_[:,:2]
        bbox_[:,2:] = np.exp(Y[:,2:]) * bbox_[:,2:]
        bbox_[:,:2] = bbox_[:,:2] - bbox_[:,2:]/2
        
        r = overlap_ratio(bbox, bbox_)
        s = np.prod(bbox[:,2:], axis=1) / np.prod(bbox_[:,2:], axis=1)
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])
        idx = np.logical_not(idx)
        bbox_[idx] = bbox[idx]
 
        bbox_[:,:2] = np.maximum(bbox_[:,:2], 0)
        bbox_[:,2:] = np.minimum(bbox_[:,2:], self.img_size - bbox[:,:2])

        return bbox_
    
    def get_examples(self, bbox, gt):
        bbox[:,:2] = bbox[:,:2] + bbox[:,2:]/2
        gt[:,:2] = gt[:,:2] + gt[:,2:]/2

        dst_xy = (gt[:,:2] - bbox[:,:2]) / bbox[:,2:]
        dst_wh = np.log(gt[:,2:] / bbox[:,2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y


