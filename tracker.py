import os
import re

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from lib.options import options
from lib.model import MDNet
from lib.model import BinaryLoss
from lib.generator import gen_samples
from lib.generator import SampleGenerator
from lib.utils import RegionExtractor
from lib.utils import BBRegressor
from lib.utils import AUC

def load_data(data):
    def check_extension(file):
        return file.lower().endswith(('.png', '.jpg', '.jpeg'))
    def parse(line):
        return list(map(int, re.split(';|,| |\t', line.replace('\n', ''))))

    with open(os.path.join(data, 'groundtruth_rect.txt')) as f:
        truths = list(map(parse, f.readlines()))

    images = list(map(lambda x: os.path.join(data, 'img', x), filter(check_extension, sorted(os.listdir(os.path.join(data, 'img'))))))

    return images, truths

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, options['img_size'], options['padding'], options['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if torch.cuda.is_available():
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        feats = feat.data.clone() if i == 0 else torch.cat((feats,feat.data.clone()),0)
    return feats


def set_optimizer(model, lr_base, lr_mult=options['lr_mult'], momentum=options['momentum'], w_decay=options['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()
    
    batch_pos = options['batch_pos']
    batch_neg = options['batch_neg']
    batch_test = options['batch_test']
    batch_neg_cand = max(options['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer, neg_pointer = 0, 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()
        
        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)
        
        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), options['grad_clip'])
        optimizer.step()

def run_mdnet(images, init):
    # Init bbox
    target_bbox = np.array(init)
    result = np.zeros((len(images),4))
    result_bb = np.zeros((len(images),4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    model = MDNet(options['model_path'])
    if torch.cuda.is_available():
        model = model.cuda()
    model.set_learnable_params(options['ft_layers'])
    
    # Init criterion and optimizer 
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, options['lr_init'])
    update_optimizer = set_optimizer(model, options['lr_update'])

    # Load first image
    image = Image.open(images[0]).convert('RGB')

    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, options['n_bbreg'], options['overlap_bbreg'], options['scale_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, options['n_pos_init'], options['overlap_pos_init'])

    neg_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                target_bbox, options['n_neg_init']//2, options['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                target_bbox, options['n_neg_init']//2, options['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)
    feat_dim = pos_feats.size(-1)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, options['maxiter_init'])
    
    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, options['trans_f'], options['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:options['n_pos_update']]]
    neg_feats_all = [neg_feats[:options['n_neg_update']]]
    
    # Main loop
    for i in range(1, len(images)):

        # Load image
        image = Image.open(images[i]).convert('RGB')

        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, options['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > options['success_thr']
        
        # Expand search area at failure
        sample_generator.set_trans_f(options['trans_f'] if success else options['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        
        # Copy previous result at failure
        if not success:
            target_bbox = result[i-1]
            bbreg_bbox = result_bb[i-1]
        
        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox, 
                                       options['n_pos_update'],
                                       options['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox, 
                                       options['n_neg_update'],
                                       options['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > options['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > options['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(options['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, options['maxiter_update'])
        
        # Long term update
        elif i % options['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all ,0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, options['maxiter_update'])

    return result, result_bb

if __name__ == '__main__':
    dataset = [
        'Basketball',
        'Bird1',
        'Bolt',
        'Car1',
        'Diving',
        'Football',
        'Ironman',
        'Matrix',
        'Soccer',
        'Surfer'
    ]

    np.random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    total = 0
    for data in dataset:
        path = './train/{}'.format(data)
        images, truths = load_data(path)
        results, result_bb = run_mdnet(images, truths[0])
        x = np.arange(0.001, 1.001, 0.001)
        auc = AUC(results, truths, x)
        print ('AUC of {}'.format(path), sum(auc) / len(x))
        total += sum(auc) / len(x)
        with open(os.path.join(path, 'bounding_rect.txt'), 'w') as f:
            for result in results:
                f.write('{}\n'.format(','.join(map(str, result))))
    print ('AUC of all {}'.format(total / len(dataset)))
