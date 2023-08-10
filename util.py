import math
import numpy as np
import torch
import torch.optim as optim
import colorsys
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from sklearn import manifold
from sklearn.metrics import pairwise_distances

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, cur_iter, total_iter):
    lr = args.lr
    eta_min = lr * 1e-3
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * cur_iter / total_iter)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
 
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
 
    return hls_colors
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
 
    return rgb_colors

def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

def tsne(all_features, all_labels, pic_name):
    distance_matrix = pairwise_distances(all_features, all_features, metric='cosine')
    tsne = manifold.TSNE(n_components=2, metric='precomputed').fit_transform(distance_matrix)
    plt.figure(figsize=(8, 8))
    unique_labels = np.unique(all_labels)
    labels_num = len(unique_labels)
    colors = list(map(lambda x: color(tuple(x)), ncolors(labels_num)))
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        labels = (all_labels == label)
        tsne_label = tsne[labels]
        plt.scatter(tsne_label[:, 0], tsne_label[:, 1], color=colors[i], label=label)
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(pic_name)

def preprocess(traffic_feature):
    feature = traffic_feature.tolist()
    i = 0
    while i < len(feature)-1:
        if feature[i] == 1 and feature[i+1] == -1:
            if i == 0:
                if feature[i+2] == 1:
                    del feature[i]
                    del feature[i]
                    i -= 1
            elif i+1 == len(feature)-1:
                if feature[i-1] == -1:
                    del feature[i]
                    del feature[i]
                    i -= 1
            else:
                if feature[i-1] == -1 and feature[i+2] == 1:
                    del feature[i]
                    del feature[i]
                    i -= 1
        i += 1
    return feature

def line_up(feature, length):
    feature = np.array(preprocess(feature))
    if len(feature) > length:
        feature = feature[:length]
    else:
        feature = np.hstack((feature, np.array([0]*(length-len(feature)))))
    return feature