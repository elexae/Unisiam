import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from model.unisiam import UniSiam
from model.df import df
from model.cnn import cnn
from transform.flow_augmentation import SimpleTrans
from dataset.torFlowNet import torFlowNet
from dataset.sampler import EpisodeSampler
from torchvision import transforms

@torch.no_grad()
def evaluate_fewshot(
    encoder, loader, n_way=5, n_shots=[1,5], n_query=15, classifier='LR', power_norm=False):

    encoder.eval()

    accs = {}
    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []

    for idx, (images, _) in enumerate(loader):

        images = images.cuda(non_blocking=True)
        f = encoder(images)
        f = f/f.norm(dim=-1, keepdim=True)

        if power_norm:
            f = f ** 0.5

        max_n_shot = max(n_shots)
        test_batch_size = int(f.shape[0]/n_way/(n_query+max_n_shot))
        sup_f, qry_f = torch.split(f.view(test_batch_size, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)
        qry_f = qry_f.reshape(test_batch_size, n_way*n_query, -1).detach().cpu().numpy()
        qry_label = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).numpy()

        for tb in range(test_batch_size):
            for n_shot in n_shots:
                cur_sup_f = sup_f[tb, :, :n_shot, :].reshape(n_way*n_shot, -1).detach().cpu().numpy()
                cur_sup_y = torch.arange(n_way).unsqueeze(1).expand(n_way, n_shot).reshape(-1).numpy()
                cur_qry_f = qry_f[tb]
                cur_qry_y = qry_label

                if classifier == 'LR':
                    clf = LogisticRegression(penalty='l2',
                                            random_state=0,
                                            C=1.0,
                                            solver='lbfgs',
                                            max_iter=1000,
                                            multi_class='multinomial')
                elif classifier == 'SVM':
                    clf = LinearSVC(C=1.0)
                elif classifier == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')

                clf.fit(cur_sup_f, cur_sup_y)
                cur_qry_pred = clf.predict(cur_qry_f)
                acc = accuracy_score(cur_qry_y, cur_qry_pred)

                accs[f'{n_shot}-shot'].append(acc)
    result_file = 'result.txt'
    result_file = open(result_file, 'a')
    for n_shot in n_shots:
        acc = np.array(accs[f'{n_shot}-shot'])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])
        print('classifier: {}, power_norm: {}, {}-way {}-shot \nacc: {:.2f}+{:.2f}'.format(
            classifier, power_norm, n_way, n_shot, mean*100, c95*100), file=result_file)
    return 


def build_fewshot_loader(mode='test'):

    assert mode in ['train', 'val', 'test']

    data_path, flow_size = '/home/dataset/raw', 5000

    test_transform = SimpleTrans(flow_size)
    
    test_dataset = torFlowNet(
        data_path=data_path,
        flow_size=flow_size,
        partition=mode,
        transform=test_transform)

    n_test_task, test_batch_size, n_way = 20, 1, 10
    test_sampler = EpisodeSampler(
        test_dataset.labels, n_test_task//test_batch_size, n_way, 25, test_batch_size)
    test_loader =torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_sampler, shuffle=False, drop_last=False, pin_memory=True)

    return test_loader

# few-shot knn test
def evaluate_knn(train_data_path, test_data_path, encoder, n_way):
    train_data = np.load(train_data_path, allow_pickle=True)
    test_data = np.load(test_data_path, allow_pickle=True)
    test_transform = SimpleTrans(5000)
    train_feature = train_data['feature']
    train_label = train_data['label']
    tf = []
    for f in train_feature:
        tf.append(test_transform(f[f!=0]))
    tf = np.array(tf)
    train_feature_encoded = np.array([])
    train_label_filtered = np.array([])
    for label in np.unique(train_label)[:n_way]:
        indexs = np.where(train_label == label)
        indexs = indexs[0]
        temp_feature = encoder(torch.stack(tf[indexs].tolist()))
        temp_feature = temp_feature.detach().cpu().numpy()
        if len(train_feature_encoded) == 0:
            train_feature_encoded = temp_feature
        else:
            train_feature_encoded = np.vstack((train_feature_encoded, temp_feature))
        if len(train_label_filtered) == 0:
            train_label_filtered = train_label[indexs]
        else:
            train_label_filtered = np.hstack((train_label_filtered, train_label[indexs]))
    train_feature = train_feature_encoded
    
    test_feature = test_data['feature']
    test_label = test_data['label']
    tf = []
    for f in test_feature:
        tf.append(test_transform(f[f!=0]))
    tf = np.array(tf)
    test_feature_encoded = np.array([])
    test_label_filtered = np.array([])
    for label in np.unique(train_label)[:n_way]:
        indexs = np.where(test_label == label)
        indexs = indexs[0]
        temp_feature = encoder(torch.stack(tf[indexs].tolist()))
        temp_feature = temp_feature.detach().cpu().numpy()
        if len(test_feature_encoded) == 0:
            test_feature_encoded = temp_feature
        else:
            test_feature_encoded = np.vstack((test_feature_encoded, temp_feature))
        if len(test_label_filtered) == 0:
            test_label_filtered = test_label[indexs]
        else:
            test_label_filtered = np.hstack((test_label_filtered, test_label[indexs]))
    test_feature = test_feature_encoded
    features = np.vstack((train_feature, test_feature))
    labels = np.hstack((train_label_filtered, test_label_filtered))
    np.savez('/home/dataset/temp/data.npz', feature=features.tolist(), label=labels)
    
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2, metric='cosine', algorithm='brute')
    clf.fit(train_feature, train_label_filtered)
    y_pred = clf.predict(test_feature)
    acc = accuracy_score(test_label_filtered, y_pred)
    micro_pre = precision_score(test_label_filtered, y_pred, average='micro', zero_division=True)
    macro_pre = precision_score(test_label_filtered, y_pred, average='macro', zero_division=True)
    micro_recall = recall_score(test_label_filtered, y_pred, average='micro', zero_division=True)
    macro_recall = recall_score(test_label_filtered, y_pred, average='macro', zero_division=True)
    micro_f1 = f1_score(test_label_filtered, y_pred, average='micro', zero_division=True)
    macro_f1 = f1_score(test_label_filtered, y_pred, average='macro', zero_division=True)
    print(f'accuracy : {acc}, micro_precision : {micro_pre}, macro_precision : {macro_pre}, micro_recall : {micro_recall}, macro_recall : {macro_recall}, micro_f1score : {micro_f1}, macro_f1score : {macro_f1}')

    cfm = confusion_matrix(test_label_filtered, y_pred, labels=clf.classes_) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(20,20))
    disp.plot(ax=ax, xticks_rotation='vertical')
    
    plt.savefig('cfm.png')

if __name__ == '__main__':
    
    # load model
    encoder = df((1,5000), 4096)

    model = UniSiam(encoder=encoder, lamb=0.1, temp=2, dim_hidden=None, dist=False)

    model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.cuda()
    msg = model.load_state_dict(torch.load('./trained_model/noise_injection/1/last.pth')['model'])
    model.eval()
    encoder = model.encoder

    #test_loader = build_fewshot_loader(mode='test')

    #evaluate_fewshot(encoder, test_loader, n_way=10, n_shots=[5,], n_query=20, classifier='KNN', power_norm=False)

    evaluate_knn('/home/dataset/AWF/few-shot/train.npz', '/home/dataset/AWF/few-shot/test.npz', encoder, 100)