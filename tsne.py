import sys
sys.path.append('/home/OWWF/')
import util
from util import tsne
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
from transform.flow_augmentation import SimpleTrans
from model.unisiam import UniSiam
from model.df import df
import torch

encoder = df((1,5000), 4096)

model = UniSiam(encoder=encoder, lamb=0.1, temp=0.1, dim_hidden=None, dist=False)

model.encoder = torch.nn.DataParallel(model.encoder)
model = model.cuda()
msg = model.load_state_dict(torch.load('./trained_model/noise_injection/1/last.pth')['model'])
model.eval()
encoder = model.encoder

data = np.load('/home/dataset/AWF/test.npz', allow_pickle=True)
size = 5000
label_num = 10
features = data['feature']
labels_name = data['label']
labels_unique = np.unique(labels_name)
test_transform = SimpleTrans(size)
features_filtered = []
features_numpy = []
labels_filtered = []
for label in labels_unique[:label_num]:
    indexs = np.where(labels_name == label)
    indexs = indexs[0]
    for index in indexs[:20]:
        feature = features[index]
        features_filtered.append(test_transform(feature))
        features_numpy.append(util.line_up(feature[feature!=0], 5000))
        labels_filtered.append(labels_name[index])
features = encoder(torch.stack(features_filtered))
features = features.detach().cpu().numpy()
#np.savez('encoded_feature.npz', feature=features.tolist(), label=labels_filtered)
tsne(features_numpy, np.array(labels_filtered), f'raw_{label_num}.png')
tsne(features, np.array(labels_filtered), f'encoded_{label_num}.png')