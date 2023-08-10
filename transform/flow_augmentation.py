import random
from typing import Any
import numpy as np
import torch

class SimpleTrans:

    def __init__(self, size) -> None:
        self.size = size
        
    def __call__(self, x) -> Any:

        x = x.tolist()
        if len(x) > self.size:
            x = x[:self.size]
        else:
            x = x + [0] * (self.size - len(x))
        
        x = torch.Tensor(x)
        return x.unsqueeze(0)
        
class WindowSlicing:

    def __init__(self) -> None:
        pass

class NoiseInjection:

    def __init__(self, scale=0.02, size=5000):
        self.scale = scale
        self.size = size

    def __call__(self, x):

        injection_size = max(round(self.scale * len(x)), 10)
        injection_pos = np.random.uniform(low=0, high=1, size=injection_size)
        injection_pos = [round(pos*len(x)) for pos in injection_pos]
        
        injection_pos.sort()
        injection_pos = [i+pos for i, pos in enumerate(injection_pos)]

        x = x.tolist()
        for pos in injection_pos:
            x.insert(pos, random.choice([1, -1]))

        if len(x) > self.size:
            x = x[:self.size]
        else:
            x = x + [0] * (self.size - len(x))

        x = torch.Tensor(x)
        return x.unsqueeze(0)
    
class MaskAug:
    
    def __init__(self, scale=0.01, size=5000):
        self.scale = scale
        self.size = size

    def __call__(self, x):

        mask_size = max(round(self.scale * len(x)), 10)
        mask_pos = np.random.uniform(low=0, high=1, size=mask_size)
        mask_pos = [int(pos*len(x)) for pos in mask_pos]

        x = x.tolist()
        for pos in mask_pos:
            x[pos] = 0
        
        if len(x) > self.size:
            x = x[:self.size]
        else:
            x = x + [0] * (self.size - len(x))

        x = torch.Tensor(x)
        return x.unsqueeze(0)
    
class DropAug:
    
    def __init__(self, scale=0.01, size=5000):
        self.scale = scale
        self.size = size

    def __call__(self, x):

        drop_size = max(round(self.scale * len(x)), 10)
        drop_pos = np.random.uniform(low=0, high=1, size=drop_size)
        drop_pos = [int(pos*len(x)) for pos in drop_pos]

        x = x.tolist()
        drop_pos.sort(reverse=True)
        for pos in np.unique(drop_pos):
            try:
                del x[pos]
            except:
                pass
        
        if len(x) > self.size:
            x = x[:self.size]
        else:
            x = x + [0] * (self.size - len(x))

        x = torch.Tensor(x)
        return x.unsqueeze(0)

class Permutaion:

    def __init__(self, max_segment) -> None:
        self.max_segment = max_segment

class RRPAug:
    """
    基于不同序列生成流量视图
    """

    def __init__(self, max_loss_rate, max_switch_rate, feature_length):
        self.max_loss_rate = max_loss_rate
        self.max_switch_rate = max_switch_rate
        self.feature_length = feature_length

    def fit_data(self, seq, fl):
        res = []
        for rrp in seq:
            res += [1] * rrp[0] + [-1] * rrp[1]
        if len(res) > fl:
            res = res[:fl]
        else:
            res = res + [0] * (fl - len(res))
        res = torch.Tensor(res)
        return res.unsqueeze(0)

    def reshape(self, traffic_feature):
        feature = traffic_feature.tolist()
        i = 0
        rrg_list = []
        while i < len(feature):
            out_num = 0
            in_num = 0
            while i < len(feature) and feature[i] == 1:
                out_num += 1
                i += 1
            while i < len(feature) and feature[i] == -1:
                in_num += 1
                i += 1
            rrg_list.append((out_num, in_num))
        return rrg_list

    def __call__(self, x):
        x = self.reshape(x)
        res = []
        res.append(x.pop(0))
        loss_rate = random.random() * self.max_loss_rate
        i = 0
        while i < len(x):
            if random.random() < loss_rate:
                x.pop(i)
                continue
            i += 1
        switch_rate = random.random() * self.max_switch_rate
        while len(x) > 0:
            for i in range(len(x)):
                if random.random() > switch_rate:
                    res.append(x.pop(i))
                    break
        # for rrp in res:

        return self.fit_data(res, self.feature_length)