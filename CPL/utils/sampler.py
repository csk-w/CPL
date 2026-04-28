# --coding:utf-8--
from __future__ import absolute_import

import time
from collections import defaultdict
from torch.utils.data.sampler import Sampler

import numpy as np
import random
import torch


def no_index(A, b):
    return np.nonzero(A != b)[0]


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.pid_cam = dict()
        self.pid_index = dict()

        pid_cam = defaultdict(list)
        pid_index = defaultdict(list)

        # === START MODIFICATION ===
        for index, item in enumerate(data_source):
            pid, cam = item[1], item[2]
            # === END MODIFICATION ===

            if isinstance(pid, list):
                pid = pid[0]
            if pid < 0:
                continue
            pid_cam[pid].append(cam)
            pid_index[pid].append(index)

        self.pids = list(pid_index.keys())
        for pid in self.pids:
            self.pid_cam[pid] = np.array(pid_cam[pid])
            self.pid_index[pid] = np.array(pid_index[pid])
        self.num_pids = len(self.pids)

    def __len__(self):
        return self.num_pids * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_pids).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            # === START MODIFICATION ===
            item = self.data_source[i]
            i_pid, i_cam = item[1], item[2]
            # === END MODIFICATION ===

            if isinstance(i_pid, list):
                i_pid = i_pid[0]

            cams = self.pid_cam[i_pid]
            index = self.pid_index[i_pid]

            select_cams = no_index(cams, i_cam)
            num_choices = self.num_instances - 1

            if select_cams.size:
                if select_cams.size >= num_choices:
                    cam_indexes = np.random.choice(select_cams, size=num_choices, replace=False)
                else:
                    indexes0 = select_cams[np.random.permutation(select_cams.size)]
                    num_choices -= select_cams.size
                    indexes1 = np.random.choice(select_cams, size=num_choices,
                                                replace=select_cams.size < num_choices)
                    cam_indexes = np.append(indexes0, indexes1)

                ret += [i] + index[cam_indexes].tolist()
            else:
                select_indexes = no_index(index, i)
                if not select_indexes.size:
                    continue
                if select_indexes.size >= num_choices:
                    ind_indexes = np.random.choice(select_indexes, size=num_choices, replace=False)
                else:
                    indexes0 = select_indexes[np.random.permutation(select_indexes.size)]
                    num_choices -= select_indexes.size
                    indexes1 = np.random.choice(select_indexes, size=num_choices,
                                                replace=select_indexes.size < num_choices)
                    ind_indexes = np.append(indexes0, indexes1)

                ret += [i] + index[ind_indexes].tolist()

        return iter(ret)


class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.pid_index = dict()
        pid_index = defaultdict(list)

        # === START MODIFICATION ===
        for index, item in enumerate(data_source):
            pid = item[1]
            # === END MODIFICATION ===
            if pid < 0:
                continue
            pid_index[pid].append(index)

        self.pids = list(pid_index.keys())
        for pid in self.pids:
            self.pid_index[pid] = np.array(pid_index[pid])
        self.num_pids = len(self.pids)

    def __len__(self):
        return self.num_pids * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_pids).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            # === START MODIFICATION ===
            item = self.data_source[i]
            i_pid = item[1]
            # === END MODIFICATION ===

            ret.append(i)
            index = self.pid_index[i_pid]

            select_indexes = no_index(index, i)
            num_choices = self.num_instances - 1
            if not select_indexes.size:
                ret.pop()
                continue
            elif select_indexes.size >= num_choices:
                ind_indexes = np.random.choice(select_indexes, size=num_choices, replace=False)
            else:
                indexes0 = select_indexes[np.random.permutation(select_indexes.size)]
                num_choices -= select_indexes.size
                indexes1 = np.random.choice(select_indexes, size=num_choices,
                                            replace=select_indexes.size < num_choices)
                ind_indexes = np.append(indexes0, indexes1)

            ret += index[ind_indexes].tolist()

        return iter(ret)


if __name__ == '__main__':
    from data import IterLoader
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset

    class D(Dataset):
        def __init__(self):
            self.a = []
            for _ in range(20000):
                pid = np.random.randint(0, 700)
                cam = np.random.randint(0, 8)
                self.a.append([[], pid, cam])

        def __getitem__(self, index):
            return self.a[index]

        def __len__(self):
            return len(self.a)

    d = []
    for _ in range(50000):
        pid = np.random.randint(0, 700)
        cam = np.random.randint(0, 8)
        d.append([[], pid, cam])

    data_loader = IterLoader(
        DataLoader(dataset=d,
                   batch_size=256,
                   sampler=RandomMultipleGallerySampler(d, 16),
                   num_workers=0,
                   drop_last=True,
                   pin_memory=True),
        length=300
    )

    t = time.time()
    data_loader.new_epoch()
    data_loader.next()
    print(time.time()-t)
    pass