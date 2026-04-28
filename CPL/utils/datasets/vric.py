# vric.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from ..data import BaseImageDataset

class VRIC(BaseImageDataset):
    """
    VRIC (Vehicle Re-Identification in Context)
    Reference:
    Kanaci, A., Zhu, X., Gong, S.: Vehicle Re-Identificaition in Context. GCPR (2018)

    Dataset statistics:
    # identities: 2811 (train) + 2811 (test)
    # images: 54808 (train) + 2811 (query) + 2811 (gallery)
    """
    dataset_dir = 'VRIC'

    def __init__(self, root, verbose=True, **kwargs):
        super(VRIC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train_images')
        self.query_dir = osp.join(self.dataset_dir, 'probe_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_images')

        self.train_list_file = osp.join(self.dataset_dir, 'vric_train.txt')
        self.query_list_file = osp.join(self.dataset_dir, 'vric_probe.txt')
        self.gallery_list_file = osp.join(self.dataset_dir, 'vric_gallery.txt')

        self.check_before_run()

        train = self._process_txt_file(self.train_list_file, self.train_dir, relabel=True)
        query = self._process_txt_file(self.query_list_file, self.query_dir, relabel=False)
        gallery = self._process_txt_file(self.gallery_list_file, self.gallery_dir, relabel=False)

        if verbose:
            print("=> VRIC dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """检查数据集路径是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError('"{}" is not available'.format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.gallery_dir))
        if not osp.exists(self.train_list_file):
            raise RuntimeError('"{}" is not available'.format(self.train_list_file))
        if not osp.exists(self.query_list_file):
            raise RuntimeError('"{}" is not available'.format(self.query_list_file))
        if not osp.exists(self.gallery_list_file):
            raise RuntimeError('"{}" is not available'.format(self.gallery_list_file))

    def _process_txt_file(self, file_path, img_dir, relabel=False):
        """
        根据txt标注文件处理数据
        txt文件格式: [Image_name] [ID label] [Cam Label]
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        dataset = []
        pid_container = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            fname, pid, camid = line.split()
            pid, camid = int(pid), int(camid)

            if relabel:
                pid_container.add(pid)

            img_path = osp.join(img_dir, fname)
            dataset.append((img_path, pid, camid))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            relabeled_dataset = []
            for img_path, pid, camid in dataset:
                relabeled_dataset.append((img_path, pid2label[pid], camid))
            return relabeled_dataset

        return dataset