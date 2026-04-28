from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
from collections import defaultdict
from ..data import BaseImageDataset



class VeRiWild(BaseImageDataset):
    """
    VeRI-Wild
    Reference:
    Lou et al. VERI-Wild: A Large Dataset and a New Method for Vehicle Re-Identification in the Wild. CVPR 2019.

    Dataset statistics:
    # identities: 30671 (train)
    # images: 277797 (train)
    # test sets: 3000, 5000, 10000 identities
    """
    dataset_dir = 'VERI-Wild'

    def __init__(self, root, verbose=True, test_size=3000, **kwargs):
        super(VeRiWild, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.data_dir = osp.join(self.dataset_dir, 'data')
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')

        if test_size not in [3000, 5000, 10000]:
            raise ValueError("test_size must be one of [3000, 5000, 10000]")

        self.train_list_file = osp.join(self.split_dir, 'train_list.txt')
        self.query_list_file = osp.join(self.split_dir, f'test_{test_size}_query.txt')
        self.gallery_list_file = osp.join(self.split_dir, f'test_{test_size}.txt')
        self.vehicle_info_file = osp.join(self.split_dir, 'vehicle_info.txt')

        self.check_before_run()

        # NEW: Pre-load image path to camera ID mapping
        self._load_vehicle_info()

        train = self.process_dir(self.train_list_file, relabel=True)
        query = self.process_dir(self.query_list_file, relabel=False)
        gallery = self.process_dir(self.gallery_list_file, relabel=False)

        if verbose:
            print(f'=> VeRi-Wild (test_size={test_size}) loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.image_dir):
            raise RuntimeError('"{}" is not available'.format(self.image_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError('"{}" is not available'.format(self.split_dir))
        if not osp.exists(self.train_list_file):
            raise RuntimeError('"{}" is not available'.format(self.train_list_file))
        if not osp.exists(self.query_list_file):
            raise RuntimeError('"{}" is not available'.format(self.query_list_file))
        if not osp.exists(self.gallery_list_file):
            raise RuntimeError('"{}" is not available'.format(self.gallery_list_file))
        if not osp.exists(self.vehicle_info_file):
            raise RuntimeError('"{}" is not available'.format(self.vehicle_info_file))

    def _load_vehicle_info(self):
        """Load image path and camera ID from vehicle_info.txt."""
        self.img_path_to_camid = {}
        with open(self.vehicle_info_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.split(';')
                # if len(parts) == 2:
                img_path, camid = parts[0], parts[1]
                self.img_path_to_camid[img_path] = int(camid)  #

    def process_dir(self, list_file, relabel=False):
        """
        Process a list file to create the dataset.
        Vehicle ID (pid) is parsed from the directory name.
        Camera ID (camid) is retrieved from the pre-loaded vehicle_info mapping.
        """
        with open(list_file, 'r') as f:
            img_paths_relative = f.read().splitlines()

        pid_container = set()
        for img_path in img_paths_relative:
            pid = int(img_path.split('/')[0])
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths_relative:
            pid = int(img_path.split('/')[0])

            # NEW: Get camera ID from the pre-loaded mapping
            camid = self.img_path_to_camid.get(img_path)
            if camid is None:
                print(f"Warning: Camera ID for {img_path} not found in vehicle_info.txt. Skipping this image.")
                continue

            full_img_path = osp.join(self.image_dir, img_path) + '.jpg'

            if relabel:
                pid = pid2label[pid]

            dataset.append((full_img_path, pid, camid))

        return dataset
