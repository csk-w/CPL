# main1.py (MODIFIED for GA-GPM Framework)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import os.path as osp
import time
import random
import collections
import copy
import argparse
import json

from CPL.hcf_gnn import HCFGNNClusterer

from torch.backends import cudnn
from sklearn.cluster import DBSCAN
from CPL.models.teacher_student import TeacherStudentModel
from CPL.ga_gpm import GAGPM
from CPL.ga_trainer import GATrainer

from CPL.trainer import Trainer

from CPL.evaluator import exact_features, Evaluator
from CPL.utils.dataloader import get_train_loader, get_test_loader
from CPL.utils.datasets import create
from CPL.configs import config
from CPL.utils.memory_table import MemoryTable
from CPL.HDC import HDC_final
from CPL.utils.lr_scheduler import WarmupMultiStepLR
from CPL.utils.logger import Logger
from CPL.utils.faiss_rerank import compute_jaccard_distance


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    return centers


def main_work(args):
    print('\033[1;31;10m' + args.notes + '\033[0m')
    cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = create(args.dataset, args.root_dir)

    original_train_data = []
    for path, pid, cid in dataset.train:
        original_train_data.append((path, pid, cid, 0, 1.0, path))

    generated_train_data = []
    if args.generated_data_dir and args.generated_data_json_map:
        print(f'\033[1;32mLoading generated data from: {args.generated_data_dir}\033[0m')
        #original_path_map = {osp.basename(item[0]): item for item in original_train_data}
        original_path_map = {osp.join(*item[0].split(os.sep)[-2:]): item for item in original_train_data}

        with open(args.generated_data_json_map, 'r') as f:
            mapping_data = json.load(f)

        max_cid = max([item[2] for item in original_train_data])

        for gen_fname, orig_fname in mapping_data.items():
            if orig_fname in original_path_map:
                source_item = original_path_map[orig_fname]
                pid = source_item[1]
                virtual_cid = max_cid + 1
                img_path = osp.join(args.generated_data_dir, gen_fname)

                if osp.exists(img_path):
                    generated_train_data.append((img_path, pid, virtual_cid, 1, 0.9, source_item[0]))

    num_to_add = args.num_generated_images
    if num_to_add < 0 or num_to_add > len(generated_train_data):
        final_generated_list = generated_train_data
    else:
        final_generated_list = random.sample(generated_train_data, num_to_add)

    dataset.train = original_train_data + final_generated_list
    random.shuffle(dataset.train)

    print(f'Original data: {len(original_train_data)}, Generated data added: {len(final_generated_list)}')
    print(f'Total training samples: {len(dataset.train)}')

    full_dataset_is_gen = np.array([item[3] for item in dataset.train])


    cluster_loader = get_test_loader(args, dataset.train)
    test_loader = get_test_loader(args, dataset.gallery)
    query_loader = get_test_loader(args, dataset.query)

    model = TeacherStudentModel(
        args.arch,
        ema_alpha=args.ema_alpha,
        pretrained=args.resnet_pretrained,
        norm=True,
        pooling_type=args.pooling_type,
        num_parts=args.num_parts
    )
    model.to(device)
    model = nn.DataParallel(model)


    # -----------------------------------------------------------------
    all_camera_ids = [item[2] for item in dataset.train]
    num_total_cameras = max(all_camera_ids) + 1
    print(f"HCF-GNN & GAGPM: Total camera IDs detected (including virtual): {num_total_cameras}")
    # -----------------------------------------------------------------

    gagpm = GAGPM(
        num_features=model.module.student.num_features,
        num_clusters=dataset.num_train_pids,
        num_cameras=num_total_cameras,
        momentum=args.cm_momentum
    ).to(device)

    params = [{'params': [value]} for _, value in model.module.student.named_parameters() if value.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.lr_weight_decay)

    scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=0.1, warmup_iters=10)


    evaluator = Evaluator(model)
    logger = Logger(args)

    if args.clusterer == 'hcf_gnn':
        print("Using HCF-GNN for clustering.")
        clusterer = HCFGNNClusterer(args, model.module.student.num_features, num_total_cameras)
    else:
        print("Using DBSCAN for clustering.")
        clusterer = DBSCAN(eps=args.dbscan_eps, min_samples=4, metric='precomputed', n_jobs=-1)



    for i_epoch in range(args.num_epochs):

        features, camera_indexes = exact_features(model, cluster_loader, camera=True) # camera=True is correct

        print('Clustering and updating labels...')
        start = time.time()
        if args.clusterer == 'hcf_gnn':
            pseudo_labels, num_clusters = clusterer.cluster(
                features,
                camera_indexes,
                full_dataset_is_gen,
                dataset.num_train_pids
            )
        else:
            rerank_dist = compute_jaccard_distance(features, k1=args.jaccard_k1, k2=args.jaccard_k2)
            pseudo_labels = clusterer.fit_predict(rerank_dist)
            num_clusters = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        cluster_time = time.time() - start
        print(f'Clustering done in {cluster_time:.2f}s. Found {num_clusters} clusters.')

        all_cams = np.array([item[2] for item in dataset.train])
        gagpm.initialize_prototypes(features, pseudo_labels, all_cams, full_dataset_is_gen)
        del features

        new_dataset = []
        for i, label in enumerate(pseudo_labels):
            if label != -1:
                item = dataset.train[i]
                new_item = (item[0], int(label), item[2], item[3], item[4], item[5])
                new_dataset.append(new_item)
            	


        train_loader = get_train_loader(args, new_dataset, args.iters)
        trainer = GATrainer(model.module, gagpm, optimizer, scheduler, args)


        trainer.train(train_loader=train_loader, title=f'[Epoch {i_epoch + 1}/{args.num_epochs}] Train',i_epoch=i_epoch)

        if (i_epoch + 1) % args.eval_step_epoch == 0 or i_epoch == 0:
            evaluator.evaluate(query_loader, test_loader)
            print(
                f"Epoch {i_epoch + 1} Evaluation | mAP: {evaluator.mAP:.2%} | Rank-1: {evaluator.cmc_scores[0][1]:.2%}")

            is_best = evaluator.mAP > logger.best_mAP['mAP']
            logger.record(i_epoch, trainer, evaluator, num_clusters, -1, cluster_time)
            if is_best:
                torch.save(model.module.student.state_dict(), osp.join(logger.log_dir, 'model_best.pth'))

    logger.finish()


def main():
    parser = config()

    # === START MODIFICATION: Add New Arguments ===
    parser.add_argument('--generated_data_dir', type=str, default=None, help='Path to generated images.')
    parser.add_argument('--generated_data_json_map', type=str, default=None,
                        help='Path to JSON map for generated images.')
    parser.add_argument('--num_generated_images', type=int, default=-1, help='-1 to use all generated images.')

    parser.add_argument('--ema_alpha', type=float, default=0.999, help='EMA alpha for teacher model update.')
    parser.add_argument('--lambda_cons', type=float, default=1.0, help='Weight for consistency loss.')

    parser.add_argument('--proto_temp', type=float, default=0.1, help='Temperature for Proto-NCE loss.')

    parser.add_argument('--clusterer', type=str, default='hcf_gnn', choices=['hcf_gnn', 'dbscan'],
                        help='Which clusterer to use.')
    parser.add_argument('--k0_factor', type=float, default=2.0,
                        help='Over-segmentation factor for HCF-GNN (k0 = num_true_ids * k0_factor).')
    parser.add_argument('--knn_m', type=int, default=10,
                        help='Number of neighbors (M) for building the sparse graph in HCF-GNN.')

    parser.add_argument('--beta', type=float, default=0.2,
                        help='Weight for the instance-level InfoNCE loss.')
    parser.add_argument('--instance_temp', type=float, default=0.07,
                        help='Temperature for instance-level InfoNCE loss.')

    parser.add_argument('--beta_ramp_epochs', type=int, default=20,
                        help='Number of epochs to ramp up beta weight for instance loss.')

    parser.add_argument('--disable-gnn-cam-feat', action='store_true',
                        help='If specified, do not use camera distribution as a feature for GNN nodes.')

    parser.add_argument("--num-parts", type=int, default=6,
                        help='Number of parts to split the feature map into. 1 means using global features.')



    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_work(args)


if __name__ == '__main__':
    main()
