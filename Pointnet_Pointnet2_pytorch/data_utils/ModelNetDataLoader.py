"""
@original_author: Xu Yan
@original_file: ModelNet.py
@original_time: 2021/3/19 15:51
@modifier: Yanbo Wang
"""

import os
import numpy as np
import warnings
import open3d as o3d
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split="train", process_data=False):

        self.root = root

        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        self.catfile = "data/modelnet40_normal_resampled/modelnet40_shape_names.txt"

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        dirname = os.listdir(self.root)
        dirname = sorted(dirname)
        self.datapath = []
        for cate in dirname:

            newPath = os.path.join(self.root, cate)
            objects = os.listdir(newPath)
            objects = sorted(objects)
            for obj in objects:
                obj_label = cate
                self.datapath.append([obj_label, os.path.join(newPath, obj)])

        print("The size of %s data is %d" % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            point_set = np.array(o3d.io.read_point_cloud(fn[1]).points).astype(
                np.float32
            )

            cls = self.classes[fn[0]]
            label = np.array([cls]).astype(np.int32)

            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        point_set = point_set[:, [0, 2, 1]]

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0], fn

    def __getitem__(self, index):
        return self._get_item(index)
