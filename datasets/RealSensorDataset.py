import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
import open3d as o3d


@DATASETS.register_module()
class RealSensor(data.Dataset):
    def __init__(self, config):
        """PyTorch Dataset Wrapper"""
        test_path = config.dataPath
        dirname = os.listdir(test_path)
        dirname = sorted(dirname)
        self.datapath = []
        for cate in dirname:
            newPath = os.path.join(test_path, cate)
            objects = os.listdir(newPath)
            objects = sorted(objects)
            ccounter = 0
            for obj in objects:
                ccounter += 1
                if ccounter > 50:
                    break
                obj_label = cate
                self.datapath.append([obj_label, os.path.join(newPath, obj)])

    def __len__(self):
        return len(self.datapath)

    def pc_norm(self, partail_scannet):
        """pc: NxC, return NxC"""
        m = np.max(np.sqrt(np.sum(partail_scannet**2, axis=1))) * 2

        partail_scannet = partail_scannet / m
        partail_scannet = torch.from_numpy(partail_scannet)
        return partail_scannet

    def __getitem__(self, index):
        insta = self.datapath[index]
        partail_scannet = np.array(o3d.io.read_point_cloud(insta[1]).points)
        partail_scannet = self.pc_norm(partail_scannet)
        partail_scannet = np.array(partail_scannet, dtype=float)
        return insta[0], partail_scannet, insta[1]
