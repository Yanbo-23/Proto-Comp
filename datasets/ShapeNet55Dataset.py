import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.loop = config.loop
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        self.data_num_per_category = 100
        self.used_category = [
            "02691156",
            "02933112",
            "02958343",
            "03001627",
            "03636649",
            "04256520",
            "04379243",
            "04530566",
        ]
        self.used_category_dict = {category: 0 for category in self.used_category}
        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split("-")[0]
            if self.subset == "train":
                if taxonomy_id not in self.used_category_dict.keys():
                    continue
                else:
                    self.used_category_dict[taxonomy_id] += 1
                    if (
                        self.used_category_dict[taxonomy_id]
                        > self.data_num_per_category
                    ):
                        continue

            model_id = line.split("-")[1].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
            )
        print(f"[DATASET] {len(self.file_list)} instances were loaded")

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) * 2
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx % len(self.file_list)]

        data = IO.get(os.path.join(self.pc_path, sample["file_path"])).astype(
            np.float32
        )
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample["taxonomy_id"], sample["model_id"], data

    def __len__(self):
        if self.subset == "train":
            return len(self.file_list) * self.loop
        else:
            return len(self.file_list)

