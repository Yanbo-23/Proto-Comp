import numpy as np
import torch


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __call__(self, pc, angle=None):

        trans = pc.size()[2] > pc.size()[1]
        if trans:
            pc = pc.transpose(1, 2)
        col = pc.size()[2]
        if col == 6:
            pc = pc[:, :, :3]
        
        bsize = pc.size()[0]

        for i in range(bsize):
            if angle is None:
                rotation_angle = np.random.randint(1, 5) / 4 * 2 * np.pi
            else:
                rotation_angle = angle[i]

            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
            )
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            
            
            pc[i, :, :] = torch.matmul(pc[i], R)
            
            
            
            
            
        
        zeros_tensor = torch.zeros_like(pc)
        if col == 6:
            pc = torch.cat((pc, zeros_tensor), dim=2)
        if trans:
            pc = pc.transpose(1, 2)
        

        return pc


class PointcloudReflect(object):
    def __call__(self, pc, reflect=None):

        trans = pc.size()[2] > pc.size()[1]
        if trans:
            pc = pc.transpose(1, 2)
        col = pc.size()[2]
        if col == 6:
            pc = pc[:, :, :3]
        
        bsize = pc.size()[0]

        for i in range(bsize):
            if reflect[i] == 1:
                pc[i, :, :] = pc[i, :, :][:, [1, 0, 2]]
            
            
            
            
            
            
            
            
            
            
            

            
            
            
            
            
        
        zeros_tensor = torch.zeros_like(pc)
        if col == 6:
            pc = torch.cat((pc, zeros_tensor), dim=2)
        if trans:
            pc = pc.transpose(1, 2)
        

        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2.0 / 3.0, scale_high=3.0 / 2.0, translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc, xyz1, xyz2):
        
        trans = pc.size()[2] > pc.size()[1]
        if trans:
            pc = pc.transpose(1, 2)
        col = pc.size()[2]
        if col == 6:
            pc = pc[:, :, :3]
        bsize = pc.size()[0]
        for i in range(bsize):
            pc[i, :, 0:3] = (
                torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
                + torch.from_numpy(xyz2).float().cuda()
            )

        zeros_tensor = torch.zeros_like(pc)
        if col == 6:
            pc = torch.cat((pc, zeros_tensor), dim=2)
        if trans:
            pc = pc.transpose(1, 2)
        return pc


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = (
                pc.new(pc.size(1), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
            )
            pc[i, :, 0:3] += jittered_data

        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=2.0 / 3.0, scale_high=3.0 / 2.0):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

            pc[i, :, 0:3] = torch.mul(
                pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()
            )

        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(
                low=-self.translate_range, high=self.translate_range, size=[3]
            )

            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()

        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(
                    len(drop_idx), 1
                )  
                pc[i, :, :] = cur_pc

        return pc
