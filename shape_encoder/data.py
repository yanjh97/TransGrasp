import re
import torch.utils.data as data
import numpy as np
import os, random
import pickle
import sys
sys.path.append('./')
from tools.utils import rot_X, rot_Y, rot_Z

class PartPointsDatset(data.Dataset):
    def __init__(self, data_root, category, mode, n_pts):

        self.category = category
        self.mode = mode
        self.n_pts = n_pts

        assert mode in ['train', 'eval', 'val']

        _mode = 'eval' if mode == 'val' else mode
        self.data_root = os.path.join(data_root, 'render_pc', category, _mode)
        independent_inst_names = os.listdir(self.data_root)

        self.pc_list_path = []
        eval_sample_list = ['000']
        for inst in independent_inst_names:
            for part in os.listdir(os.path.join(self.data_root, inst)):
                if self.mode == 'eval':
                    for number in eval_sample_list:
                        file = 'PC_obj' + '_' + number + '.npy'
                        self.pc_list_path.append(os.path.join(inst, part, file))
                else:   # train or val
                    for file in os.listdir(os.path.join(self.data_root, inst, part)):
                        if file.find('PC_obj') != -1:
                            self.pc_list_path.append(os.path.join(inst, part, file))

        if self.mode in ['train', 'val']:
            with open(os.path.join('datasets/gt_codes', '{0}_{1}.pkl'.format(category, _mode)), 'rb') as f:
                self.gt_codes = pickle.load(f)
        if self.mode == 'val':
            self.pc_list_path = random.sample(self.pc_list_path, len(self.pc_list_path) // 4)

    def __len__(self) -> int:
        return len(self.pc_list_path)

    def __getitem__(self, index: int):
        
        _path = os.path.join(self.data_root, self.pc_list_path[index])
        pc_obj = np.load(_path)

        if len(pc_obj) >= self.n_pts:
            sample_index = np.random.choice(pc_obj.shape[0], self.n_pts, replace=False)
        else:
            sample_index = np.random.choice(pc_obj.shape[0], self.n_pts, replace=True)
        pc_obj = pc_obj[sample_index, :]

        if self.mode == 'train':
            add_t = np.clip(0.001 * np.random.randn(pc_obj.shape[0], 3), -0.005, 0.005)
            pc_obj = np.add(pc_obj, add_t)

            alpha = np.random.uniform(-np.pi/24, np.pi/24, 1)
            beta = np.clip(np.random.normal(0, np.pi/6, 1), -np.pi/8, np.pi/8)
            gamma = np.random.uniform(-np.pi/24, np.pi/24, 1)
            pc_obj = pc_obj @ (rot_Z(gamma) @ rot_Y(beta) @ rot_X(alpha)).T

            add_t = np.clip(0.01 * np.random.randn(1, 3), -0.05, 0.05)
            pc_obj = np.add(pc_obj, add_t)

        independent_inst_name, part, pc_filename = self.pc_list_path[index].split('/')
        number = re.search("\d{3}", pc_filename)

        if self.mode in ['train', 'val']:
            gt_code = self.gt_codes[os.path.join(independent_inst_name, part)]
        
        if self.mode == 'eval':
            return pc_obj.astype(np.float32), \
                os.path.join(independent_inst_name, part), \
                number.group()
        else:            
            return pc_obj.astype(np.float32), \
                gt_code.astype(np.float32), \
                os.path.join(independent_inst_name, part), \
                number.group()

if __name__ == '__main__':
    import trimesh

    dataset = PartPointsDatset('datasets/', 'mug', 'eval', 1024)
    dataloader = data.DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True)

    print(len(dataset))

    for i, (obj_pcs, gt_codes) in enumerate(dataloader):
        print(obj_pcs.shape, gt_codes.shape)
        for i in range(obj_pcs.shape[0]):
            pc = trimesh.PointCloud(obj_pcs[i])
            trimesh.Scene([pc]).show()
        break