import torch
import numpy as np
import os
import yaml
import sys
import pickle
import configargparse
from tqdm import tqdm

sys.path.append('./')
from grasp_transfer.refine_module import PandaRefine, get_results_from_pth
from tools.utils import cond_mkdir, get_model

p = configargparse.ArgumentParser()
p.add_argument('--category', type=str, default='')
p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
p.add_argument('--steps', type=int, default=10, help='refine optim steps')
p.add_argument('--dm', action="store_true", default=False, help='Refine Direct Mapping')
opt = p.parse_args()
GRASP_DATA_ROOT = 'grasp_data'

def panda_refine(meta_params, pkl_path, opt, params):
    steps=opt.steps
    # define DIF-Net
    model = get_model(meta_params)

    with open(pkl_path, 'rb') as f:
        grasp_info = pickle.load(f)
    print(pkl_path)

    obj_scale = torch.from_numpy(grasp_info['gt_scale'].astype(np.float32)).cuda()

    shape_code = grasp_info['code']
    shape_code = torch.from_numpy(shape_code).cuda()

    model_refine = PandaRefine(grasp_info, obj_scale, shape_code)
    model_refine.cuda()

    # define optimizer
    optimizer = torch.optim.Adam(model_refine.parameters(), lr=opt.lr)
    loss_min = 1e10
    
    total_time = 0.
    with tqdm(total=steps) as pbar:
        for step in range(steps):
            
            optimizer.zero_grad()

            losses = model_refine(model)
            
            loss_refine = losses['loss']
            loss_choose = losses['loss_choose']
            l_anti = losses['l_anti']
            l_touch = losses['l_touch']
            l_collision = losses['l_collision']
            l_reg = losses['l_reg']

            loss_refine.backward()
            optimizer.step()
            pbar.update(1)

            tqdm.write('step: {}, loss: {:.4f} l_anti: {:.4f}, l_touch: {:.4f} l_collision: {:.4f}, l_reg: {:.4f}'.format(
                step, loss_refine, l_anti, l_touch, l_collision, l_reg))

            if loss_choose < loss_min:
                # tqdm.write('step: {},  loss_choose: {}'.format(step, loss_choose))
                loss_min = loss_choose
                torch.save(model_refine.state_dict(), '{}/{}.pth'.format(opt.save_pth_dir, params['filename']))

    deltas = torch.load('{}/{}.pth'.format(opt.save_pth_dir, params['filename']))
    grasp_info = get_results_from_pth(grasp_info, deltas)
    with open('{}/{}.pkl'.format(opt.save_dir, params['filename']), 'wb') as f:
        pickle.dump(grasp_info, f)
    
    return total_time

def findAllFilesWithSpecifiedSuffix(target_dir, target_suffix="txt"):
    find_res = []
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            if file.endswith(target_suffix):
                find_res.append(os.path.join(root_path, file))
    return find_res

if __name__ == '__main__':
    mode = 'eval'
    opt.dif_config = 'DIF_decoder/configs/generate/{0}.yml'.format(opt.category)

    with open(os.path.join(opt.dif_config),'r') as stream: 
        meta_params = yaml.safe_load(stream)

    exp_name = meta_params['experiment_name']
    opt.pkl_root = os.path.join(GRASP_DATA_ROOT, exp_name)
    
    if opt.dm:
        obj_list = findAllFilesWithSpecifiedSuffix(opt.pkl_root, '_dm.pkl')
    else:
        obj_list = findAllFilesWithSpecifiedSuffix(opt.pkl_root, '_tf.pkl')

    save_root = opt.pkl_root + '_refine'
    cond_mkdir(os.path.join(save_root, mode))
    cond_mkdir(os.path.join(save_root, '{}_pth'.format(mode)))

    for i, file_path in enumerate(obj_list):
        _, _, _, inst_name, part, filename = file_path.split('/')

        tqdm.write('{} / {} : {}/{}'.format(i + 1, len(obj_list), inst_name, part))
        opt.save_dir = os.path.join(save_root, mode, inst_name, part)
        cond_mkdir(opt.save_dir)
        opt.save_pth_dir = os.path.join(save_root, '{}_pth'.format(mode), inst_name, part)
        cond_mkdir(opt.save_pth_dir)
        
        pms = {'mode':mode, 'filename':filename.split('.')[0] + '_refine'} 
        panda_refine(meta_params, file_path, opt, pms)

    print('Finish')