import torch
import os
import yaml
import sys
import pickle
import configargparse
from tqdm import tqdm

sys.path.append('./')
from tools.utils import cond_mkdir, get_model
from grasp_transfer.refine_module import PandaRefine, get_results_from_pth

p = configargparse.ArgumentParser()
p.add_argument('--category', type=str, default='')
p.add_argument('--lr', type=float, default=1.25e-2, help='learning rate')
p.add_argument('--steps', type=int, default=100, help='refine optim steps')
opt = p.parse_args()

GRASP_DATA_ROOT = 'grasp_data/'


def refine_template(meta_params, params, opt):
    steps=opt.steps
    obj_scale = params['average_scale']

    with open('{}/{}/{}/{}.pkl'.format(GRASP_DATA_ROOT, meta_params['experiment_name'], params['mode'], params['filename']), 'rb') as f:
        grasp_info = pickle.load(f)

    # DIF-Net Decoder
    model = get_model(meta_params)
    # Refine Module
    shape_code = None
    model_refine = PandaRefine(grasp_info, obj_scale, shape_code)
    model_refine.cuda()

    # define optimizer
    optimizer = torch.optim.Adam(model_refine.parameters(), lr=opt.lr)
    loss_min = 1e10
    
    cond_mkdir(os.path.join(opt.save_dir, params['inst_name']))
    cond_mkdir(os.path.join(opt.save_pth_dir, params['inst_name']))
    
    with tqdm(total=steps) as pbar:
        for step in range(steps):
            losses = model_refine(model)

            optimizer.zero_grad()
            loss_refine = losses['loss']
            l_anti = losses['l_anti']
            l_touch = losses['l_touch']
            l_collision = losses['l_collision']
            l_reg = losses['l_reg']

            loss_refine.backward()
            optimizer.step()
            pbar.update(1)

            tqdm.write('Name:{}, step: {}, loss: {}'.format(params['filename'], step, loss_refine))
            tqdm.write('l_anti: {}, l_touch: {}'.format(l_anti, l_touch))
            tqdm.write('l_collision: {}, l_reg: {}\n'.format(l_collision, l_reg))
            
            # Save the best model
            if loss_refine < loss_min:
                loss_min = loss_refine

                torch.save(model_refine.state_dict(), os.path.join(opt.save_pth_dir, params['filename']))
    deltas = torch.load(os.path.join(opt.save_pth_dir, params['filename']))
    grasp_info = get_results_from_pth(grasp_info, deltas)
    
    with open('{}.pkl'.format(os.path.join(opt.save_dir, params['filename'])), 'wb') as f:
        pickle.dump(grasp_info, f)
    
if __name__ == '__main__':
    mode='train'
    opt.dif_config = 'DIF_decoder/configs/generate/{0}.yml'.format(opt.category)
    average_scale_dict = {
        'mug': 0.195,
        'bowl':0.2115,
        'bottle':0.189,
    }
    
    with open(os.path.join(opt.dif_config),'r') as stream: 
        meta_params = yaml.safe_load(stream)
    
    opt.save_dir = os.path.join(GRASP_DATA_ROOT, '{}_refine'.format(meta_params['experiment_name']), mode)
    cond_mkdir(opt.save_dir)
    opt.save_pth_dir = os.path.join(GRASP_DATA_ROOT, '{}_refine'.format(meta_params['experiment_name']), '{}_pth'.format(mode))
    cond_mkdir(opt.save_pth_dir)

    pms = dict()
    pms['average_scale'] = average_scale_dict[opt.category]
    file = 'template/template'
    pms['inst_name']=file.split('/')[0]
    pms['part']=file.split('/')[1]
    pms['filename']=file
    pms['mode']=mode

    refine_template(meta_params, pms, opt)
    print('Finish')