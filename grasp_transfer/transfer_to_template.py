import numpy as np
from tqdm import tqdm
import os, sys, yaml, configargparse, pickle
import torch
sys.path.append('./')
from tools.utils import *


p = configargparse.ArgumentParser()
p.add_argument('--category', type=str, default='')
p.add_argument('--grasp_data_root', type=str, default='grasp_data/')
p.add_argument('--config_root', type=str, default='DIF_decoder/configs/generate/')
opt = p.parse_args()

def transfer_grasp_file_info_to_template(meta_params, params):
    
    assert params['mode']=='train'

    model = get_model(meta_params)
    shape_code = load_code(model, meta_params, params)

    source_path = os.path.join(opt.grasp_data_root, meta_params['experiment_name'], pms['mode'])
    with open(os.path.join(source_path, pms['filename'] + '.pkl'), 'rb') as f:
        panda_file_info = pickle.load(f)
    
    panda_file_info_template = {'filename': 'template', 'grasp_params':[]}

    with tqdm(total=len(panda_file_info['grasp_params'][:1000])) as pbar:
        for grasp_info in panda_file_info['grasp_params'][:1000]:
            points = np.array([grasp_info['left_points'], grasp_info['right_points']], dtype=np.float32)
            points_cuda = torch.from_numpy(points).cuda()
            template_points = model.get_template_coords(points_cuda, shape_code).squeeze().detach().cpu().numpy().astype(np.float32)
            
            panda_file_info_template['grasp_params'].append({
                'left_points':template_points[0],
                'right_points':template_points[1],
                'approach_vector':grasp_info['approach_vector'],
                'depth':grasp_info['depth']
            })
            pbar.update(1)

    save_dir = os.path.join(opt.grasp_data_root, meta_params['experiment_name'], params['mode'], 'template')
    cond_mkdir(save_dir)
    with open(os.path.join(save_dir, 'template.pkl'), 'wb') as f:
        pickle.dump(panda_file_info_template, f)


if __name__ == '__main__':

    mode = 'train'
    opt.config = os.path.join(opt.config_root, opt.category+'.yml')

    with open(os.path.join(opt.config),'r') as stream: 
        meta_params = yaml.safe_load(stream)

    cond_mkdir('grasp_data/{0}/{1}'.format(meta_params['experiment_name'], mode))

    # transfer grasps on SOURCE model to template
    if not os.path.exists('grasp_data_panda/{0}/train/template.pkl'.format(meta_params['experiment_name'])):
        
        grasp_source_info = {
        'mug':{'mode':'train', 'inst_name':'62634df2ad8f19b87d1b7935311a2ed0', 'part':'0',
                'filename':'62634df2ad8f19b87d1b7935311a2ed0/0'},
        'bowl':{'mode':'train', 'inst_name':'8bb057d18e2fcc4779368d1198f406e7', 'part':'0',
                'filename':'8bb057d18e2fcc4779368d1198f406e7/0'},
        'bottle':{'mode':'train', 'inst_name':'3108a736282eec1bc58e834f0b160845', 'part':'0',
                'filename':'3108a736282eec1bc58e834f0b160845/0'},                              
        }
        pms=grasp_source_info[opt.category]
        with open('DIF_decoder/split/train/{}.txt'.format(opt.category), 'r') as f:
            i = 0
            line = f.readline().rstrip('\n')
            while (line):
                if line == pms['filename']:
                    pms['subject_idx'] = i
                    break
                i += 1
                line = f.readline().rstrip('\n')
        
        transfer_grasp_file_info_to_template(meta_params, pms)
