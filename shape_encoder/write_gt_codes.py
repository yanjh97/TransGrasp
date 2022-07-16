import os, sys, argparse, yaml
import torch
import pickle as pkl
import numpy as np
sys.path.append('./')
sys.path.append('./DIF_decoder')
from DIF_decoder.dif_net import DeformedImplicitField

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default='mug')
parser.add_argument("--config", type=str, default='DIF_decoder/configs/generate/mug.yml')
args = parser.parse_args()

def read_obj_list(obj_list_path):
    obj_list = []
    with open(obj_list_path, 'r') as f:
        obj = f.readline()
        while obj!='':
            obj_list.append(obj.rstrip('\n'))
            obj=f.readline()
    return obj_list

def write_train_pkl(obj_list_path, meta, output_dir, cate):
    obj_list=read_obj_list(obj_list_path)
    model=DeformedImplicitField(**meta)
    model.load_state_dict(torch.load(meta['checkpoint_path']))
    assert len(obj_list) == model.latent_codes.weight.size()[0]
    pkl_info={}
    for i in range(len(obj_list)):
        pkl_info.update({obj_list[i]:model.get_latent_code(torch.tensor(i)).detach().numpy()})
    with open(output_dir + '{0}_train.pkl'.format(cate), 'wb') as f:
        pkl.dump(pkl_info, f)

def write_eval_pkl(obj_list_path, meta, output_dir, cate):
    obj_list = read_obj_list(obj_list_path)
    assert len(obj_list)==len(os.listdir('DIF_decoder/eval/{0}'.format(meta['experiment_name'])))
    pkl_info={}
    for i in range(len(obj_list)):
        shape_code = np.loadtxt(os.path.join('DIF_decoder/eval/{0}'.format(meta['experiment_name']), 
                                              obj_list[i], 
                                             'checkpoints/embedding_epoch_0049.txt'), dtype=np.float32)
        pkl_info.update({obj_list[i]:shape_code})
    with open(output_dir+'{0}_eval.pkl'.format(cate), 'wb') as f:
        pkl.dump(pkl_info, f)

if __name__ == '__main__':

    save_dir='datasets/gt_codes/'
    os.makedirs(save_dir, exist_ok=True)
    with open(args.config,'r') as stream:
        meta_params = yaml.safe_load(stream)
    print('---Write gt codes of training set for {} category---'.format(args.category))
    obj_list_file_path = 'DIF_decoder/split/'+'train'+'/'+'{0}.txt'.format(args.category)
    write_train_pkl(obj_list_file_path, meta_params, save_dir, args.category)
    print('---Write gt codes of evaluation set for {} category---'.format(args.category))
    obj_list_file_path = 'DIF_decoder/split/'+'eval'+'/'+'{0}.txt'.format(args.category)
    write_eval_pkl(obj_list_file_path, meta_params, save_dir, args.category)