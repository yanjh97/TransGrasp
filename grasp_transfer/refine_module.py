import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('./')
from tools.utils import get_points_sdf_normal, get_template_points_sdf_normal

##########################################################
##################Some weights for loss function##########
##########################################################
w_all = {'l_anti':100,
        'l_touch':20,
        'l_collision':10,
        'l_reg':2
        }
w_all_choose_best = {'l_anti':100,
                    'l_touch':20,
                    'l_reg':0
                    }
w_reg = [100,  # w_reg_1
        100,  # w_reg_2
        100,  # w_reg_d
        100   # w_reg_v
        ]
##########################################################
##########################################################

def get_RTs_from_grasp_params_tensor(obj_left_points, obj_right_points, \
                                    vectors, depths, obj_scale):

    hori = obj_right_points - obj_left_points
    assert (torch.norm(hori, dim=1) > 0).all(), "Warning! Only one point!!"
    hori = hori / torch.norm(hori, dim=1, keepdim=True)
    normal = torch.cross(hori, vectors, dim=1)
    normal = normal / torch.norm(normal, dim=1, keepdim=True)
    
    appro_correct = torch.cross(normal, hori, dim=1)

    RTs = torch.eye(4, 4).repeat(vectors.shape[0], 1, 1).cuda()
    RTs[:, :3, :3] = torch.stack((normal, hori, appro_correct), dim=2)
    RTs[:, :3, 3] = (obj_left_points + obj_right_points) *  (obj_scale / 2) / 2
    RTs[:, :3, 3] -= appro_correct*depths.reshape(-1, 1)

    return RTs.float().cuda()

def get_width_from_grasp_params_tensor(obj_left_points, obj_right_points, obj_scale):

    left_points = obj_left_points * (obj_scale / 2)
    right_points = obj_right_points * (obj_scale / 2)
    hori = right_points - left_points
    width = torch.norm(hori, dim=1, keepdim=False)

    return width

def get_mesh_sampled_points(gripper_name, num_points):
    pts = np.load('grasp_transfer/assets/{}.npz'.format(gripper_name))
    sample_index_1 = np.random.choice(pts['hand'].shape[0], num_points // 5, replace=False)
    sample_index_2 = np.random.choice(pts['left_finger'].shape[0], num_points // 5 * 2, replace=False)
    sample_index_3 = np.random.choice(pts['right_finger'].shape[0], num_points // 5 * 2, replace=False)
    # return np.concatenate([pts['hand'][sample_index_1], pts['left_finger'][sample_index_2], pts['right_finger'][sample_index_3]], axis=0)
    return [pts['hand'][sample_index_1], pts['left_finger'][sample_index_2], pts['right_finger'][sample_index_3]]

class PandaRefine(nn.Module):
    def __init__(self, grasp_info, obj_scale, shape_code=None):
        super(PandaRefine, self).__init__()

        self.grasp_info = grasp_info
        self.grasp_params = grasp_info['grasp_params']
        self.obj_scale = obj_scale
        self.shape_code = shape_code

        if isinstance(self.grasp_params, list):
            self.num_grasps = len(self.grasp_params)
        elif isinstance(self.grasp_params, dict):
            self.num_grasps = self.grasp_params['left_points'].size()[0]
        elif isinstance(self.grasp_params, np.ndarray):
            self.num_grasps = self.grasp_params.shape[0]
        else:
            raise TypeError('Unkown type for grasp_params')

        self.emb = nn.Embedding(self.num_grasps, 10)

        self.best_anti = 0
        self.best_touch = 0
        self.best_collision = 0

        nn.init.zeros_(self.emb.weight)

    def forward(self, model):
        losses = self.get_refine_loss(model)
        return losses

    def get_refine_loss(self, model):
        
        deltas = self.emb.weight
        obj_scale = self.obj_scale
        shape_code = self.shape_code

        delta_p1 = deltas[:, :3]
        delta_p2 = deltas[:, 3:6]
        delta_depth = deltas[:, 6]
        delta_vector = deltas[:, 7:]
  
        if isinstance(self.grasp_params, list) or isinstance(self.grasp_params, np.ndarray):
            obj_left_points = []
            obj_right_points = []
            depths = []
            vectors = []            
            # Transfer grasp_params to tensor:
            for grasp_param in self.grasp_params:
                obj_left_points.append(grasp_param['left_points'])
                obj_right_points.append(grasp_param['right_points'])
                depths.append(grasp_param['depth'])
                vectors.append(grasp_param['approach_vector'])
            obj_left_points = np.array(obj_left_points, dtype=np.float32)
            obj_right_points = np.array(obj_right_points, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
            vectors = np.array(vectors, dtype=np.float32)
            obj_left_points = torch.from_numpy(obj_left_points).cuda()
            obj_right_points = torch.from_numpy(obj_right_points).cuda()
            vectors = torch.from_numpy(vectors).cuda()
            depths = torch.from_numpy(depths).cuda()
            # Add delta:
            obj_left_points = obj_left_points + delta_p1
            obj_right_points = obj_right_points + delta_p2
            vectors = vectors + delta_vector
            depths = depths + delta_depth     
        elif isinstance(self.grasp_params, dict):            
            obj_left_points = self.grasp_params['left_points'] + delta_p1
            obj_right_points = self.grasp_params['right_points'] + delta_p2
            vectors = self.grasp_params['approach_vector'] + delta_vector
            depths = self.grasp_params['depth'] + delta_depth


        # Get RT from grasp_params:
        RTs = get_RTs_from_grasp_params_tensor(obj_left_points, obj_right_points, vectors, depths, obj_scale)

        ##########################################################
        ##########################l_anti##########################
        ##########################################################
        # Antipodal rules:

        if shape_code is None:
            lp_sdfs, lp_normals = get_template_points_sdf_normal(model, obj_left_points)
            rp_sdfs, rp_normals = get_template_points_sdf_normal(model, obj_right_points)
        else:
            lp_sdfs, lp_normals = get_points_sdf_normal(model, shape_code, obj_left_points)
            rp_sdfs, rp_normals = get_points_sdf_normal(model, shape_code, obj_right_points)
        lp_normals = lp_normals / torch.norm(lp_normals, dim=1, keepdim=True)
        rp_normals = rp_normals / torch.norm(rp_normals, dim=1, keepdim=True)
        vec_r2l = obj_left_points - obj_right_points
        vec_r2l = vec_r2l / torch.norm(vec_r2l, dim=1, keepdim=True)
        vec_l2r = obj_right_points - obj_left_points
        vec_l2r = vec_l2r / torch.norm(vec_l2r, dim=1, keepdim=True)

        l_anti = -torch.sum(lp_normals * vec_r2l, dim=1) - torch.sum(rp_normals * vec_l2r, dim=1)
        l_anti += 2.

        ##########################################################
        ##########################l_touch#########################
        ##########################################################
        l_touch = torch.norm(lp_sdfs, dim=1) + torch.norm(rp_sdfs, dim=1)

        ##########################################################
        ##########################l_collision###########################
        ##########################################################
        # sdfs of Gripper's points
        num_grasp_points = 150
        width = get_width_from_grasp_params_tensor(obj_left_points.detach(), obj_right_points.detach(), obj_scale)
        # get gripper points (np.numpy)
        gripper_points_list = get_mesh_sampled_points('fat_hand_part',num_grasp_points)
        
        hand_pts_num = gripper_points_list[0].shape[0]
        lf_pts_num = gripper_points_list[1].shape[0]
        rf_pts_num = gripper_points_list[2].shape[0]
        # hand points
        hand_pts = torch.from_numpy(gripper_points_list[0]).float().cuda()
        hand_pts = hand_pts.view(1, hand_pts_num, 3).repeat(self.num_grasps, 1, 1)
        # left gripper points
        lf_pts = torch.from_numpy(gripper_points_list[1]).float().cuda()
        lf_pts = lf_pts.view(1, lf_pts_num, 3).repeat(self.num_grasps, 1, 1)
        width_lf = width.view(self.num_grasps, 1, 1).repeat(1, lf_pts_num, 1)
        lf_pts[:,:,1:2] += -width_lf
        # right gripper points
        rf_pts = torch.from_numpy(gripper_points_list[2]).float().cuda()
        rf_pts = rf_pts.view(1, rf_pts_num, 3).repeat(self.num_grasps, 1, 1)
        width_rf = width.view(self.num_grasps, 1, 1).repeat(1, rf_pts_num, 1)
        rf_pts[:,:,1:2] += width_rf

        gripper_points = torch.cat((hand_pts, lf_pts, rf_pts), axis=1)
        gripper_points = gripper_points.detach()

        assert gripper_points.size()[0] == self.num_grasps
        assert gripper_points.size()[1] == (hand_pts_num+lf_pts_num+rf_pts_num)
        
        gripper_points_in_objcoords = torch.bmm(gripper_points, RTs[:,:3,:3].transpose(1,2)) + RTs[:,:3,3].unsqueeze(1)
        gripper_points_in_objcoords = gripper_points_in_objcoords.view(-1,3) / (obj_scale /2.)
        
        if shape_code is None:
            gripper_sdfs, _ = get_template_points_sdf_normal(model, gripper_points_in_objcoords)
        else:
            gripper_sdfs, _ = get_points_sdf_normal(model, shape_code, gripper_points_in_objcoords)
        gripper_sdfs = -torch.where(gripper_sdfs>0, torch.zeros_like(gripper_sdfs), gripper_sdfs)
        l_collision = torch.sum(gripper_sdfs.view(self.num_grasps, num_grasp_points), dim=1)

        ##########################################################
        ##########################l_reg###########################
        ##########################################################
        l_reg_v = torch.norm(delta_vector, dim=1)

        l_reg = w_reg[0]*torch.norm(delta_p1, dim=1) \
            + w_reg[1]*torch.norm(delta_p2, dim=1) \
            + w_reg[2]*torch.abs(delta_depth) \
            + w_reg[3]*l_reg_v

        ##########################################################
        #######################loss_total#########################
        ##########################################################
        loss_total = w_all['l_anti']*l_anti + w_all['l_touch']*l_touch + \
                     w_all['l_collision']*l_collision + w_all['l_reg']*l_reg

        ##########################################################

        no_collision_index = torch.where(l_collision == 0)[0] # torch.where's output is tuple(numpy)
        loss_choose = w_all_choose_best['l_anti']*l_anti[no_collision_index] \
                        + w_all_choose_best['l_touch']*l_touch[no_collision_index] \
                        + w_all_choose_best['l_reg']*l_reg[no_collision_index]

        if no_collision_index.shape[0] != 0:
            _idx = torch.argmin(loss_choose)
            loss_choose = loss_choose[_idx]
            best_idx = no_collision_index[_idx]
        else:
            _idx = torch.argmin(loss_total)
            loss_choose = loss_total[_idx]
            best_idx = _idx

        loss_total = loss_total.mean()
        l_anti = l_anti.mean()
        l_touch = l_touch.mean()
        l_collision = l_collision.mean()
        l_reg = l_reg.mean()

        return {'loss':loss_total,
                'l_touch':l_touch,
                'l_anti':l_anti,
                'l_collision':l_collision,
                'l_reg':l_reg,
                'loss_choose':loss_choose,
                'best_idx': best_idx,
                }

def get_results_from_pth(grasp_info, deltas):
    
    deltas = deltas['emb.weight'].data.cpu().numpy()
    delta_p1 = deltas[:, :3]
    delta_p2 = deltas[:, 3:6]
    delta_depth = deltas[:, 6]
    delta_vector = deltas[:, 7:]

    if type(grasp_info['grasp_params']) is dict:
        grasp_param = grasp_info['grasp_params']
        grasp_param['left_points'] += delta_p1[0]
        grasp_param['right_points'] += delta_p2[0]
        grasp_param['depth'] += delta_depth[0]
        grasp_param['approach_vector'] += delta_vector[0]
    else:
        for i, grasp_param in enumerate(grasp_info['grasp_params']):
            grasp_param['left_points'] += delta_p1[i]
            grasp_param['right_points'] += delta_p2[i]
            grasp_param['depth'] += delta_depth[i]
            grasp_param['approach_vector'] += delta_vector[i]
    return grasp_info

def get_best_results_from_pth(grasp_params, deltas, best_idx):
    deltas = deltas['emb.weight']
    delta = deltas[best_idx]
    delta_p1 = delta[:3]
    delta_p2 = delta[3:6]
    delta_depth = delta[6]
    delta_vector = delta[7:]

    if type(grasp_params) is dict:
        grasp_params['left_points'] = grasp_params['left_points'][best_idx] + delta_p1
        grasp_params['right_points'] =  grasp_params['right_points'][best_idx] + delta_p2
        grasp_params['depth'] = grasp_params['depth'][best_idx] + delta_depth
        grasp_params['approach_vector'] = grasp_params['approach_vector'][best_idx] + delta_vector
    return grasp_params