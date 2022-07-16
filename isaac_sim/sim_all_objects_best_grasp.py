import sys, os, time, math, pickle
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_mul, tf_combine
import torch

sys.path.append('./')
from tools.utils import get_RT_from_grasp_params, get_width_from_grasp_params, R2q

# parse arguments
args = gymutil.parse_arguments(
    description="Test the best grasp",
    custom_parameters=[
        {"name": "--category", "type": str, "default": ''},
        {"name": "--force", "type": float, "default": -0.98},
        {"name": "--asset_root", "type": str, "default": "./isaac_sim/assets"},
        {"name": "--test_mode", "type": str, "default": "pose"},
        {"name": "--pkl_root", "type": str, "default": ''},
        {"name": "--mode", "type": str, "default": 'eval'},
        {"name": "--vis", "action": 'store_true', "default": False},
        ])

resolution = {'mug':300000, 'bottle':8000000, 'bowl':1000000}
force = args.force
asset_root = args.asset_root

def batch_panda_sim(cate, mode):

    # Here, we define a force list
    # where force range from 1/num_envs*force to force:
    force_list = None
    if args.test_mode=="force":
        force_list = torch.linspace(1/args.num_envs*force, force, args.num_envs)

    # set random seed
    np.random.seed(42)

    torch.set_printoptions(precision=4, sci_mode=False)

    # acquire gym interface
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    else:
        raise Exception("This example can only be used with PhysX")

    # set torch device
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'

    # create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise Exception("Failed to create sim")

    #################################################
    # visualize
    #################################################
    if args.vis:
        # create viewer
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 75.0
        camera_props.width = 1920
        camera_props.height = 1080
        camera_props.use_collision_geometry = True
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise Exception("Failed to create viewer")
    #################################################
    
    #################################################
    # load shapenet obj asset START
    #################################################
    obj_asset_list = []
    grasp_pose_list = []
    pred_scale_list = []
    gt_scale_list = []
    RT_offset_list = []
    width_list = []
    for inst_name in os.listdir(os.path.join(args.pkl_root, mode)):
        for part in os.listdir(os.path.join(args.pkl_root, mode, inst_name)):
            for number_file in os.listdir(os.path.join(args.pkl_root, mode, inst_name, part)):
                if number_file.find('_select.pkl') != -1:
                    obj_asset_file = 'urdf/shapenet/{0}/{1}/{2}.urdf'.format(cate, mode, os.path.join(inst_name, part))
                    asset_options = gymapi.AssetOptions()
                    asset_options.vhacd_enabled = True
                    asset_options.vhacd_params.resolution = resolution[args.category]
                    asset_options.vhacd_params.max_convex_hulls = 40
                    asset_options.vhacd_params.max_num_vertices_per_ch = 64
                    # asset_options.static_friction = 10
                    obj_asset = gym.load_asset(sim, asset_root, obj_asset_file, asset_options)
                    obj_asset_list.append(obj_asset)

                    with open(os.path.join(args.pkl_root, mode, inst_name, part, number_file), 'rb') as f:
                        grasp_info = pickle.load(f)
                    
                    # there should be only one grasp configuration in file, which saved in a dict
                    assert isinstance(grasp_info['grasp_params'], dict)
                    grasp_RT = get_RT_from_grasp_params(grasp_info['grasp_params'], grasp_info['pred_scale'])
                    RT_offset_list.append(grasp_info['trans_offset'])
                    width = get_width_from_grasp_params(grasp_info['grasp_params'], grasp_info['pred_scale'])
                    width_list.append(width)
                    grasp_pose_list.append(grasp_RT)
                    pred_scale_list.append(grasp_info['pred_scale'])
                    gt_scale_list.append(grasp_info['gt_scale'])

    num_envs = len(obj_asset_list)

    grasp_R_in_objcoords = torch.zeros((num_envs, 4), dtype=torch.float32)
    grasp_T_in_objcoords = torch.zeros((num_envs, 3), dtype=torch.float32)

    offset_R = torch.zeros((num_envs, 4), dtype=torch.float32)
    offset_T = torch.zeros((num_envs, 3), dtype=torch.float32)

    for i in range(num_envs):

        q = R2q(grasp_pose_list[i][:3,:3])
        t = grasp_pose_list[i][:3,3]
        grasp_R_in_objcoords[i] = torch.from_numpy(q)
        grasp_T_in_objcoords[i] = torch.from_numpy(t)

        q = R2q(RT_offset_list[i][:3,:3])
        t = RT_offset_list[i][:3,3]
        offset_R[i] = torch.from_numpy(q)
        offset_T[i] = torch.from_numpy(t)

    grasp_R_in_objcoords = grasp_R_in_objcoords.cuda()
    grasp_T_in_objcoords = grasp_T_in_objcoords.cuda()
    offset_R = offset_R.cuda()
    offset_T = offset_T.cuda()
    #################################################
    # load shapenet obj asset END
    #################################################

    # load franka asset
    franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
    
    # franka_asset_file = "urdf/marker_description/robots/panda_marker_gripper.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    asset_options.vhacd_enabled = True
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    # configure franka dofs
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_upper_limits = franka_dof_props["upper"]

    # use position drive for all dofs
    franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)

    # grippers
    franka_dof_props["stiffness"].fill(800.0)
    franka_dof_props["damping"].fill(40.0)

    # default dof states and position targets
    franka_num_dofs = gym.get_asset_dof_count(franka_asset) # 2
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
    # grippers open
    default_dof_pos = franka_upper_limits

    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # configure env grid
    num_per_row = int(math.sqrt(num_envs))
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    print("Creating %d environments" % num_envs)

    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(0,0,0.5)

    obj_pose = gymapi.Transform()

    envs = []
    obj_idxs = []
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add obj
        obj_pose.p.x = 0
        obj_pose.p.y = 0
        obj_pose.p.z = 0.5
        obj_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi / 2)
        z_rot_90 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)
        obj_rot_tensor = quat_mul(torch.tensor([z_rot_90.x, z_rot_90.y, z_rot_90.z, z_rot_90.w]),
                                torch.tensor([obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w]))
        obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w = \
            obj_rot_tensor[0],  obj_rot_tensor[1], obj_rot_tensor[2], obj_rot_tensor[3]
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        obj_handle = gym.create_actor(env, obj_asset_list[i], obj_pose, "obj", i, 0)
        gym.set_actor_scale(env, obj_handle, gt_scale_list[i] / 0.05)
        gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        obj_idx = gym.get_actor_rigid_body_index(env, obj_handle, 0, gymapi.DOMAIN_SIM)
        obj_idxs.append(obj_idx)
        
        # add franka
        obj_pos_tensor = torch.tensor([obj_pose.p.x, obj_pose.p.y, obj_pose.p.z])
        ############################################################################################################
        # GT object pose
        # gripper_rot_tensor, gripper_pos_tensor = tf_combine(obj_rot_tensor, obj_pos_tensor, 
        #                                         grasp_R_in_objcoords[i].cpu(), grasp_T_in_objcoords[i].cpu())        
        ############################################################################################################
        
        ############################################################################################################
        # estimated object pose
        est_obj_rot_tensor, est_obj_pos_tensor = tf_combine(obj_rot_tensor, obj_pos_tensor,
                                                offset_R[i].cpu(), offset_T[i].cpu())
        gripper_rot_tensor, gripper_pos_tensor = tf_combine(est_obj_rot_tensor, est_obj_pos_tensor, 
                                                grasp_R_in_objcoords[i].cpu(), grasp_T_in_objcoords[i].cpu())
        ############################################################################################################

        franka_pose.p.x, franka_pose.p.y, franka_pose.p.z = \
                            gripper_pos_tensor[0], gripper_pos_tensor[1], gripper_pos_tensor[2]
        franka_pose.r.x, franka_pose.r.y, franka_pose.r.z, franka_pose.r.w = \
            gripper_rot_tensor[0],  gripper_rot_tensor[1], gripper_rot_tensor[2], gripper_rot_tensor[3]
        franka_handle = gym.create_actor(env, franka_asset, franka_pose, "gripper", i, 1)
        gym.set_actor_scale(env, franka_handle, 1.0)

        # set dof properties
        gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        if args.category == 'bottle':
            default_dof_state["pos"] = franka_upper_limits
        else:
            if width_list[i]*2 > 0.04: # upper limits of Panda Hand
                default_dof_state["pos"] = franka_upper_limits
            else:
                default_dof_state["pos"] = np.array([width_list[i]*2, width_list[i]*2], dtype=np.float32)
        gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
        # set initial position targets
        gym.set_actor_dof_position_targets(env, franka_handle, default_dof_state["pos"])

    if args.vis:
        # point camera at middle env
        cam_pos = gymapi.Vec3(4 / 4, 3 / 4, 2 / 4)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    num_bodies = gym.get_asset_rigid_body_count(obj_asset) + gym.get_asset_rigid_body_count(franka_asset)

    dt_cnt = 0
    # ==== prepare tensors =====
    # from now on, we will use the tensor API that can run on CPU or GPU
    gym.prepare_sim(sim)

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    dt_cnt += 1

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    # get rigid body state tensor
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)
    init_obj_pos = rb_states[obj_idxs, :3].cpu().numpy()

    start_time = time.time()

    sr_list = []
    # simulation loop
    while time.time() - start_time < 1000:

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        dt_cnt+=1

        # refresh tensors
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        if dt_cnt > (10 * 60):
            grip_acts = torch.Tensor([[0.0, 0.0]] * num_envs).to(device).unsqueeze(-1)
        
            # set new position targets
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(grip_acts))

        # if time passed more than threshold seconds, set the force
        if dt_cnt > (13 * 60):
            forces = torch.zeros((num_envs*num_bodies, 3), device=device, dtype=torch.float)
            forces[obj_idxs, 2] = force_list if force_list is not None else force
            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.GLOBAL_SPACE)
        
        # if time passed more than threshold seconds, compute the Success Rate
        if dt_cnt == (18 * 60):
            obj_pos = rb_states[obj_idxs, :3].cpu().numpy()
            success_num = np.sum(np.linalg.norm(obj_pos - init_obj_pos, axis=1) < 0.1)
            sr_list.append(100 * success_num / num_envs)
        # 10s
        elif dt_cnt == (23 * 60):
            obj_pos = rb_states[obj_idxs, :3].cpu().numpy()
            success_num = np.sum(np.linalg.norm(obj_pos - init_obj_pos, axis=1) < 0.1)
            sr_list.append(100 * success_num / num_envs)
        # 15s
        elif dt_cnt == (28 * 60):
            obj_pos = rb_states[obj_idxs, :3].cpu().numpy()
            success_num = np.sum(np.linalg.norm(obj_pos - init_obj_pos, axis=1) < 0.1)
            sr_list.append(100 * success_num / num_envs)
            break
        else:
            pass
        if args.vis:
            # update viewer
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)
    # cleanup
    if args.vis:
        time.sleep(5)
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    grasp_R_in_objcoords = grasp_R_in_objcoords.cpu()
    grasp_T_in_objcoords = grasp_T_in_objcoords.cpu()
    forces = forces.cpu()
    # torques = torques.cpu()

    # print("total use time : {0}".format(time.time() - start_time))
    torch.cuda.empty_cache() 

    return sr_list


if __name__ == '__main__':
    mode = args.mode
    sr = batch_panda_sim(args.category, mode)
    print('Success rate: {:.2f}%'.format(sr[-1]))

