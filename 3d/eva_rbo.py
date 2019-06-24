"""
Code for coords regression + articulation params RANSAC
Following BMVC 2015 paper
Author: Xiaolong Li
"""
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()
import _init_paths
import argparse
import os
import random
from random import randint
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import _init_paths
from datasets.rbo.dataset_pascal import PoseDataset as PoseDataset_rbo
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default = 'rbo', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '/work/cascades/lxiaol9/6DPOSE/partnet/pose_articulated', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--model', type=str, default ='/work/cascades/lxiaol9/6DPOSE/checkpoints/densecoord/0.6/pose_model_220_0.0038036994917906427.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def ransac_pose(x_whole, y_whole):
    """
    x_whole: original 3D points cloud cooridinates in camera space;
    y_whole: estimated coordinates in local canonical part coordinates;

    output:
    y_k: coordinates estimated in the shared coordiates of part k;
    R/T: rotation and translation for part k from canonical to camera;
    """
    # pick x,y pairs from the results;
    # angle estimation;
    # angle selection by applying the range limit;
    # pose estimation
    # rank pose hopothyesis via energy minimization
    # for i in range(50):
    i = 0
    k_depth = 1
    k_coord = 1
    k_obj   = 1
    E_depth = None
    E_coord = None
    E_obj   = None # mask segmentation
    E_overall = k_depth * E_depth + k_coord * E_coord + k_obj * E_obj
    # Nelder-Mead simplex optimization for theta and H
    return None

def angle_estimation(pt1_cam, pt2_cam, pt1, pt2, art_type='revolute'):
    """
    pt1: estimated coordinates in local reference system;
    pt2: estimated coordinates in local reference system;

    output
    angle  : the articulation angle compared to canonical pose
    """
    # each point is a numpy array
    dx         = np.norm(pt1_cam - pt2_cam)**2
    a          = 2*(pt1[1] * pt2[2] - pt1[2] * pt2[1])
    b          = 2*(pt1[1] * pt2[1] + pt1[2] * pt2[2])
    dis        = dx - (pt1[0] - pt2[0])**2 - pt1[1]**2 - pt2[1]**2 - pt1[2]**2 - pt2[2]**2
    bias_ang   = np.arctan2(b, a)
    # by default the joint axis is the [1, 0, 0] around x axis
    theta_a    = np.arcsin(dis/np.norm(a**2 + b**2)) - bias_ang
    theta_b    = np.pi - theta_a

    return [theta_a, theta_b]

def pose_estimation():
    """
    Here we implement the code for Kabsch pose estimation, ePnP for pose
    """
    return None

if __name__=="__main__":

    num_objects = 3
    objlist = [0, 1, 2]
    objlist = ['laptop_new', 'laptop', 'laptop_near']
    num_points = 500
    iteration = 2
    bs = 1
    output_result_dir = 'experiments/eval_result/synthetic'
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)

    knn = KNearestNeighbor(1)
    # model
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()
    # data
    testdataset = PoseDataset_rbo('test', num_points, False, opt.dataset_root, 0.0, True)
    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()

    #>>>>>>>>>>>>>>>>> how to get the diameter for each objects?? <<<<<<<<<<<<<<<<<<
    diameter = [1, 1, 1]
    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

    opt.num_points = 500
    criterion = Loss(500, [7, 8])
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    # for i, data in enumerate(testdataloader, 0):
    index = 5
    data = testdataset.__getitem__(5)
    img, points, cloud_canon, model_points, choose, mask, num_parts, idx = data
    img = img.unsqueeze(0)
    points = points.unsqueeze(0)
    cloud_canon = cloud_canon.unsqueeze(0)
    points = Variable(points).cuda(0)
    choose = choose.view(-1, 1, opt.num_points)
    choose = Variable(choose).cuda(0)
    img = Variable(img).cuda(0)
    idx = Variable(idx).cuda(0)
    cloud_canon = Variable(cloud_canon).cuda(0)
    pred_r_whole, pred_c_whole, emb = estimator(img, points, choose, idx)
    loss, dis =  criterion(pred_r_whole, pred_c_whole, cloud_canon, idx, 0.01, False)

    #>>>>>>>>>>>>>>>>>>>>>> split pred_r, prediction <<<<<<<<<<<<<<<<<#
    num_parts    = len(mask)
    bs, num_p, _ = pred_c_whole.size()
    pred_c_list = torch.split(pred_c_whole, int(num_p/num_parts), dim=1)
    pred_r_list = torch.split(pred_r_whole, int(num_p/num_parts), dim=1)
    points_list = torch.split(points, int(num_p/num_parts), dim=1) # by default [bs, 500, 3]
    target_list = torch.split(cloud_canon, int(num_p/num_parts), dim=1)

    num_points = int(num_p/num_parts)


    from mpl_toolkits.mplot3d import axes3d

    fig1 = plt.figure(dpi=150)
    ax = plt.subplot(111, projection='3d')
    ax.set_aspect('equal')
    groups = ['model points', 'target points', 'masked cloud']
    color_set = ['b', 'k']
    for j in range(2):
        pred_r = pred_r_list[j]
        pred_c = pred_c_list[j]
        pred_c = pred_c.view(bs, num_points)

        target_point_part = target_list[j].cpu().detach().numpy()[0]
        pred_point_part = pred_r.cpu().detach().numpy()
        model_points_part = model_points[j].numpy()
        print('pred part {}:\n'.format(j), pred_point_part.shape)
        print('target part {}:\n'.format(j), target_point_part.shape)
        print('model part {}:\n'.format(j), pred_point_part.shape)

        pred_target0   = pred_point_part[0]
        input_point_part = points_list[j][0]
        ax.scatter(pred_target0[:, 0], pred_target0[:, 1], pred_target0[:, 2], c=color_set[j], label='part{} pred'.format(j))
        ax.scatter(target_point_part[:, 0], target_point_part[:, 1], target_point_part[:, 2], c='r', label='part{} GT'.format(j))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-0.4, .00)
    ax.set_xlim(-0.35, 0.1)
    set_axes_equal(ax)
    plt.grid()
    plt.legend(loc=1)
    plt.title('3D points')
    plt.show()
    plt.pause(1)
    plt.close()
    # >>>>>>>>>>>>>>> points in joint coords
    select_ind = [randint(0, 249) for j in range(num_parts)]
    coords_urdf = [pred_r_list[j].cpu().detach().numpy() for j in range(num_parts)]
    coords_cam  = [points_list[j].cpu().detach().numpy() for j in range(num_parts)]
    pts     = []
    pts_cam = []
    for j in range(num_parts):
        while(coords_urdf[j][0][select_ind[j]][2] < 0.1):
            select_ind[j] = randint(0, 249)
        pts.append(coords_urdf[j][0][ select_ind[j] ])
        pts_cam.append(coords_cam[j][0] [ select_ind[j] ])
    # top
    pt1_cam  = pts_cam[1][1, 2, 0]
    pt2_cam  = pts_cam[0][1, 2, 0]
    pt1      = pts[1]  - np.array([0, 0, 0.04]) # bottom
    pt1      = pt1[1, 2, 0]
    pt2      = pts[0]  - np.array([0, 0, 0.04]) # top
    pt2      = pt2[1, 2, 0]
    arti_ang = angle_estimation(pt1_cam, pt2_cam, pt1, pt2, art_type='revolute')
    arti_ang_gt = testdataset.list_status[index]
    # visit every data sample to get overall accuracy
    total_dis = []
    total_dis0 = []
    total_dis1 = []
    for i, data in enumerate(testdataloader):
        img, points, cloud_canon, model_points, choose, mask, num_parts, idx = data
        if idx == torch.LongTensor([0]):
    #         print('No.{0} NOT Pass! Not enough rigion!'.format(i))
            fw.write('No.{0} NOT Pass! Not enough rigion!\n'.format(i))
            continue
        points = Variable(points).cuda(0)
        choose = choose.view(-1, 1, opt.num_points)
        choose = Variable(choose).cuda(0)
        img = Variable(img).cuda(0)
        idx = Variable(idx).cuda(0)
        cloud_canon = Variable(cloud_canon).cuda(0)
        pred_r_whole, pred_c_whole, emb = estimator(img, points, choose, idx)
        loss, dis =  criterion(pred_r_whole, pred_c_whole, cloud_canon, idx, 0.01, False)
        coords_pred  = pred_r_whole.cpu().detach().numpy()
        coords_target= cloud_canon.cpu().detach().numpy()
        dis          = np.mean(np.linalg.norm(coords_pred - coords_target, axis=2))
        dis_0        = np.mean(np.linalg.norm(coords_pred[0:1, 0:250, :] - coords_target[0:1, 0:250, :], axis=2))
        dis_1        = np.mean(np.linalg.norm(coords_pred[0:1, 250:, :] - coords_target[0:1, 250:, :], axis=2))
        total_dis.append(dis)
        total_dis0.append(dis_0)
        total_dis1.append(dis_1)
        print(coords_pred.shape, dis)
        #>>>>>>>>>>>>>>>>>>>>>> split pred_r, prediction <<<<<<<<<<<<<<<<<#
        num_parts    = len(mask)
        bs, num_p, _ = pred_c_whole.size()
        pred_c_list = torch.split(pred_c_whole, int(num_p/num_parts), dim=1)
        pred_r_list = torch.split(pred_r_whole, int(num_p/num_parts), dim=1)
        points_list = torch.split(points, int(num_p/num_parts), dim=1) # by default [bs, 500, 3]
        target_list = torch.split(cloud_canon, int(num_p/num_parts), dim=1)
        num_points = int(num_p/num_parts)
        #>>>>>>>>>>>>>>>>>>>>>> split pred_r, prediction <<<<<<<<<<<<<<<<<#
    print('Averaging distance error on whole objects: {}', np.mean(np.array(total_dis)))
    print('Averaging distance error on part 0: {}', np.mean(np.array(total_dis0)))
    print('Averaging distance error on part 1: {}', np.mean(np.array(total_dis1)))
