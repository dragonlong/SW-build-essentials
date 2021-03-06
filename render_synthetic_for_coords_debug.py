"""
Used for data rendering from URDF
Author: Xiaolong Li
# make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
# otherwise use testrender.py (slower but compatible without numpy)
# you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)
"""
import numpy as np
import matplotlib.pyplot as plt
import pybullet

import math
import platform
import sys
import os
import time

from scipy import misc
from skimage import io
import cv2
from transformations import quaternion_from_euler
import h5py
import yaml
from mathutils import Vector

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom

try:
        # for python newer than 2.7
    from collections import OrderedDict
except ImportError:
        # use backport from pypi
    from ordereddict import OrderedDict

# try to use LibYAML bindings if possible
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)

Dumper.add_representer(str,
                       SafeRepresenter.represent_str)


def look_at(loc_camera, point):
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Y', 'Z')
    # rotation_euler = rot_quat.to_euler()
    return rot_quat

def print_viewpoint(camPos, camOrn):
    # yaw, pitch, roll
    for element in camPos:
        f.write("%lf " %element)
    for ang in camOrn:
        f.write("%lf " %ang)
    f.write("\n")

def print_matrix(f, mat):
    for i in range(4):
        for j in range(4):
            f.write("%lf " % mat[i][j])
        f.write("\n")

#>>>>>>>>>>>>>>>>>>>>>>---------Rendering setup----------<<<<<<<<<<<<<<<<<<<<<<<<<#
def render_data(path_urdf, ind_urdf, _WRITE_FLAG=True, _RENDER_FLAG=True, _CREATE_FOLDER=True, RENDER_NUM=100, ARTIC_CNT=20):
    camTargetPos = [0, 0, 0]
    cameraUp     = [0, 0, 1]        # z axis
    cameraPos    = [-1.1, -1.1, 1.1]
    pybullet.connect(pybullet.GUI)
    # pybullet.connect(pybullet.DIRECT)

    # add by xiaolong for better data simulation,
    # lightColor:     vec3, list of 3 floats
    # >>>>>>>>>>>>>>> directional light color in [RED,GREEN,BLUE] in range 0..1
    # lightDistance: float,
    # >>>>>>>>>>>>>>> distance of the light along the normalized lightDirection
    # shadow: int,
    # >>>>>>>>>>>>>>> 1 for shadows, 0 for no shadows
    # lightAmbientCoeff, float,
    # >>>>>>>>>>>>>>> light ambient coefficient
    # lightDiffuseCoeff, float,
    # >>>>>>>>>>>>>>> light diffuse coefficient
    # lightSpecularCoeff, float,
    # >>>>>>>>>>>>>>> light specular coefficient
    # initialization of angles
    # pitch = -10
    roll=0
    steeringAngle = 0
    camPosX       = 0
    camPosY       = 0
    camPosZ       = 0

    upAxisIndex = 2 # align with z
    camDistance = 2.5
    pixelWidth  = 512
    pixelHeight = 512
    nearPlane   = 0.01
    farPlane    = 100
    fov = 90

    camInfo  = pybullet.getDebugVisualizerCamera()
    # planeId   = pybullet.loadURDF("plane.urdf", [0, 0, -1])
    obj       = pybullet.loadURDF("{}/{:04d}/syn.urdf".format(path_urdf, ind_urdf))
    tree_urdf = ET.parse("{}/{:04d}/syn.urdf".format(path_urdf, ind_urdf))
    root      = tree_urdf.getroot()

    num_joints = 0
    for joint in root.iter('joint'):
        num_joints +=1
    print('We have {} joints here'.format(num_joints))
    #>>>>>>>>>>>>>>>>>>>>>-------- where we could add more seperate URDF files---------<<<<<<<<<<<<<<<<#

    #>>>>>>>>>>>>>>>>>>>>>-------- Pybullet simulation env configuration---------------<<<<<<<<<<<<<<<<#
    pybullet.setGravity(0, 0, -10)
    pybullet.setRealTimeSimulation(1)
    #>>>>>>>>>>>>>>>>>>>>>>---------Rendering setup ended----------<<<<<<<<<<<<<<<<<<<<<<<<<#
    for joint in range(pybullet.getNumJoints(obj)):
        print("joint[",joint,"]=", pybullet.getJointInfo(obj,joint))
        pybullet.setJointMotorControl2(obj,joint,pybullet.VELOCITY_CONTROL,targetVelocity=0,force=0)
        pybullet.getJointInfo(obj, joint)
    #>>>>>>>>>>>>>>>>>>>>>>>>-------set constraints--------<<<<<<<<<<<<<<<<<<<<<<#
    #>>>>>>>>>>>>>>>>>>>>>>>>-------Interactions for Vis--------<<<<<<<<<<<<<<<<<<<<<<#
    steeringSlider              = pybullet.addUserDebugParameter("steering",-1,1,0) # start value with 0
    rdn_offsetSlider            = pybullet.addUserDebugParameter("rdn_offset",-1,1,0)
    lightDistanceSlider         = pybullet.addUserDebugParameter("lightDistance",1,10,0)
    lightSpecularCoeffSlider    = pybullet.addUserDebugParameter("lightSpecularCoeff",0,1,0.5)
    lightAmbientCoeffSlider     = pybullet.addUserDebugParameter("lightAmbientCoeff", 0,1,0.5)
    lightDiffuseCoeffSlider     = pybullet.addUserDebugParameter("lightDiffuseCoeff", 0,1,0.5)

    lightDirectionArray         = 10* np.ones((ARTIC_CNT, RENDER_NUM, 3)) # coming direction of light
    lightColorArray             = np.ones((ARTIC_CNT, RENDER_NUM, 3))

    if platform.uname()[0] == 'Darwin':
        print("Now it knows it's in my local Mac")
        base_path = '/Users/DragonX/Downloads/ARC/6DPOSE/synthetic'
    elif platform.uname()[1] == 'viz1':
        base_path = '/home/xiaolong/Downloads/6DPOSE/synthetic'
    elif platform.uname()[1] == 'vllab3':
        base_path = '/mnt/data/lxiaol9/rbo'
    else:
        base_path = '/work/cascades/lxiaol9/6DPOSE/partnet/synthetic'

    simu_cnt   = 0
    main_start = time.time() # measure how long it will take for the whole rendering

    # rdn_offset                  = np.random.rand(ARTIC_CNT, RENDER_NUM)        # camera offset
    # lightDirectionArray         = 10* np.random.rand(ARTIC_CNT, RENDER_NUM, 3) # coming direction of light
    # lightDistanceArray          = 5   + 10  * np.random.rand(ARTIC_CNT, RENDER_NUM)
    # lightColorArray             = 0.5 + 0.5 * np.random.rand(ARTIC_CNT, RENDER_NUM, 3)
    # # specular coefficient", which is the brightness of the reflection. x=cos(θ)s, x is the specular coefficient, θ is the angle, s is the specular exponent
    # lightSpecularCoeffArray     = 0.9 + 0.1 * np.random.rand(ARTIC_CNT, RENDER_NUM)
    # # ambient component using a percentage of the original intensities of the light source, float with a value between zero (0%) and one (100%)
    # lightAmbientCoeffArray      = 0.9 + 0.1 * np.random.rand(ARTIC_CNT, RENDER_NUM)
    # lightDiffuseCoeffArray      = 0.9 + 0.1 * np.random.rand(ARTIC_CNT, RENDER_NUM)

    while (simu_cnt < ARTIC_CNT):
        if (not os.path.exists(base_path + '/{0:04d}/{1}/depth/'.format(ind_urdf, simu_cnt))) and _CREATE_FOLDER:
            os.makedirs(base_path + '/{0:04d}/{1}/depth/'.format(ind_urdf, simu_cnt))
            os.makedirs(base_path + '/{0:04d}/{1}/rgb/'.format(ind_urdf, simu_cnt))
            os.makedirs(base_path + '/{0:04d}/{1}/mask/'.format(ind_urdf, simu_cnt))
        yml_dict = OrderedDict()
        yml_file = base_path + '/{0:04d}/{1}/gt.yml'.format(ind_urdf, simu_cnt)
        # set articulation status
        for steer in range(num_joints):
            pybullet.setJointMotorControl2(obj, steer, pybullet.POSITION_CONTROL, targetPosition=steeringAngle)
        pybullet.stepSimulation()

        joint_pos = OrderedDict()
        for joint in range(pybullet.getNumJoints(obj)):
            lstate = pybullet.getLinkState(obj, linkIndex=joint, computeForwardKinematics=True)
            joint_pos[joint] = OrderedDict(
                                [(0, list(lstate[0])),
                                 (1, list(lstate[1])),
                                 (2, list(lstate[2])),
                                 (3, list(lstate[3])),
                                 (4, list(lstate[4])),
                                 (5, list(lstate[5]))]
            )
        img_id = 0
        lastTime = time.time()

        for yaw in [50]:
            for pitch in [-35]*RENDER_NUM:
                if(img_id < RENDER_NUM and _RENDER_FLAG):
                    nowTime = time.time()
                    ################# Interaction ###############
                    # steeringAngle = pybullet.readUserDebugParameter(steeringSlider)
                    # offset                 = rdn_offset[simu_cnt, img_id]
                    # lightDirection         = lightDirectionArray[simu_cnt, img_id, :]
                    # lightDistance          = lightDistanceArray[simu_cnt, img_id]
                    # lightColor             = list(lightColorArray[simu_cnt, img_id, :])
                    # lightAmbientCoeff      = lightAmbientCoeffArray[simu_cnt, img_id]
                    # lightDiffuseCoeff      = lightDiffuseCoeffArray[simu_cnt, img_id]
                    # lightSpecularCoeff     = lightSpecularCoeffArray[simu_cnt, img_id]
                    offset                 = pybullet.readUserDebugParameter(steeringSlider)
                    lightDirection         = lightDirectionArray[simu_cnt, img_id, :]
                    lightDistance          = pybullet.readUserDebugParameter(lightDistanceSlider)
                    lightColor             = list(lightColorArray[simu_cnt, img_id, :])
                    lightAmbientCoeff      = pybullet.readUserDebugParameter(lightAmbientCoeffSlider)
                    lightDiffuseCoeff      = pybullet.readUserDebugParameter(lightDiffuseCoeffSlider)
                    lightSpecularCoeff     = pybullet.readUserDebugParameter(lightSpecularCoeffSlider)

                    camDistance_final      = camDistance + offset
                    viewMatrix             = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance_final, yaw, pitch, roll, upAxisIndex)
                    aspect                 = pixelWidth / pixelHeight
                    projectionMatrix       = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

                    img_arr = pybullet.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1, \
                                                      lightDirection=lightDirection, \
                                                      lightColor=lightColor,\
                                                      lightDistance=lightDistance,\
                                                      lightAmbientCoeff=lightAmbientCoeff,\
                                                      lightDiffuseCoeff=lightDiffuseCoeff,\
                                                      lightSpecularCoeff=lightSpecularCoeff,\
                                                      renderer=pybullet.ER_TINY_RENDERER)
                    w=img_arr[0]        # width of the image, in pixels
                    h=img_arr[1]        # height of the image, in pixels
                    rgb       = img_arr[2]    # color data RGB
                    depth_raw = img_arr[3].astype(np.float32) #depth data
                    mask      = img_arr[4]
                    depth     = 255.0 * nearPlane / (farPlane - (farPlane - nearPlane) * depth_raw) # *farPlane/255.0
                    far  = farPlane
                    near = nearPlane
                    depth_to_save = 2.0 * far * near / (far  + near - (far - near) * (2 * depth_raw - 1.0))

                    np_rgb_arr = np.reshape(rgb, (h, w, 4))[:, :, :3]
                    np_depth_arr = np.reshape(depth, (h, w, 1))#.astype(np.uint8)
                    np_mask_arr = (np.reshape(mask, (h, w, 1))).astype(np.uint8)
                    image_path = base_path + '/{0:04d}/{1}'.format(ind_urdf, simu_cnt)

                    rgb_name   = image_path + '/rgb/{0:06d}.png'.format(img_id)
                    depth_img_name   = image_path + '/depth/{0:06d}.png'.format(img_id)
                    depth_name   = image_path + '/depth/{0:06d}.h5'.format(img_id)
                    mask_name  = image_path + '/mask/{0:06d}.png'.format(img_id)
                    print("rendering Image %d" % (img_id))

                    if _WRITE_FLAG is True:
                        misc.imsave(rgb_name, np_rgb_arr)
                        cv2.imwrite(depth_img_name, np_depth_arr)
                        cv2.imwrite(mask_name, np_mask_arr)
                        hf = h5py.File(depth_name, 'w')
                        hf.create_dataset('data', data=depth_to_save)
                    yml_dict['frame_{}'.format(img_id)] = OrderedDict( [ ('obj', joint_pos),
                                                      ('viewMat', list(viewMatrix)),
                                                      ('projMat', list(projectionMatrix))
                                                      ] )
                    #>>>>>>>>>>>>>>>>>>>>>>---------Image Infos----------<<<<<<<<<<<<<<<<<<<<<<<<<#
                    # fig = plt.figure(dpi=200)
                    # plt.imshow(np.squeeze(np_mask_arr),interpolation='none',animated=True,label="blah")
                    # plt.show()
                    # plt.pause(0.1)
                    img_id+=1
                    lastTime = nowTime
                    # pybullet.addUserDebugLine(list(camPos), list(target_point), lineColorRGB=[1.0, 0, 0], lineWidth=5, lifeTime=100)
        if _WRITE_FLAG:
            with open(yml_file, 'w') as f:
                yaml.dump(yml_dict, f, default_flow_style=False)
        steeringAngle +=0.15
        simu_cnt      += 1
        plt.pause(0.01)
        stop = time.time()

    main_stop = time.time()
    print ("Total time %f" % (main_stop - main_start))
    pybullet.resetSimulation()

if __name__ == "__main__":
    #     5000   *      20      *     5          images with different params(random illumination, random )
    #(every urdf) (articulation) (view angles)
    _WRITE   = False
    _RENDER  = True
    _CREATE  = False
    num_render= 500   # per articulation status
    cnt_artic = 5    # number of articulation change
    path_urdf = "/Users/DragonX/Downloads/ARC/DATA"
    ind_urdf  = 0
    render_data(path_urdf, ind_urdf, _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, _CREATE_FOLDER=_CREATE, RENDER_NUM=num_render, ARTIC_CNT=cnt_artic)
