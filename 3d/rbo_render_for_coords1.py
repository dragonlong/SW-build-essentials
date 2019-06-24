
#make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
#otherwise use testrender.py (slower but compatible without numpy)
#you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)
import numpy as np
import matplotlib.pyplot as plt
import pybullet

from scipy import misc
from skimage import io
import cv2
# from pyquaternion import Quaternion
from transformations import quaternion_from_euler
import h5py
import yaml
import time
from mathutils import Vector
import math
import platform
import sys
import os


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
_WRITE_FLAG  = True
_RENDER_FLAG = True
_CREATE_FOLDER = True
RENDER_NUM     = 700
TRANS_CNT      = 20
# RENDER_NUM     = 500
# TRANS_CNT      = 20

camTargetPos = [0, 0, 0]
cameraUp     = [0, 0, 1]        # z axis
cameraPos    = [-1.1, -1.1, 1.1]
# pybullet.connect(pybullet.GUI)
pybullet.connect(pybullet.DIRECT)

# pitch = -10
roll=0
# add by xiaolong for better data simulation,
# lightColor, vec3, list of 3 floats
# directional light color in [RED,GREEN,BLUE] in range 0..1
# lightDistance: float, distance of the light along the normalized lightDirection
# shadow: int, 1 for shadows, 0 for no shadows
# lightAmbientCoeff, float, light ambient coefficient
# lightDiffuseCoeff, float, light diffuse coefficient,
# lightSpecularCoeff, float,light specular coefficient
# initialization of angles
steeringAngle = 0
camPosX       = 0
camPosY       = 0
camPosZ       = 0

upAxisIndex = 2 # align with z
camDistance = 0.65
pixelWidth  = 512
pixelHeight = 512
nearPlane   = 0.01
farPlane    = 100

fov = 60

# camInfo = pybullet.getDebugVisualizerCamera()
obj_name = 'laptop_challenge'
planeId=pybullet.loadURDF("plane.urdf", [0, 0, -1])
obj=pybullet.loadURDF("laptop1_b0.urdf") # is the origin on
obj_base=pybullet.loadURDF("laptop1_b1.urdf")
pybullet.setGravity(0, 0, -10)
#>>>>>>>>>>>>>>>>>>>>>>---------Rendering setup ended----------<<<<<<<<<<<<<<<<<<<<<<<<<#
# plt.switch_backend('agg')
# plt.ion()
# img = np.random.rand(200, 320)
# #img = [tandard_normal((50,100))
# image = plt.imshow(img,interpolation='none',animated=True,label="blah")
# ax = plt.gca()
for joint in range(pybullet.getNumJoints(obj)):
    print("joint[",joint,"]=", pybullet.getJointInfo(obj,joint))
    pybullet.setJointMotorControl2(obj,joint,pybullet.VELOCITY_CONTROL,targetVelocity=0,force=0)
    pybullet.getJointInfo(obj, joint)
#>>>>>>>>>>>>>>>>>>>>>>>>-------set constraints--------<<<<<<<<<<<<<<<<<<<<<<#
# <constraint between different objects and child links>
# c = p.createConstraint(obj, 9, obj,11,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
# p.changeConstraint(c, gearRatio=1, maxForce=10000)

#>>>>>>>>>>>>>>>>>>>>>>>>-------Interactions for Vis--------<<<<<<<<<<<<<<<<<<<<<<#
# targetVelocitySlider = pybullet.addUserDebugParameter("wheelVelocity",-50,50,0)
# maxForceSlider = pybullet.addUserDebugParameter("maxForce",0,50,20)
steeringSlider= pybullet.addUserDebugParameter("steering",-1,1,0) # start value with 0
camPosXSlider = pybullet.addUserDebugParameter("X",-1,1,0)
camPosYSlider = pybullet.addUserDebugParameter("Y",-1,1,0)
camPosZSlider = pybullet.addUserDebugParameter("Z",-1,1,0)


if platform.uname()[0] == 'Darwin':
    print("Now it knows it's in my local Mac")
    base_path = '/Users/DragonX/Downloads/NR_WORK/6DPOSE/partnet/pose_articulated'
elif platform.uname()[1] == 'viz1':
    base_path = '/home/xiaolong/Downloads/6DPOSE/partnet/pose_articulated'
elif platform.uname()[1] == 'vllab3':
    base_path = '/mnt/data/lxiaol9/rbo'
else:
    base_path = '/work/cascades/lxiaol9/6DPOSE/partnet/pose_articulated'


simu_cnt   = 0
main_start = time.time() # measure how long it will take for the whole rendering
rdn_offset             = np.random.rand(TRANS_CNT, RENDER_NUM)
lightDirectionArray    = np.random.rand(TRANS_CNT, RENDER_NUM, 3) - 0.5

 # ambient component using a percentage of the original intensities of the light source, float with a value between zero (0%) and one (100%)
lightDistanceArray          = 1 + 10* np.random.rand(TRANS_CNT, RENDER_NUM)
lightColorArray             = np.random.rand(TRANS_CNT, RENDER_NUM, 3)
# specular coefficient", which is the brightness of the reflection. x=cos(θ)s, x is the specular coefficient, θ is the angle, s is the specular exponent
lightSpecularCoeffArray     = np.random.rand(TRANS_CNT, RENDER_NUM)
# reflection model basically specifies a minimum brightness
lightAmbientCoeffArray      = np.random.rand(TRANS_CNT, RENDER_NUM)
lightDiffuseCoeffArray      = 0.1 + 0.9 * np.random.rand(TRANS_CNT, RENDER_NUM)
while (simu_cnt < TRANS_CNT):
    if (not os.path.exists(base_path + '/{}/{}/depth/'.format(obj_name, simu_cnt))) and _CREATE_FOLDER:
        os.makedirs(base_path + '/{}/{}/depth/'.format(obj_name, simu_cnt))
        os.makedirs(base_path + '/{}/{}/rgb/'.format(obj_name, simu_cnt))
        os.makedirs(base_path + '/{}/{}/mask/'.format(obj_name, simu_cnt))
    yml_dict = OrderedDict()
    yml_file = base_path + '/{}/{}/gt.yml'.format(obj_name, simu_cnt)
    # set articulation status
    for steer in [0]:
        pybullet.setJointMotorControl2(obj, steer, pybullet.POSITION_CONTROL, targetPosition=-steeringAngle)
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

    for pitch in range (0,360,10):
        for yaw in range(0,360,15):
            # in this way, you could also control number of images generated
            if(img_id < RENDER_NUM and _RENDER_FLAG):
                nowTime = time.time()
                ################# Interaction ###############
                # steeringAngle = pybullet.readUserDebugParameter(steeringSlider)
                # camPosX = pybullet.readUserDebugParameter(camPosXSlider)
                # camPosY = pybullet.readUserDebugParameter(camPosYSlider)
                # camPosZ = pybullet.readUserDebugParameter(camPosZSlider
                ################ used for querying the state of every parts
                # if (nowTime - lastTime>1):
                ################# setting regarding to camera pose ###################
                # method 1: Use computeProjectionMatrixFOV
                # camPos = tuple([cameraPos[0] + camPosX, cameraPos[1] + camPosY, cameraPos[2] + camPosZ])
                # target_point = Vector((0.0, 0.0, 0.0))
                # camOrn = tuple(look_at(Vector(camPos), target_point))
                # camMat = pybullet.getMatrixFromQuaternion(camOrn)
                # forwardVec = [camMat[0], camMat[3], camMat[6]]
                # #sideVec =  [camMat[1],camMat[4],camMat[7]]
                # camUpVec =  [camMat[2], camMat[5], camMat[8]]
                # camUpTarget = [camPos[0]+camUpVec[0],camPos[1]+camUpVec[1],camPos[2]+camUpVec[2]]
                # camTarget = [0, 0, 0]
                # upVector = [0, 0, 1]      # along z axis
                # viewMat = pybullet.computeViewMatrix(camPos, camTarget, upVector)
                # aspect = pixelWidth / pixelHeight
                # projMat = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
                # img_arr = pybullet.getCameraImage(pixelWidth, pixelHeight, viewMat, projMat, shadow=1, lightDirection=[1,1,1], lightColor=[1, 1, 1],  renderer=pybullet.ER_TINY_RENDERER)#pybullet.ER_BULLET_HARDWARE_OPENGL)#
                offset = rdn_offset[simu_cnt, img_id]
                camDistance_final = camDistance + offset

                viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance_final, yaw, pitch, roll, upAxisIndex)
                aspect = pixelWidth / pixelHeight
                projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
                lightDirection         = [10*lightDirectionArray[simu_cnt, img_id, 0], 10*lightDirectionArray[simu_cnt, img_id, 1], 10*(lightDirectionArray[simu_cnt, img_id, 1]+0.6)]
                lightDistance          = lightDistanceArray[simu_cnt, img_id]
                lightColor             = list(np.array([0.5, 0.5, 0.5]) + 0.5 * lightColorArray[simu_cnt, img_id, :])
                lightAmbientCoeff      = lightAmbientCoeffArray[simu_cnt, img_id]
                lightDiffuseCoeff      = lightDiffuseCoeffArray[simu_cnt, img_id]
                lightSpecularCoeff     = lightSpecularCoeffArray[simu_cnt, img_id]
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
                image_path = base_path + '/{}/{}'.format(obj_name, simu_cnt)


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
                                                  ])

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
    steeringAngle +=0.1
    simu_cnt      += 1
    # plt.pause(0.4)
    stop = time.time()

main_stop = time.time()
print ("Total time %f" % (main_stop - main_start))
pybullet.resetSimulation()
