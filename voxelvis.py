#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.geometry import create_box
from vispy.visuals.transforms import MatrixTransform, STTransform

import numpy as np
import time
from matplotlib import pyplot as plt
def breakpoint():
    import pdb; pdb.set_trace()

class PointVis:
  """Class that creates and handles a visualizer for a pointcloud"""
  def __init__(self, target_pts=None, viz_dict=None, viz_point=True, viz_label=True, viz_joint=False, viz_box=False):
    self.viz_point = viz_point
    self.viz_label = viz_label
    self.viz_joint = viz_joint
    self.viz_box   = viz_box
    self.viz_label  = viz_label
    self.reset()

    self.update_scan(target_pts, viz_dict)

  def reset(self, sem_color_dict=None):
    """ Reset. """
    # new canvas prepared for visualizing data
    self.map_color(sem_color_dict=sem_color_dict)
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_view.camera = 'turntable'

    self.scan_vis = visuals.Markers()
    self.scan_view.add(self.scan_vis)

    if self.viz_joint:
      self.joint_vis = visuals.Arrow(connect='segments', arrow_size=18, color='blue', width=10, arrow_type='angle_60')
      self.arrow_length = 10
      self.scan_view.add(self.joint_vis)
    if self.viz_box:
      vertices, faces, outline = create_box(width=1, height=1, depth=1, width_segments=1, height_segments=1, depth_segments=1)
      vertices['color'][:, 3]=0.2
      # breakpoint()
      self.box = visuals.Box(vertex_colors=vertices['color'],
                                   edge_color='b')
      self.box.transform = STTransform(translate=[-2.5, 0, 0])
      self.theta = 0
      self.phi   = 0
      self.scan_view.add(self.box)
    visuals.XYZAxis(parent=self.scan_view.scene)

    # add nocs
    if self.viz_label:
      print("Using nocs in visualizer")
      self.nocs_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.nocs_view, 0, 1)
      self.label_vis = visuals.Markers()
      self.nocs_view.camera = 'turntable'
      self.nocs_view.add(self.label_vis)
      visuals.XYZAxis(parent=self.nocs_view.scene)
      self.nocs_view.camera.link(self.scan_view.camera)

  def update_scan(self, points, viz_dict=None):
    # then change names
    self.canvas.title = "scan "
    if viz_dict is not None:
        # input_point_seq  = viz_dict['input']
        # input_index_seq  = viz_dict['coord'] #(3, 150000, 2)
        # # npts = viz_dict['npts'] # (3,)
        # npts = [150000, 150000, 150000]
        # for i, num in enumerate(npts[:1]):
        #     target_pts = input_point_seq[i, 0, :num, 0:3]
        #     indices = input_index_seq[i, 0]
        #     idxs = np.where(indices[:, 0]==200)[0]
        #     target_pts = target_pts[idxs]
        #     indices = indices[idxs]
        #     y_value = [indices[100, 1], indices[800, 1], indices[1200,1]]
        #     idys = list(np.where(indices[:, 1]==y_value[2])[0]) # + list(np.where(indices[:, 1]==y_value[1])[0]) + list(np.where(indices[:, 1]==y_value[2])[0])
        #     target_pts = target_pts[idys]

        target_pts = np.concatenate([viz_dict['input1'], viz_dict['input2']], axis=0)
        gt_labels  = np.concatenate([viz_dict['label'], 4* np.ones((viz_dict['input2'].shape[0]), dtype=np.int32)], axis=0)
        print(np.max(target_pts, axis=0).reshape(1, 3))
        print(np.min(target_pts, axis=0).reshape(1, 3))
        # target_pts = target_pts - (np.max(target_pts, axis=0) + np.min(target_pts, axis=0))/2

        power = 16
        range_data = np.copy(np.linalg.norm(target_pts, axis=1))
        range_data = range_data**(1 / power)
        viridis_range = ((range_data - range_data.min()) /
                         (range_data.max() - range_data.min()) *
                         255).astype(np.uint8)
        viridis_map = self.get_mpl_colormap("viridis")
        viridis_colors = viridis_map[viridis_range]

        self.scan_vis.set_data(target_pts,
                               face_color=viridis_colors[..., ::-1],
                               edge_color=viridis_colors[..., ::-1],
                               size=5)

        # plot nocs
        if self.viz_label:
            label_colors = self.sem_color_lut[gt_labels]
            label_colors = label_colors.reshape((-1, 3))
            self.label_vis.set_data(target_pts,
                              face_color=label_colors[..., ::-1],
                              edge_color=label_colors[..., ::-1],
                              size=5)
            # time.sleep(15)
    if self.viz_joint:
      self.update_joints()

    if self.viz_box:
      self.update_boxes()

  def map_color(self, max_classes=20, sem_color_dict=None):
    # make semantic colors
    if sem_color_dict:
      # if I have a dict, make it
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.2)
      self.sem_color_lut[4] = np.full((3), 0.6)
      self.sem_color_lut[1] = np.array([1.0, 0.0, 0.0])
      self.sem_color_lut[2] = np.array([0.0, 0.0, 1.0])

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_joints(self, joints=None):
    # plot
    if joints is not None:
      start_coords  = joints['p'].reshape(1, 3)
      point_towards = start_coords + joints['l'].reshape(1, 3)
    else:
      start_coords  = np.array([[1, 0, 0],
                                [-1, 0, 0]])
      point_towards = np.array([[0, 0, 1], [0, 0, 1]])
    direction_vectors = (start_coords - point_towards).astype(
        np.float32)
    norms = np.sqrt(np.sum(direction_vectors**2, axis=-1))
    direction_vectors[:, 0] /= norms
    direction_vectors[:, 1] /= norms
    direction_vectors[:, 2] /= norms

    vertices = np.repeat(start_coords, 2, axis=0)
    vertices[::2]  = vertices[::2] + ((0.5 * self.arrow_length) * direction_vectors)
    vertices[1::2] = vertices[1::2] - ((0.5 * self.arrow_length) * direction_vectors)

    self.joint_vis.set_data(
        pos=vertices,
        arrows=vertices.reshape((len(vertices)//2, 6)),
    )

  def update_boxes(self):
    pass

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()

if __name__ == '__main__':
  for i in range(1):
    target_pts = np.random.rand(320,3) * 50
    labels = np.random.rand(320) * 10 # labels are encoded as color
    labels = labels.astype(np.int8)
    # we further have boxes, joints
    vis = PointVis(target_pts=target_pts, viz_joint=True, viz_box=True, viz_nocs=True)
    vis.run()
