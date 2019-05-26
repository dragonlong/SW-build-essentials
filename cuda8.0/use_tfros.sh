# https://blog.csdn.net/weixin_40527235/article/details/80385141
rosrun mask_rcnn_ros mask_rcnn_node /mask_rcnn/input:=/camera/rgb/image_color
rosrun image_view image_view image:=/camera/rgb/image_color

