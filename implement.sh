python main.py --model_dir=/work/cascades/lxiaol9/6DPOSE/keypointnet/keypointnet_trained_model_car
--input=/work/cascades/lxiaol9/ARC/6DPOSE/keypointnet/keypointnet_trained_model_car/test_images/ --predict

# run training
python main.py --model_dir=MODEL_DIR --dset=DSET
total batch size of 256 (8 x 32 replicas).

python main.py --model_dir=/home/lxiaol9/6DPose2019/keypointnet/checkpoints/ --dset=/home/lxiaol9/6DPose2019/keypointnet/dataset/YCB_benchmarks/004_sugar_box/tfrecord/
python main.py --model_dir=/home/lxiaol9/6DPose2019/keypointnet/checkpoints/ --input=/home/lxiaol9/6DPose2019/keypointnet/dataset/YCB_benchmarks/004_sugar_box/test_images/ --predict

blender -b --python /home/lxiaol9/6DPose2019/keypointnet/tools/render.py -- -m /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/002_master_chef_can/tsdf/textured.obj -o /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/output/ -s 128 -n 120 -fov 5
blender -b --python /home/lxiaol9/6DPose2019/keypointnet/tools/render_mesh.py -- -m /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/002_master_chef_can/tsdf/textured.obj -o /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/mesh/ -s 128 -n 10 -fov 5

sbatch --array=1-4 -J blender

python tools/gen_tfrecords.py --input=/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/mesh --output=/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/mesh/tfrecord/train.tfrecord
python tools/gen_tfrecords_test.py --input=/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/mesh --output=/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/mesh/tfrecord/test.tfrecord

nvdu_viz /home/lxiaol9/WORK/6DPOSE/fat/mixed/kitchen_0 --name_filters *.left.jpg *.right.jpg
nvdu_viz /home/lxiaol9/WORK/6DPOSE/fat/single/002_master_chef_can_16k/kitchen_0 --name_filters *.left.jpg *.right.jpg

# depth data only
./blender -b --python /home/lxiaol9/3DGenNet2019/src/tools/render_depth.py \
-- -m /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/002_master_chef_can/tsdf/textured.obj \
-o /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/depth/ \
-t /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/002_master_chef_can/tsdf/textured.png \
-s 128 -n 5 -fov 5

# mengli
./blender -b --python /home/lxiaol9/3DGenNet2019/src/dataset/render/render.py -- \
-m /work/cascades/lxiaol9/6DPOSE/shapenet/ShapeNetCore.v2/02691156/b07608c9c3962cf4db73445864b72015/models/model_normalized.obj  \
-o /work/cascades/lxiaol9/test/ \
-s 128 -n 5 -fov 5

# GQN data
./blender -b --python /home/lxiaol9/3DGenNet2019/src/tools/render_auto3d.py \
-- -m /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/002_master_chef_can/tsdf/textured.obj \
-o /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/gqn/ \
-t /work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/ycb/002_master_chef_can/tsdf/textured.png \
-s 128 -n 5 -fov 5

# GQN tfrecord files
cd /home/lxiaol9/3DGenNet2019/src/dataset
python auto3d_gentfrecords.py --input=/work/cascades/lxiaol9/6DPOSE/ycb-benchmarks/gqn  \
                --output=/work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/ycb_benchmarks/train/0000-of-0003.tfrecord

# Run my model training
python3 train_auto3d_draw.py --debug\
  --data_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/ \
  --dataset ycb_benchmarks \
  --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/auto3d/

# Run my model training, hu013
python train_auto3d_draw.py --debug\
  --data_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/ \
  --dataset ycb_benchmarks --adam_lr_alpha 0.0005 \
  --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/auto3d/1/

# Run my model for training, hu013 for shapenet dataset
python train_ae3d_draw.py --debug\
  --data_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/ \
  --dataset shapenet --batch_size 32  \
  --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/gqn/0/

python train_ae3d_draw.py --debug\
  --data_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/ \
  --dataset shapenet --batch_size 32 --adam_lr_alpha 0.0005 \
  --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/gqn/1/

  python train_ae3d_draw.py --debug\
    --data_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/ \
    --dataset simple_shapenet --batch_size 16  \
    --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/gqn/2/
# run VAE encoder,
train_ae3d.py   --root_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/  \
                --dataset shapenet --batch_size=32  --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/ae3d/0/

python train_ae3d.py  --root_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/  \
                       --dataset shapenet --batch_size=64  --model_type=single_view_bottleneck \
                       --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/ae3d/4/

python train_ae3d.py --root_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/  \
                       --dataset shapenet --batch_size=16 --config_name=ae3d_params_novel_view.py    \
                       --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/ae3d/4/

python -m pdb train_ae3d.py --root_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/    \
                     --dataset shapenet --batch_size=8 --config_name=ae3d_params_novel_view.py     \
                     --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/ae3d/4/

python train_ae3d.py --root_dir /work/cascades/lxiaol9/6DPOSE/datasets_3DGEN/   \
                      --dataset shapenet --batch_size=16 --config_name=ae3d_params_novel_view.py \
                      --model_dir /work/cascades/lxiaol9/6DPOSE/checkpoints/ae3d/4/

ggID='0B12XukcbU7T7OHQ4MGh6d25qQlk'
ggURL='https://drive.google.com/uc?export=download'
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"

"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/benchviseblue.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/bowl.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/can.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cat.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cup.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/driller.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/duck.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/glue.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/holepuncher.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/iron.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/lamp.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/phone.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cam.zip"
"http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/eggbox.zip"
gqn/gqn_encoder.py: pool_encoder, tower_encoder;
gqn/gqn_draw.py:    generator_rnn, inference_rnn: GQNLSTMCell, GeneratorLSTMCell, InferenceLSTMCell;
gqn/gqn_graph.py:   gqn_draw, gqn_vae;
gqn/gqn_model.py:   gqn_draw_model_fn, gqn_draw_identity_model_fn, gqn_vae_model_fn;
gqn/gqn_objective.py: gqn_draw_elbo, gqn_vae_elbo;

#dataset generation with shapenet
./blender -b --python /home/lxiaol9/3DGenNet2019/src/tools/render_ae3d_shapenet.py \
-- -m /work/cascades/lxiaol9/6DPOSE/shapenet/ShapeNetCore.v2/02691156/fff513f407e00e85a9ced22d91ad7027/models/model_normalized.obj \
-o /work/cascades/lxiaol9/6DPOSE/shapenet/ae3d/ \
-s 128 -n 5 -fov 5
/work/cascades/lxiaol9/6DPOSE/shapenet/ShapeNetCore.v2/02691156/
fff513f407e00e85a9ced22d91ad7027/models/model_normalized.obj



bazel run -c opt :pretrain_rotator -- --step_size=160000 --init_model=/work/cascades/lxiaol9/6DPOSE/checkpoints/transformer --inp_dir=/work/cascades/lxiaol9/6DPOSE/shapenet_tf
bazel run -c opt :pretrain_rotator -- --step_size=160000 --init_model=/work/cascades/lxiaol9/6DPOSE/checkpoints/transformer1 --inp_dir=/work/cascades/lxiaol9/6DPOSE/shapenet_tf
