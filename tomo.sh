source activate tf_cc
which python
MY_PATH='/work/cascades/lxiaol9/Zhengrui_Tomo_data_analysis'
CONDA_PATH='/home/lxiaol9/anaconda3/envs/tf_cc'
WORK_PATH='/work/cascades/lxiaol9/Zhengrui_Tomo_data_analysis/Tomo_3D_Reconstruction'
conda install -c conda-forge tomopy
cp $MY_PATH/Setting_up_environment/stripe.py $CONDA_PATH/lib/python3.7/site-packages/tomopy/prep/
cp $MY_PATH/Setting_up_environment/phase.py $CONDA_PATH/lib/python3.7/site-packages/tomopy/prep/
cp $MY_PATH/Setting_up_environment/rotation.py $CONDA_PATH/lib/python3.7/site-packages/tomopy/recon/

conda install -c conda-forge dxchange
git clone https://github.com/xianghuix/dxchange.git
cp $MY_PATH/Setting_up_environment/writer.py $CONDA_PATH/lib/python3.7/site-packages/dxchange

pip install pystackreg
# working folder
cd $WORK_PATH
python BNL_FXI_tomo_recon_template_v2.py
