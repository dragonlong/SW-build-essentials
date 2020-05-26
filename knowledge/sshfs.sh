sudo apt-get install sshfs
sudo sshfs -o allow_other,defer_permissions,IdentityFile=~/.ssh/google_compute_engine xiaolongli@35.238.86.75:~ /usr/local/google/home/xiaolongli/xiaolong-simu

gcloud beta compute --project "brain-reach-testing" ssh --zone "us-central1-a" "xiaolong-simu"
sshfs -o  IdentityFile=~/.ssh/google_compute_engine.pub <user_name>@<instance-name>.<region>.<project_id>:/home/<user_name> /mnt/gce

ps -A | grep sshfs
fusermount -u YOUR_MNT_DIR
sudo umount -l YOUR_MNT_DIR
sshfs lxiaol9@newriver1.arc.vt.edu:/work/cascades/lxiaol9/ ./ARCwork/
sshfs lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/ ./ARChome/
