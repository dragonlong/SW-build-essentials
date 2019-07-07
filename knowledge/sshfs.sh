sudo apt-get install sshfs
sudo sshfs -o allow_other,defer_permissions,IdentityFile=~/.ssh/google_compute_engine xiaolongli@35.238.86.75:~ /usr/local/google/home/xiaolongli/xiaolong-simu

gcloud beta compute --project "brain-reach-testing" ssh --zone "us-central1-a" "xiaolong-simu"
