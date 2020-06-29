alias ..='cd ..'
alias apt-get='sudo apt-get'
alias ll='ls -la'
alias l.='ls -d .* --color=auto'
alias h='history'
alias j='jobs -l'
alias rm='rm -I --preserve-root'
alias code='cd /homes/xili/cv0_home/AutoSeg/'
alias bev='cd /homes/xili/cv0_home/AutoSeg/fuse_1.1d/'
alias data='cd /homes/xili/datasets/'
alias kitti='cd /homes/xili/datasets/kitti'
alias ki='cd /homes/xili/datasets/kitti'
alias log='cd /homes/xili/cv0_home/log/'
alias pred='cd /homes/xili/datasets/kitti/predictions'
alias pr='cd /homes/xili/datasets/kitti/predictions'
alias nu='cd /homes/xili/datasets/nuscene'
alias base='cd /homes/xili/cv0_home/AutoSeg/lidar-bonnetal/train/tasks/semantic'
alias sq='squeue'
alias nv='nvidia-smi'
alias rv='cd /homes/xili/cv0_home/AutoSeg/lidar-bonnetal/train/tasks/semantic'
alias g='grep'
alias seg='cd /homes/xili/cv0_home/AutoSeg/pointseg_1'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/homes/xili/cv0_home/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/homes/xili/cv0_home/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/homes/xili/cv0_home/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/homes/xili/cv0_home/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source activate py36