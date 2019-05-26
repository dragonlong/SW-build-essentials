salloc -n1 --nodelist=hu008 -t 24:00:00 --mem-per-cpu=32G --gres=gpu:pascal:4
srun -N2 -n1 -l /bin/hostname

