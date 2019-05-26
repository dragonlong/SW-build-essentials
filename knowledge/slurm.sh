salloc -N1 --nodelist=dt029 --partition=normal_q --account=arctest --mem-per-cpu=100MB
srun -n2 --mpi=pmi2 /home/lxiaol9/jobs/osu-micro-benchmarks-4.4.1/mpi/pt2pt/osu_latency

#!/bin/bash
#SBATCH -J hello-world
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -t 10:00
echo "hello world"
sbatch hello.sh

salloc -N1 -t 10:00
#
https://www.vanderbilt.edu/accre/documentation/slurm/
