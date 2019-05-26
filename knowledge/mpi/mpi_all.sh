Sol 1)
mpirun -x LD_LIBRARY_PATH=$PGI_LIB -np 4 ./executable

Sol 2)
export LD_LIBRARY_PATH=$PGI_LIB:$LD_LIBRARY_PATH
mpirun_rsh -export -n 4 -hostfile $PBS_NODEFILE ./executable
