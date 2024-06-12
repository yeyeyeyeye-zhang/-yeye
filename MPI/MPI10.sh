#MPI10.sh
#!/bin/sh
#PBS -N MPI10
#PBS -l nodes=node1_vir1+node1_vir2
pssh -h $PBS_NODEFILE mkdir -p /home/s2212108/bx/MPI 1>&2
scp master:/home/s2212108/bx/MPI/MPI10 /home/s2212108/bx/MPI
pscp -h $PBS_NODEFILE master:/home/s2212108/bx/MPI/MPI10 /home/s2212108/bx/MPI 1>&2
mpiexec -np 2 -machinefile $PBS_NODEFILE /home/s2212108/bx/MPI/MPI10
