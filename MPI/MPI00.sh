#MPI00.sh
#!/bin/sh
#PBS -N MPI00
#PBS -l nodes=node1_vir1

pssh -h $PBS_NODEFILE mkdir -p /home/s2212108/bx/MPI 1>&2
scp master:/home/s2212108/bx/MPI/MPI00 /home/s2212108/bx/MPI
pscp -h $PBS_NODEFILE master:/home/s2212108/bx/MPI/MPI00 /home/s2212108/bx/MPI 1>&2
/home/s2212108/bx/MPI/MPI00
