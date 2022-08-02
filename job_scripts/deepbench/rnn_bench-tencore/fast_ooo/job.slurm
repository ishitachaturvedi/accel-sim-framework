#!/bin/bash
#SBATCH --job-name=rnn-ten-energy     # create a short name for your job
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=130:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=ishitac@princeton.edu

export SINGULARITYENV_LD_LIBRARY_PATH=/mylibs
nohup singularity exec -B /scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/gpu-simulator/gpgpu-sim/lib/gcc-7.5.0/cuda-10010/release:/mylibs /home/ishitac/tigress/gpgpu_sim.img bash run_script.sh > out
