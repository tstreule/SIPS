#!/bin/bash

# Script to run a Weights & Biases Sweep on ETH Euler cluster.
#
#
# Run this script on ETH Euler using the following command:
# $ sbatch run_sweep.sh
#
# NOTE: Do not forget to first update the wandb agent option (see "submit job")
#

#SBATCH --ntasks-per-node=1                      # number of cores/threads
#SBATCH --cpus-per-task=1                        # number of cpus
#SBATCH --mem-per-cpu=4000                       # cpu memory size (exact)
#SBATCH --gpus-per-node=2                        # number of gpus
#SBATCH --gres=gpumem:10g                        # gpu memory size (or higher)
#SBATCH --time=1:00:00                           # wall-clock time (h:mm:ss)

#SBATCH --output="./euler_out/sweep_logs_agent2.txt"    # logging file
#SBATCH --open-mode=truncate                     # clear logging file before writing

# Load modules (required in order to use GPU/CUDA)
module load eth_proxy                            # allow internet access
module load gcc/8.2.0                            # required for python_gpu/3.11.2
module load python_gpu/3.11.2                    # load python modules with cuda enabled
module list

# Add module to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(realpath sips)

# Submit job
WANDB__SERVICE_WAIT=300 wandb agent kp2dsonar/sips/g0gbhsnr

# Exit
echo -e "\n\nINFO: Exiting bash script..."
exit 0
