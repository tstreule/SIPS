#!/bin/bash

# Script to run the ORB and SIFT baseline evaluation on ETH Euler cluster. 
# Run this script on ETH Euler using the following command:
# $ sbatch run_baseline_eval.sh

#SBATCH --ntasks-per-node=4                      # number of cores/threads
# !!! keep num cpus in sync with config.datasets.num_workers
#SBATCH --cpus-per-task=4                        # number of cpus
#SBATCH --mem-per-cpu=1000                       # cpu memory size (exact)
# !!! every gpu will use batch size config.datasets.batch_size
#SBATCH --gpus-per-node=2                        # number of gpus
#SBATCH --gres=gpumem:20g                        # gpu memory size (or higher)
#SBATCH --time=12:00:00                          # wall-clock time (h:mm:ss)

#SBATCH --job-name=SIFT_100_3                    # job name
#SBATCH --output="./euler_out/SIFT_100_3.txt"    # logging file
#SBATCH --open-mode=truncate                     # clear logging file before writing

# Load modules (required in order to use GPU/CUDA)
module load eth_proxy                            # allow internet access
module load gcc/8.2.0                            # required for python_gpu/3.11.2
module load python_gpu/3.11.2                    # load python modules with cuda enabled
module list

# Add module to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(realpath sips)
#echo "HI"
python3 sips/baselines/baseline_keypoints.py --nfeatures 100 --model_name "SIFT" --matching_threshold 3

# Exit
echo -e "\n\nINFO: Exiting bash script..."
exit 0
