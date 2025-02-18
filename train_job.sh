#!/bin/bash
#SBATCH --job-name=attgan_training       
#SBATCH --output=attgan_training.out     
#SBATCH --error=attgan_training.err     
#SBATCH --partition=gpu_prod_long            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00                 
#SBATCH --cpus-per-task=8                

# Load required modules (adjust as needed)
module load python/3.10  
module load cuda/11.8    

# Move to the project directory
cd /usr/users/siapartnerscomsportif/bohin_ant/att-gan/

# Install missing packages
pip install --user -r requirements.txt

# Run the training script
python3 train_github.py --gpu --experiment_name test_full_training --epochs 45