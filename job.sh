#!/bin/bash
#SBATCH -A oee001                # account / proje adi
#SBATCH -p a100q
#SBATCH -N 1             # This needs to match Trainer(num_nodes=...)
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

module load cuda/cuda-12.1-a100q
module load ANACONDA/Anaconda3-2023.03-python-3.9
source /ari/progs/ANACONDA/Anaconda3-2023.03-python-3.9/etc/profile.d/conda.sh
conda activate dsai544

#Girdi dosyalari kopyalaniyor
# \cp -r   /ari/users/mkocyigit/dataset /YEREL

# Calistirmak istedigimiz programa uygun parametrelerle /YEREL diskini kullanmasi soylenmeli
# Tabiki kullanacaginiz programin parametreleri degisik olacaktir
srun python slurm_lightning/train.py fit --trainer.max_epochs=40

#Program bitince olusan ve saklamak istediklerimizi geri ev dizinine kopyalamali
# \cp -r /YEREL/tb_logs /ari/users/mkocyigit/