from slurm_lightning.model import LightningCifarClassifier
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from jsonargparse import lazy_instance

def main():
   cli = LightningCLI(LightningCifarClassifier,
                      trainer_defaults={
                        'precision':"16-true",
                        'profiler':"simple",
                        'logger': lazy_instance(WandbLogger,project="dsai544", log_model=True), 
                     })

if __name__ == "__main__":
   main()