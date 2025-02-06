import os
import pathlib
import hydra
import subprocess
import logging 
from omegaconf import DictConfig

# config path set automatically but can also be manually changed
CONFIG_PATH = pathlib.Path(__file__).parent.resolve()
print(CONFIG_PATH)
os.environ['SSL_CERT_FILE']="/scratch/mr7401/projects/meta_comp/cacert.pem" # allows wandb to run in singularity
os.environ['PYTHONPATH']="/scratch/mr7401/projects/meta_comp/:$PYTHONPATH" 

@hydra.main(version_base=None, config_path='.', config_name="config")
def launch(cfg: DictConfig):
    # Add the logging variables from config 
    cfg.command += f" --logger={cfg.logger} --project={cfg.project} --log_dir={cfg.log_dir} --run_name={cfg.run_name}"
    print(cfg.command)
    subprocess.run(cfg.command, shell=True)

if __name__ == "__main__":
    launch()

