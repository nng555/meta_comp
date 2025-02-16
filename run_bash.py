import os
import pathlib
import hydra
import subprocess
from omegaconf import DictConfig

# config path set automatically but can also be manually changed
CONFIG_PATH = pathlib.Path(__file__).parent.resolve()
os.environ['SSL_CERT_FILE']="/scratch/mr7401/projects/meta_comp/cacert.pem" # allows wandb to run in singularity
os.environ['PYTHONPATH']="/scratch/mr7401/projects/meta_comp/:$PYTHONPATH" 

@hydra.main(version_base=None, config_path='.', config_name="config")
def launch(cfg: DictConfig):
  
    hydra_args = cfg.args
  
    # Add hydra args as parameters to pass to script
    for key in hydra_args.keys():
        if hydra_args[key] is not None:
            cfg.command += f" --{key}={hydra_args[key]}"
        
    # Add the logging variables from config, if available
    if cfg.logging:
        cfg.command += f" --logger={cfg.logging.logger} --project={cfg.logging.project} --log_dir={cfg.logging.log_dir} --run_name={cfg.logging.run_name}"
    
    print(f"Run_Bash: run_bash.py is running the following command: \n {cfg.command}")
    subprocess.run(cfg.command, shell=True)

if __name__ == "__main__":
    launch()

