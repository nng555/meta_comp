import os
import pathlib
import hydra
import subprocess
from omegaconf import DictConfig

# config path set automatically but can also be manually changed
CONFIG_PATH = pathlib.Path(__file__).parent.resolve()
print(CONFIG_PATH)
os.environ['SSL_CERT_FILE']="/scratch/mr7401/projects/meta_comp/cacert.pem" # allows wandb to run in singularity
os.environ['PYTHONPATH']="/scratch/mr7401/projects/meta_comp/:$PYTHONPATH" 

@hydra.main(version_base=None, config_path='.', config_name="config")
def launch(cfg: DictConfig):
  
    #print("run_bash is running the following command:")
    hydra_args = cfg.args
    print(hydra_args)

    for key in hydra_args.keys():
        if hydra_args[key] is not None:
            print(key)
            print(hydra_args[key])
            cfg.command += f" --{key}={hydra_args[key]}"
        
    # Add the logging variables from config 
    cfg.command += f" --logger={cfg.logging.logger} --project={cfg.logging.project} --log_dir={cfg.logging.log_dir} --run_name={cfg.logging.run_name}"
    
    print(cfg.command)
    subprocess.run(cfg.command, shell=True)

if __name__ == "__main__":
    launch()

