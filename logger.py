import os
import wandb
import logging
import warning

def add_log_args(parser):
    parser.add_argument("--logger", type=str, choices=["local", "wandb", "both", None], default=None, help="Log metrics to local only, wandb, or both")
    parser.add_argument("--project", type=str, default = "test", help="project name for logging")
    parser.add_argument("--log_dir", type=str, default = ".", help="Directory to save logs")
    parser.add_argument("--run_name", type=str, default = ".", help="name of the run (category)")
    return parser

class Logger:
    def __init__(self, logger="local", project="test", log_dir = ".", run_name = "test", **kwargs): 
        self.project = project
        
        # Set up WandB
        if logger == "wandb" or logger == "both":
            print("Setting up WandB")
            self.log_to_wandb = True 
            os.environ["WANDB_DIR"] = log_dir
            wandb.init(project=project)
            self.wandb_id = wandb.run.id
        else: 
            self.log_to_wandb = False
        
        # Set up local 
        if logger == "local" or logger == "both":
            print("Setting up local logging")
            self.log_locally = True
            self.local_logger = logging.getLogger(project)

            pid = os.getpid()  # Get process ID
            self.run_id = self.wandb_id if self.log_to_wandb else pid

            os.makedirs(f"{log_dir}/outputs/{run_name}/{self.run_id}", exist_ok = True)
            logging.basicConfig(filename=f"{log_dir}/outputs/{run_name}/{self.run_id}/local.log", level=logging.DEBUG)
        else:
            self.log_locally = False
        
        if not self.log_to_wandb and not self.log_locally:
            warning.warn("\n\n\n\n WARNING: No logging mode selected. To log results, select one of 'wandb', 'local', or 'both'. Only Hydra/slurm logging will be generated. \n\n\n\n")
    
    def log(self, metrics):
        """Logs to wandb or writes to a file, based on the mode."""     
        if self.log_to_wandb:
            wandb.log(metrics)
        if self.log_locally:
            self.local_logger.info(metrics)
    
    def finish(self):
        if self.log_to_wandb:
            wandb.finish()
        

