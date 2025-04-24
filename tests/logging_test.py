# import wandb
import random
from logger import Logger, add_log_args
import argparse
def simple_log_test(logger):
    
    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        
        # log metrics 
        logger.log({"acc": acc, "loss": loss})

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="WandB Test Script")
    parser = add_log_args(parser)
    args = parser.parse_args()

    # Set up logger
    logger = Logger(**vars(args))
    simple_log_test(logger = logger)
    logger.finish()
