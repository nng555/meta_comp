import argparse
from logger import add_log_args

def main(model_name, **kwargs):
    print(f"Running Script with {model_name}")

if __name__ == "__main__":
    # print("Arg Test Script Running")
    # # Parse arguments
    parser = argparse.ArgumentParser(description="WandB Test Script")
    parser.add_argument('--model_name', type=str, help='Name of the model to use for generation')
    parser = add_log_args(parser)
    known_args, unknown_args = parser.parse_known_args()
    
    main(**vars(known_args))

    

    