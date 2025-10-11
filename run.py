# The code for infrared and visible image fusion
import argparse
import yaml
import torch
import os
from utils import datasets
from models import Diffusion_Model, Diffusion_Generation

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Image Fusion')
    parser.add_argument("--config", default="config.yml", type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='/path/ckpt_CDMFusion.pth.tar', type=str,
                        help='Checkpoint for evaluation')
    parser.add_argument("--image_folder", default='/path/results_test_MSRS/', type=str,
                        help="Location to save results")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device
    if torch.cuda.is_available():
        print('Currently supports evaluations when run only on a single GPU!')

    print("=> Now evaluating for dataset '{}'".format(config.data.dataset))
    DATASET = datasets.Fusion_Data(config)
    _, val_loader = DATASET.get_loaders(parse_patches=False)

    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = Diffusion_Model(args, config)
    model = Diffusion_Generation(diffusion, args, config)
    model.generation(val_loader)


if __name__ == '__main__':
    main()
