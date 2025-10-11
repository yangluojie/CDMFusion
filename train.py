import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
from utils import datasets
from models import Diffusion_Model


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--config", type=str, default='config.yml',
                        help="Path to config file")
    parser.add_argument("--model_path", type=str, default='/path/results_MSRS/',
                        help="The folder to save training models")
    parser.add_argument("--image_folder", default='/path/results_MSRS/', type=str,
                        help="Location to save validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.Fusion_Data(config)

    print("=> creating denoising-diffusion model...")
    diffusion = Diffusion_Model(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
