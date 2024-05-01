import json
import os
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import numpy as np
import torch
import torch.multiprocessing as mp

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import models


def main(config_path):
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    from tta_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    if configs["distributed"] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train()


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """

    return models.__dict__[configs["arch"]]


def get_dataset(configs):
    from utils.fer2013dataset import fer2013
    train_set = fer2013("train", configs)
    val_set = fer2013("val", configs)
    test_set = fer2013("test", configs, tta=True, tta_size=10)
    return train_set, val_set, test_set


if __name__ == "__main__":
    main("C:/Users/menon/OneDrive/University/Code/Pytorch/Facial Emotion Recognition/utils/fer2013_config.json")