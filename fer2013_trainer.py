import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.metrics import accuracy
from utils.radam import RAdam

EMO_DICT = {0: "neutral", 1: "angry", 2: "disgust", 3: "fear", 4: "happy", 5: "sad", 6: "surprise"}

class Trainer(object):
    def __init__(self):
        pass

class FER2013Trainer(Trainer):
    def __init__(self,model,train_set,val_set,test_set,configs):
        super().__init__()
        print("Starting Trainer..")
        print(configs)

        #loading the config from the json file
        self._configs = configs
        self._lr = self._configs["lr"]
        self._batch_size = self._configs["batch_size"]
        self._momentum = self._configs["momentum"]
        self._weight_decay = self._configs["weight_decay"]
        self._distributed = self._configs["distributed"]
        self._num_workers = self._configs["num_workers"]
        self._device = torch.device(self._configs["device"])
        self._max_epoch_num = self._configs["max_epoch_num"]
        self._max_plateau_count = self._configs["max_plateau_count"]

        #loading dataloader + model
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set
        self._model = model(
            in_channels=configs["in_channels"],
            num_classes=configs["num_classes"],
        )
        self._model.fc = nn.Linear(512, 7)
        self._model = self._model.to(self._device)

        if self._distributed == 1:
        #nvidia collective communications library for high performance gpu go whoosh
            #enabling training over both my gpus
            torch.distributed.init_process_group(backend="nccl")
            self._model = nn.parallel.DistributedDataParallel(
                self._model, find_unused_parameters=True
            )
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=lambda x: np.random.seed(x),
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )

            self._test_loader = DataLoader(
                self._test_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
        else:
            #without the nccl 
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=True,
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )
            self._test_loader = DataLoader(
                self._test_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )

        #critereon and optimizer
        class_weights = torch.FloatTensor(np.array([1.02660468,9.40661861,1.00104606,
                                                    0.56843877,0.84912748,1.29337298,
                                                    0.82603942,]))
        
        #Critereon -> Cross Entropy
        if self._configs["weighted_loss"] == 0:
            self._criterion = nn.CrossEntropyLoss().to(self._device)
        else:
            self._criterion = nn.CrossEntropyLoss(class_weights).to(self._device)
        
        #Optimizer -> Adam
        self._optimizer = RAdam(
            params=self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        #keeps track of current metric loss and the epoch count till last improvement
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            patience=self._configs["plateau_patience"],
            min_lr=1e-6,
            verbose=True,
        )

        #training stuff
        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        log_dir = os.path.join(
            self._configs["cwd"],
            self._configs["log_dir"],
            "{}_{}".format(
                self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
            ),
        )
        #keeping a directory of the progress
        self._writer = SummaryWriter(log_dir)
        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []
        self._best_loss = 1e9
        self._best_acc = 0
        self._test_acc = 0.0
        self._plateau_count = 0
        self._current_epoch_num = 0
        #setting checkpoints
        self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self._checkpoint_path = os.path.join(
            self._checkpoint_dir,
            "{}_{}".format(
                self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
            ),
        )
    
    def _train(self):
        self._model.train()
        train_loss = 0.0
        train_acc = 0.0

        for i, (images, targets) in tqdm(
            enumerate(self._train_loader), total=len(self._train_loader), leave=False
        ):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # get -> output, accuracy and loss
            outputs = self._model(images)
            loss = self._criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]

            train_loss += loss.item()
            train_acc += acc.item()

            # get gradient and do a SGD step to clear optimizer
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        i += 1
        self._train_loss.append(train_loss / i)
        self._train_acc.append(train_acc / i)

        #risidualmaskingnetwork/trainers/_fer2013_trainer.py
        #line 211
