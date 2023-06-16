import torch as tor
import numpy as np
from torch import nn
import inspect


def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)

    return wrapper


def make_vec_single(v, length, dtype=tor.float64):
    return tor.tensor(v, dtype=dtype).repeat(length)


class HyperPrameters:
    def save_hyperparameters(self, ignore=[]):
        """
        This function saves the arguments of the last frame as the attributes of this instance
        """
        frame = inspect.currentframe().f_back  # access the frame of last function call
        _, _, _, local_vars = inspect.getargvalues(frame)
        for k, v in local_vars.items():
            if k not in ignore:
                setattr(self, k, v)


class ProgressBoard(HyperPrameters):
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
    ):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented


class Module(nn.Module, HyperPrameters):
    def __init__(self, plot_train_per_epoch=2, plot_vlaid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def __repr__(self):
        return "I am a Module object"

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net")
        return self.net(X)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1], batch[-1]))
        return l

    def configure_optimizer(self):
        raise NotImplementedError


class DataModule(HyperPrameters):
    def __init__(self, root="./data", num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class Trainer(HyperPrameters):
    def __init__(self, max_epochs: int, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.train_loss = []
        assert num_gpus == 0

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model: Module):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model: Module, data: DataModule):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
