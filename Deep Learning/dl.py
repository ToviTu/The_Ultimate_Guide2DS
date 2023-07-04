import torch as tor
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
import inspect
import random

rcParams["figure.figsize"] = (6, 4)


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


class Module(nn.Module, HyperPrameters):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

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
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def configure_optimizers(self):
        assert self.net is not None
        return SGD(
            [
                item
                for module in self.net
                if type(module) in [nn.Conv2d, nn.Linear]
                for item in [module.weight, module.bias]
            ],
            self.eta,
        )


class DataModule(HyperPrameters):
    num_train = 0
    num_val = 0

    def __init__(self, root="./", num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train=True):
        if train:
            indices = list(range(0, self.num_train))
            random.shuffle(indices)
        else:
            indices = list(range(self.num_train, self.num_train + self.num_val))

        for i in range(0, len(indices), self.batch_size):
            batch_indices = tor.tensor(indices[i : i + self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.num_train // self.batch_size

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class Trainer(HyperPrameters):
    def __init__(self, max_epochs: int, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.train_loss = []
        self.val_loss = []
        assert num_gpus == 0

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model: Module):
        model.trainer = self
        self.model = model

    def prepare_batch(self, batch):
        return batch

    def fit(self, model: Module, data: DataModule):
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.prepare_data(data)
            self.fit_epoch()

    def plot_loss(self):
        plt.plot(range(len(self.train_loss)), self.train_loss)
        if self.val_loss:
            plt.plot(range(len(self.val_loss)), self.val_loss)

    def fit_epoch(self):
        batch_loss = []
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with tor.no_grad():
                loss.backward()
                self.optim.step()
            batch_loss.append(loss.item())
        self.train_loss.append(np.mean(batch_loss))

        if self.val_dataloader is None:
            return
        batch_loss = []
        for batch in self.val_dataloader:
            with tor.no_grad():
                batch_loss.append(
                    self.model.validation_step(self.prepare_batch(batch)).item()
                )
            self.val_batch_idx += 1
        self.val_loss.append(np.mean(batch_loss))


class SGD(HyperPrameters):
    def __init__(self, params, eta):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.eta * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
