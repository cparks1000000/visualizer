from __future__ import annotations

import argparse
from typing import List, Dict

import torch
from torch import Tensor, nn, argmax
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader


def weights_init_normal(model: nn.Module):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.constant_(model.bias, 0.0)


class ClassifierNetwork:

    # noinspection PyMethodMayBeStatic,PyMissingTypeHints
    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--decay_epoch', type=int, default=100,
                            help='epoch to start linearly decaying the learning rate to 0')
        return parser.parse_args()

    def __init__(self, channels_in: int, height: int, width: int, number_of_classes: int, device: str, training_data: Dataset, testing_data: Dataset, batch_size: int, num_epochs: int):
        super().__init__()

        self._device = device
        self._num_epochs = num_epochs


        if torch.cuda.is_available() and device == 'cpu':
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self._net: Classifier = Classifier(
                channels_in, height, width, number_of_classes
        )
        self._net.set_requires_grad(True)
        self._net.to(device)
        self._net.apply(weights_init_normal)
        self._loss: nn.Module = nn.CrossEntropyLoss()
        self._optimizer: Optimizer = torch.optim.SGD(self._net.parameters(), 0.005)
        # self._scheduler: LambdaLR = opt.scheduler(self._optimizer)
        self._training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        # noinspection PyTypeChecker
        self._testing_loader = DataLoader(testing_data, batch_size=1)
        #self._logger = Logger("classifier", len(self._training_loader))

    def do_training(self, save: bool = True) -> Classifier:
        for epoch in range(1, self._num_epochs+1):
            self._do_epoch(epoch)
        if save:
            self.save()
        return self._net

    def _do_epoch(self, epoch_number: int) -> None:
        for batch_number, batch in enumerate(self._training_loader, 1):
            self._do_batch(epoch_number, batch_number, batch)
        #self._scheduler.step()

    def _do_batch(self, epoch_number: int, batch_number: int, batch: List[Tensor]) -> None:

        self._optimizer.zero_grad()
        input_images: Tensor = batch[0].to(self._device)
        input_labels: Tensor = batch[1].to(self._device)
        classifications: Tensor = self._net(input_images)
        loss: Tensor = self._loss(classifications, input_labels)
        loss.backward()
        self._optimizer.step()

        if batch_number % 10 == 0:
            print("Classifier loss during batch", batch_number, "of epoch", epoch_number, "is", loss.item())
        #self.logger.plot_losses()

    @staticmethod
    def _count_incorrect(classifications: Tensor, labels: Tensor) -> int:
        count = 0
        for classification, label in zip(classifications, labels):
            if argmax(classification) != label:
                count += 1
        return count

    def do_testing(self) -> int:
        data = iter(self._testing_loader).__next__()
        input_images: Tensor = data[0].to(self._device)
        input_labels: Tensor = data[1].to(self._device)
        classifications: Tensor = self._net(input_images)
        return self._count_incorrect(classifications, input_labels)

    def save(self) -> None:
        self._net.save()