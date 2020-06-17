import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class Trainer:

    def __init__(self, model: nn.Module, **kwargs):
        """
        Object initialization.
        """
        # Store the given instances.
        self.model = model
        self.criterion = kwargs.get('criterion', nn.CrossEntropyLoss())
        self.optimizer = kwargs.get('optimizer', optim.Adam(self.model.parameters()))

        # User settable attributes.
        self.n_epoch = kwargs.get('n_epoch', 1)  # The total number of epochs.
        self.train_batch_size = kwargs.get('train_batch_size', 256)  # The total number of batches in train.
        self.valid_batch_size = kwargs.get('valid_batch_size', 256)  # The total number of batches in valid.

        # Initialize the internal attributes.
        self.epoch = 0  # The number of the current epoch.
        self.batch = 0  # The number of the current batch.
        self.n_train_batch = 0  # The number of batches of train data.
        self.n_valid_batch = 0  # The number of batches of valid data.

    def __call__(self, train, valid):
        """
        Main loop of the network training.
        """
        # Define data loaders.
        train_loader = DataLoader(train, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid, batch_size=self.valid_batch_size, shuffle=True, pin_memory=True)

        # Save the total number of batches.
        self.n_train_batch = len(train_loader)
        self.n_valid_batch = len(valid_loader)

        # Iterate over epochs.
        for epoch in self.on_epoch():

            # Train phase
            self.on_train_begin()
            for inputs, targets in self.on_batch(train_loader):
                outputs, loss = self.on_train(inputs, targets)
                self.on_train_eval(inputs, targets, outputs, loss)
            self.on_train_end()

            # Valid phase
            self.on_valid_begin()
            with torch.no_grad():
                for inputs, targets in self.on_batch(valid_loader):
                    outputs, loss = self.on_valid(inputs, targets)
                    self.on_valid_eval(inputs, targets, outputs, loss)
            self.on_valid_end()

    def on_epoch(self):
        """
        An epoch generator that provides a loop over the epochs.
        """
        self.on_epoch_begin()
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            yield epoch
        self.on_epoch_end()

    def on_batch(self, loader: DataLoader):
        """
        A batch generator that provides a loop over the mini batches in the given data.
        """
        self.on_batch_begin()
        for batch, (inputs, targets) in enumerate(loader):
            self.batch = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            yield inputs, targets
        self.on_batch_end()

    def on_train(self, inputs, targets):
        """
        Train step on the given batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return outputs, loss

    def on_valid(self, inputs, targets):
        """
        Valid step on the given batch.
        """
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return outputs, loss

    def on_epoch_begin(self):
        """
        An event handler for later.
        """

    def on_epoch_end(self):
        """
        An event handler for later.
        """

    def on_batch_begin(self):
        """
        An event handler for later.
        """

    def on_batch_end(self):
        """
        An event handler for later.
        """

    def on_train_begin(self):
        """
        An event handler for later.
        """
        self.model.train()

    def on_train_end(self):
        """
        An event handler for later.
        """

    def on_valid_begin(self):
        """
        An event handler for later.
        """
        self.model.eval()

    def on_valid_end(self):
        """
        An event handler for later.
        """

    def on_train_eval(self, inputs, targets, outputs, loss):
        """
        An event handler for later.
        """

    def on_valid_eval(self, inputs, targets, outputs, loss):
        """
        An event handler for later.
        """


class MyTrainer(Trainer):
    """
    Customized trainer class that shows the training progress during process.
    """
    def on_train_begin(self):
        super(MyTrainer, self).on_train_begin()
        print(f'===== Train Phase: {self.epoch} / {self.n_epoch} =====')

    def on_train_end(self):
        super(MyTrainer, self).on_train_end()
        print()

    def on_train_eval(self, inputs, targets, outputs, loss):
        super(MyTrainer, self).on_train_eval(inputs, targets, outputs, loss)
        print(f'\r {self.batch} / {self.n_train_batch}: {loss.item():.3f}', end='')

    def on_valid_begin(self):
        super(MyTrainer, self).on_valid_begin()
        print(f'===== Valid Phase: {self.epoch} / {self.n_epoch} =====')

    def on_valid_end(self):
        super(MyTrainer, self).on_valid_end()
        print()

    def on_valid_eval(self, inputs, targets, outputs, loss):
        super(MyTrainer, self).on_valid_eval(inputs, targets, outputs, loss)
        print(f'\r {self.batch} / {self.n_valid_batch}: {loss.item():.3f}', end='')


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    valid = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    model = torchvision.models.resnet18(num_classes=10).cuda()

    trainer = MyTrainer(model, train_batch_size=32, valid_batch_size=32)

    trainer(train, valid)
