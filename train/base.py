import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from data import CSVDataset, FolderDataset
import numpy as np

class BaseTrain:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, pretrained_model_path=None, lr_scheduler=None, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.pretrained_model_path = pretrained_model_path
        if pretrained_model_path is not None:
            print("Loading pretrained model...")
            self.load(pretrained_model_path)
        else:
            print("Training from scratch...")

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def get_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class BasicTraining(BaseTrain):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, pretrained_model_path=None, lr_scheduler=None, num_epochs=10):
        super(BasicTraining, self).__init__(model, train_loader, val_loader, optimizer, criterion, device, lr_scheduler, num_epochs)

    def train(self):
        print("Training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):

                data = torch.tensor(data).to(self.device)
                targets = torch.tensor(targets).to(self.device).unsqueeze(0)

                # forward
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # check accuracy on training & validation set
            train_acc = self.check_accuracy(self.train_loader, self.model)
            val_acc = self.check_accuracy(self.val_loader, self.model)
            print(f"Epoch [{epoch}/{self.num_epochs}] Train accuracy: {train_acc:.2f}, Val accuracy: {val_acc:.2f}")
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_acc = self.check_accuracy(self.val_loader, self.model)
            print(f"Val accuracy: {val_acc:.2f}")
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def check_accuracy(self, loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)
                y = np.argmax(y).unsqueeze(0)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
            )

            return float(num_correct)/float(num_samples)*100
    
    def get_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
    def get_predictions(self, loader):
        self.model.eval()
        predictions = torch.tensor([])
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                scores = self.model(x)
                _, preds = scores.max(1)
                predictions = torch.cat((predictions, preds), dim=0)
        return predictions