import torch
import torch.nn as nn
from .base import BaseTrain
from tqdm import tqdm
import os

class DistillationTraining(BaseTrain):
    def __init__(self, model, teacher_model, train_loader, val_loader, optimizer, criterion, device, pretrained_model_path=None, lr_scheduler=None, num_epochs=10):
        super(DistillationTraining, self).__init__(model, train_loader, val_loader, optimizer, criterion, device, lr_scheduler, num_epochs)
        self.teacher_model = teacher_model
    
    def train(self):
        self.model.train()
        self.teacher_model.eval()
        for epoch in range(self.num_epochs):
            for batch_idx, (input, label) in tqdm(enumerate(self.train_loader)):
                input = torch.tensor(input).to(self.device)
                label = torch.tensor(label).to(self.device)
                output = self.model(input)
                with torch.no_grad():
                    teacher_output = self.teacher_model(input)
                loss = self.criterion(output, teacher_output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # validate
            val_acc = self.validate()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Validation Accuracy: {val_acc:.2f}%")

            # save model
            self.save(f"checkpoints/distillation_{epoch}.pth")
            
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for input, label in self.val_loader:
                input = torch.tensor(input).to(self.device)
                label = torch.tensor(label).to(self.device)
                output = self.model(input)
                _, preds = torch.max(output, 1)
                val_acc = torch.tensor(torch.sum(preds == label).item() / len(preds))
        return val_acc
    
    def save(self, path):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def get_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))