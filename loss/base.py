import torch
import torch.nn as nn

class Distillation_Loss(nn.Module):
    def __init__(self, T, alpha):
        self.T = T
        self.alpha = alpha
    
    def __call__(self, y, teacher_scores, labels):
        return self.alpha * nn.KLDivLoss()(nn.functional.log_softmax(y / self.T, dim=1), nn.functional.softmax(teacher_scores / self.T, dim=1)) + (1. - self.alpha) * nn.functional.cross_entropy(y, labels)
    
    def __repr__(self):
        return f"Distillation Loss (T={self.T}, alpha={self.alpha})"
    
    def __str__(self):
        return self.__repr__()
    