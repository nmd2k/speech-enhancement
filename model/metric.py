from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def get_iou_score(outputs, labels):
    lab = torch.where(labels<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    A = lab.squeeze(1).bool()
    pred = torch.where(outputs<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    B = pred.squeeze(1).bool()
    intersection = (A & B).float().sum((1,2))
    union = (A| B).float().sum((1, 2)) 
    iou = (intersection + 1e-6) / (union + 1e-6)  
    
    return iou.cpu().detach().numpy()