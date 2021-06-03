import argparse

from torch.nn.modules import dropout
from utils.utils import normtensor, savenp2Img, tensor2np, wandb_mask

from model.metric import get_iou_score
import os
from model.config import BATCH_SIZE, DATA_PATH, DROP_RATE, EPOCHS, INPUT_SIZE, LEARNING_RATE, N_CLASSES, RUN_NAME, SAVE_PATH, START_FRAME

import torch
import wandb
import time
import numpy as np
from tqdm import tqdm
from torch import optim
from torch import nn
from model.model import UNet, UNet_ResNet
from utils.dataset import SpeechDataset, get_dataloader

import matplotlib.pyplot as plt

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--run', type=str, default='demo', help="run name")
    parser.add_argument('--model', type=str, default='Unet', help="initial weights path")
    parser.add_argument('--dropout', type=float, default=DROP_RATE, help="declear dropout rate")
    parser.add_argument('--epoch', type=int, default=EPOCHS, help="number of epoch")
    parser.add_argument('--startfm', type=int, default=START_FRAME, help="architecture start frame")
    parser.add_argument('--batchsize', type=int, default=BATCH_SIZE, help="total batch size for all GPUs (default:")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="learning rate (default: 0.0001)")
    parser.add_argument('--tuning', action='store_true', help="no plot image for tuning")
    # parser.add_argument('--size', type=int, default=INPUT_SIZE, help="input size (default: 128)")

    args = parser.parse_args()
    return args

def train(model, device, trainloader, optimizer, loss_function):
    model.train()
    running_loss = 0
    mask_list, iou = [], []
    for i, (input, mask) in enumerate(trainloader):
        # load data into cuda
        input, mask = input.to(device, dtype=torch.float), mask.to(device, dtype=torch.float)

        # forward
        predict = model(input)
        loss = loss_function(predict, mask)

        # metric
        iou.append(get_iou_score(predict, mask).mean())
        running_loss += (loss.item())
        
        # zero the gradient + backprpagation + step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # log the first image of the batch
        if ((i + 1) % 20) == 0 and not args.tuning:
            rand = np.random.randint(0, 6000)
            img, pred, mak = tensor2np(input[1]), tensor2np(predict[1]), tensor2np(mask[1])
            savenp2Img(SAVE_PATH+f'image_{rand}.jpg', img)
            savenp2Img(SAVE_PATH+f'prediction_{rand}.jpg', pred)
            savenp2Img(SAVE_PATH+f'mask_{rand}.jpg', mak)
            mask_list.extend([wandb.Image(SAVE_PATH+f'image_{rand}.jpg'),
                        wandb.Image(SAVE_PATH+f'mask_{rand}.jpg'),
                        wandb.Image(SAVE_PATH+f'prediction_{rand}.jpg'),
            ])
            
    mean_iou = np.mean(iou)
    total_loss = running_loss/len(trainloader)
    if not args.tuning:
        wandb.log({'Train loss': total_loss, 'Train IoU': mean_iou, 'Train prediction': mask_list})
    else: wandb.log({'Train loss': total_loss, 'Train IoU': mean_iou})

    return total_loss, mean_iou
    
def test(model, device, testloader, loss_function, best_iou):
    model.eval()
    running_loss = 0
    mask_list, iou  = [], []
    with torch.no_grad():
        for i, (input, mask) in enumerate(testloader):
            input, mask = input.to(device, dtype=torch.float), mask.to(device, dtype=torch.float)

            predict = model(input)
            loss = loss_function(predict, mask)

            running_loss += loss.item()
            iou.append(get_iou_score(predict, mask).mean())

            # log the first image of the batch
            if ((i + 1) % 10) == 0 and not args.tuning:
                rand = np.random.randint(0, 6000)
                img, pred, mak = tensor2np(input[1]), tensor2np(predict[1]), tensor2np(mask[1])
                savenp2Img(SAVE_PATH+f'image_{rand}.jpg', img)
                savenp2Img(SAVE_PATH+f'prediction_{rand}.jpg', pred)
                savenp2Img(SAVE_PATH+f'mask_{rand}.jpg', mak)
                mask_list.extend([wandb.Image(SAVE_PATH+f'image_{rand}.jpg'),
                            wandb.Image(SAVE_PATH+f'mask_{rand}.jpg'),
                            wandb.Image(SAVE_PATH+f'prediction_{rand}.jpg'),
                ])

    test_loss = running_loss/len(testloader)
    mean_iou = np.mean(iou)
    if not args.tuning:
        wandb.log({'Valid loss': test_loss, 'Valid IoU': mean_iou, 'Prediction': mask_list})
    else: wandb.log({'Valid loss': test_loss, 'Valid IoU': mean_iou})
    
    if mean_iou>best_iou:
    # export to onnx + pt
        try:
            torch.onnx.export(model, input, SAVE_PATH+RUN_NAME+'.onnx')
            torch.save(model.state_dict(), SAVE_PATH+RUN_NAME+'.pth')
        except:
            print('Can export weights')

    return test_loss, mean_iou

if __name__ == '__main__':
    args = parse_args()

    # init wandb
    config = dict(
        model       = args.model,
        dropout     = args.dropout,
        lr          = args.lr,
        batchsize   = args.batchsize,
        epoch       = args.epoch,
        startfm     = args.startfm,
        # size        = args.size
    )
    
    RUN_NAME = args.run
    # INPUT_SIZE = args.size

    run = wandb.init(project="Speech-enhancement", config=config)
    artifact = wandb.Artifact('Spectrograms', type='Dataset')

    # train on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    try:
        artifact.add_dir(DATA_PATH)
        run.log_artifact(artifact)
    except:
        artifact     = run.use_artifact('Spectrograms:latest', type='Dataset')
        artifact_dir = artifact.download(DATA_PATH)


    # load dataset
    dataset = SpeechDataset(DATA_PATH)
    trainloader, validloader = get_dataloader(dataset=dataset, batch_size=args.batchsize)

    # get model and define loss func, optimizer
    n_classes = N_CLASSES
    epochs = args.epoch

    tag = 'Unet'
    if args.model == 'Unet':
        model = UNet(start_fm=args.startfm).to(device)
    else:
        tag = 'UnetRes'
        model = UNet_ResNet(dropout=args.dropout, start_fm=args.startfm).to(device)

    run.tags = [tag]

    criterion = nn.SmoothL1Loss()

    # loss_func   = Weighted_Cross_Entropy_Loss()
    optimizer   = optim.Adam(model.parameters(), lr=args.lr)

    # wandb watch
    run.watch(models=model, criterion=criterion, log='all', log_freq=10)

    # training
    best_iou = -1

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_iou = train(model, device, trainloader, optimizer, criterion)
        t1 = time.time()
        print(f'Epoch: {epoch} | Train loss: {train_loss:.3f} | Train IoU: {train_iou:.3f} | Time: {(t1-t0):.1f}s')
        test_loss, test_iou = test(model, device, validloader, criterion, best_iou)
        print(f'Epoch: {epoch} | Valid loss: {test_loss:.3f} | Valid IoU: {test_iou:.3f} | Time: {(t1-t0):.1f}s')
        
        # Wandb summary
        if best_iou < test_iou:
            best_iou = test_iou
            wandb.run.summary["best_accuracy"] = best_iou
    
    if not args.tuning:
        trained_weight = wandb.Artifact(RUN_NAME, type='weights')
        trained_weight.add_file(SAVE_PATH+RUN_NAME+'.onnx')
        trained_weight.add_file(SAVE_PATH+RUN_NAME+'.pth')
        wandb.log_artifact(trained_weight)
    # evaluate