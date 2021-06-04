from torchvision import transforms
from model.metric import get_iou_score
from model.model import UNet
from model.config import START_FRAME
import numpy as np
import torch

from PIL import Image
import matplotlib.pyplot as plt
from utils.dataset import SpeechDataset, get_dataloader, scaled_in, scaled_ou

# path = './data/noisy_voice_amp_db.npy'

# images = np.load(path, allow_pickle=True)

# image = Image.fromarray(images[1])

# # image = images[1]
# # image = scaled_in(image)
# print(image)


# plt.imshow(image, cmap='jet', vmin=-1, vmax=1, origin='lower', aspect='auto')
# plt.axis('off')
# # plt.imsave("sample.jpg", image, cmap='jet', origin='lower')
# plt.show()

# # print(image)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

# dataset = SpeechDataset('./data/')
# print(len(dataset))
# trainloader, validloader = get_dataloader(dataset=dataset, batch_size=1)


# model = UNet(start_fm=START_FRAME).to(device)

# image, mak, pred = 0, 0, 0

# for i, (input, mask) in enumerate(validloader):
    # input, mask = input.to(device, dtype=torch.float), mask.to(device, dtype=torch.float)

image = np.load('./data/noisy_voice_amp_db.npy')[1]
voice = np.load('./data/voice_amp_db.npy')[1]
noise = np.load('./data/noise_amp_db.npy')[1]

gt_noise = image-voice


trans = transforms.ToTensor()
image = trans(scaled_in(image)).unsqueeze(0).to(device, dtype=torch.float)
gt    = trans(scaled_ou(gt_noise)).unsqueeze(0).to(device, dtype=torch.float)
noise = trans(scaled_in(noise)).unsqueeze(0).to(device, dtype=torch.float)

print('INPUT', image)
print('GROUND TRUTH', gt)
print('NOISE', noise)

# # predict = model(input)

print('IOU', get_iou_score(noise, gt))

# # pred  = torch.where(predict<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
# # pred  = pred.squeeze().detach().cpu().numpy()
image   = image.squeeze().detach().cpu().numpy()
gt_noise= gt.squeeze().detach().cpu().numpy()
noise   = noise.squeeze().detach().cpu().numpy()

f = plt.figure(figsize=(12, 4))

ax = f.add_subplot(131)
ax.imshow(image, cmap='jet', vmin=-1, vmax=1, origin='lower', aspect='auto')
ax.set_title("IMAGE")

ax1 = f.add_subplot(132)
ax1.imshow(gt_noise, cmap='jet', vmin=-1, vmax=1, origin='lower', aspect='auto')
ax1.set_title("GROUND TRUTH")

ax2 = f.add_subplot(133)
ax2.imshow(noise, cmap='jet', vmin=-1, vmax=1, origin='lower', aspect='auto')
ax2.set_title("NOISE")

plt.show()