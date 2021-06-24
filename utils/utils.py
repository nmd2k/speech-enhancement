from model.config import SAVE_PATH
import wandb
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def show_dataset(dataset, n_sample=4):
    pass
    # """Visualize dataset with n_sample"""
    # fig = plt.figure()

    # # show image
    # for i in range(n_sample):
    #     image, mask = dataset[i]
    #     image = transforms.ToPILImage()(image)
    #     mask = transforms.ToPILImage()(mask)
    #     print(i, image.size)

    #     plt.tight_layout()
    #     ax = plt.subplot(2, n_sample, i + 1)
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')

    #     plt.imshow(image, cmap="Greys")
    #     plt.imshow(mask, alpha=0.3, cmap="OrRd")

    #     if i == n_sample-1:
    #         plt.show()
    #         break
        
def show_image_mask(image, mask):
    pass
    # fig, ax = plt.subplots()

    # image = transforms.ToPILImage()(image)
    # mask = transforms.ToPILImage()(mask)

    # ax.imshow(image, cmap="Greys")
    # ax.imshow(mask, alpha=0.3, cmap="OrRd")

    # plt.show()

def labels():
  l = {}
  for i, label in enumerate(CLASSES):
    l[i] = label
  return l

def tensor2np(tensor):
    tensor = tensor.squeeze().cpu()
    return tensor.detach().numpy()

def savenp2Img(name, np_array):
    plt.imsave(name, np_array, cmap='jet', origin='lower')

def normtensor(tensor):
    tensor = torch.where(tensor<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    return tensor

def wandb_mask():
    images = [
        wandb.Image(SAVE_PATH+'image.jpg'),
        wandb.Image(SAVE_PATH+'prediction.jpg'),
        wandb.Image(SAVE_PATH+'mask.jpg'), 
    ]
    return images

# def wandb_mask(bg_imgs, pred_masks, true_masks):

#     return wandb.Image(bg_imgs, masks={
#         "predictions" : {
#             "mask_data" : pred_masks,
#             "class_labels" : CLASSES
#             },
#         "ground_truth" : {
#             "mask_data" : true_masks, 
#             "class_labels" : CLASSES
#             }
#         })