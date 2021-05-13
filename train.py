from model.model import UNet
import torch


def main():
    model = UNet(572, 2)
    print(model)

if __name__ == "__main__":
    main()