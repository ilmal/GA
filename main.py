import albumentations as A
from albumentations.pytorch import ToTensor

from get_data import get_data
from train import train

train_config = {
  "path": "data/train",
  "shape": (224, 224),
  "batch_size": 16,
  "num_workers": 4,
  "augmentations": A.Compose([
      A.HorizontalFlip(p=0.5),
      ToTensor(),
  ])
}


validation_config = {
  "path": "data/valid",
  "shape": (224, 224),
  "batch_size": 32,
  "num_workers": 4,
  "augmentations": None,
}


def main():

  train_loader, validation_loader = get_data(train_config, validation_config)
  print(train_loader)
  train(train_loader, validation_loader)
  
  
def start_runtime():
  main()
  
if __name__ == "__main__":
  start_runtime()
