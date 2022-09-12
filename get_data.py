import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from addict import Dict
from PIL import Image
import os

def get_data(train_config, validation_config):

  class BirdsDataset(Dataset):
      def __init__(self, path, augmentations=None, shape=(224, 224)):
          self.__augmentations = augmentations
          self.__shape = shape
          self.__images_labels = []
          self.labels = []

          if os.path.exists(path):
              self.__path = path
              self.__labels = os.listdir(self.__path)
              for label in self.__labels:
                  label_path = os.path.join(self.__path, label)
                  if os.path.isdir(label_path):
                      images = os.listdir(label_path)

                      for image in images:
                          if image.endswith("png") or image.endswith("jpg"):
                              image_path = os.path.join(label_path, image)

                              info = (image_path, label)
                              self.__images_labels.append(info)

                      self.labels.append(label)

                  else:
                      pass

          else:
              raise Exception(f"Path '{path}' for DataFrame doesn't exist!")


      def __getitem__(self, index):
          image_path, label = self.__images_labels[index]

          image = self.__load_image(image_path)

          if self.__augmentations is not None:
              image = self.__augmentations(image=image.permute(1, 2, 0).numpy())["image"]

          label = self.labels.index(label)
          return Dict({
              "image": image,
              "label": label,
          })


      def __len__(self):
          return len(self.__images_labels)


      def __load_image(self, path, channels="RGB"):
          width, height = self.__shape

          loader = A.Compose([
              A.Resize(width, height),
              ToTensor()
          ])

          image_array = np.array(Image.open(path).convert(channels))
          return loader(image=image_array)["image"]


  train_dataset = BirdsDataset(path=train_config["path"],
                               augmentations=train_config["augmentations"],
                               shape=train_config["shape"])


  validation_dataset = BirdsDataset(path=validation_config["path"],
                                    augmentations=validation_config["augmentations"],
                                    shape=validation_config["shape"])

  print(f"Train Size: {len(train_dataset)}")
  print(f"Validation Size: {len(validation_dataset)}")
  print(f"Num Classes: {len(train_dataset.labels)}")

  def collate_fn(batch):
      images, labels = [], []
      for item in batch:
          image = item["image"].tolist()
          images.append(image)

          label = item["label"]
          labels.append(label)

      return Dict({
          "images": torch.tensor(images),
          "labels": torch.tensor(labels)
      })



  train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            num_workers=train_config["num_workers"],
                            pin_memory=True,
                            batch_size=train_config["batch_size"],
                            collate_fn=collate_fn)


  validation_loader = DataLoader(validation_dataset,
                            shuffle=True,
                            num_workers=validation_config["num_workers"],
                            pin_memory=True,
                            batch_size=validation_config["batch_size"],
                            collate_fn=collate_fn)
  
  return train_loader, validation_loader
