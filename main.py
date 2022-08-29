from get_data import get_data

train_config = Dict({
  "path": "../input/100-bird-species/train",
  "shape": (224, 224),
  "batch_size": 16,
  "num_workers": 4,
  "augmentations": A.Compose([
      A.HorizontalFlip(p=0.5),
      ToTensor(),
  ])
})


validation_config = Dict({
  "path": "../input/100-bird-species/valid",
  "shape": (224, 224),
  "batch_size": 32,
  "num_workers": 4,
  "augmentations": None,
})


def main():

  get_data(train_config, validation_config)
  
  
  
def start_runtime():
  main()
  
if __name__ == "__main__":
  start_runtime()
