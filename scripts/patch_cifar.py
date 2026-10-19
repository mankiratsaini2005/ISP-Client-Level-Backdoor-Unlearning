import torchvision

print(torchvision.datasets.CIFAR100.train_list)
torchvision.datasets.CIFAR100.train_list = [["train", None]]
