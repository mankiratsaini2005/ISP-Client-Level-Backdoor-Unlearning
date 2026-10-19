import torchvision.datasets.cifar as cifar
cifar.check_integrity = lambda path, md5=None: True
import torchvision
cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
print("Success!")
