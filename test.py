import torch.utils.data
import time
import argparse
from trainer import run_training, test
import torch.nn as nn
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="fmnist")
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--arch', type=str, default="alex")
parser.add_argument('--sensitivity', type=float, default=0.3)
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and args.use_cuda
device = torch.device("cuda" if use_cuda else "cpu")
arch = "AlexNet" if args.arch == "alex" else "MobileNetV2"
PATH = 'models/' + args.data + '/' + arch +str(args.sensitivity) + "_pruned.pth"

if arch == "AlexNet":
    model = models.alexnet(pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features=n_inputs, out_features=10, bias=True)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ])
else:
    model = models.mobilenet_v2(pretrained=False)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=n_inputs, out_features=10)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
# load the saved model
model.load_state_dict(torch.load(PATH))

# in evaluation mode
model.eval()
test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
print("=" * 60)
print("TESTING")
print("=" * 60)
print("")
s = time.time()
test(model, device, test_loader, single_batch_test=True)
e = time.time()
print("Inference Time:  " + str(e - s))