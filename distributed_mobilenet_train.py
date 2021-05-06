import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from utils import get_rules, set_sparsity, get_num_weights
from pruner import Pruner
from trainer import test

def restore_model_structure(model):
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=n_inputs, out_features=10)


def sync_participants(participant, global_m):
    participant.load_state_dict(global_m.state_dict())


def weight_aggregate(global_model, participants):
    """
    This function has aggregation method 'mean'
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([participants[i].state_dict()[k].float() for i in range(len(participants))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    pruner.prune(model=global_model, stage=0, update_masks=True, verbose=False)
    for model in participants:
        model.load_state_dict(global_model.state_dict())


sparsity = 0.3
num_rounds = 100
num_participants = 5
PATH = 'models/' + 'fmnist' + '/' + 'MobileNetV2' + str(sparsity) + "_pruned.pth"

# load the mobilenet arch from imagenet
global_model = models.mobilenet_v2(pretrained=False)

# modify the arch for fmnist dataset
restore_model_structure(global_model)

participants = [models.mobilenet_v2(pretrained=False) for _ in range(num_participants)]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
# split the train data into partial iid evenly based on the # of participants
traindata_split = torch.utils.data.random_split(train_data, [int(train_data.data.shape[0] / num_participants) for _ in range(num_participants)])

train_loader = [torch.utils.data.DataLoader(x, batch_size=50, shuffle=True) for x in traindata_split]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False)
# initial stage of sync the weights from participants with the global model
for participant_model in participants:
    restore_model_structure(participant_model)
    sync_participants(participant_model, global_model)

criterion = nn.CrossEntropyLoss()
opt = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5) for model in participants]

set_sparsity(global_model, sparsity, 'MobileNetV2-D')
rule = get_rules("rules/" + 'MobileNetV2-D' + ".rule")
pruner = Pruner(rule=rule)


for e in range(5):
    participant_iterators = [iter(loader) for loader in train_loader]
    Loss = 0
    for batch_num in range(len(train_loader[0])):
        batch_loss = 0
        for i in range(len(participants)):
            participant_model = participants[i]
            participant_iterator = participant_iterators[i]
            participant_opt = opt[i]
            participant_opt.zero_grad()
            data, label = next(participant_iterator)
            predict = participant_model(data)
            loss = criterion(predict, label)
            loss.backward()
            batch_loss += loss
            participant_opt.step()
        weight_aggregate(global_model, participants)
        if batch_num % 20 == 0:
            print('\nTrain epoch {} at batch {} with loss {} [({:.0f}%)]'.format(
                e, batch_num, batch_loss, batch_num / len(train_loader[0]) * 100))
            remainder_param, r_total = get_num_weights(global_model, verbose=False)
    print('\nTrain epoch {} with loss {}'.format(
        e, Loss))
    test(global_model, 'cpu', test_loader, True)

torch.save(global_model.state_dict(), 'models/collaborated/' + 'fmnist_mobilenet' + '.pth')