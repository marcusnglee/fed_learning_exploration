from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# TODO: what does this mean lol
fds = None # Cache FederatedDataset 
BATCH_SIZE = 32

# copied from our classical setup
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Classification layers
        # After 2 pooling ops: 28x28 -> 14x14 -> 7x7, with 64 channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digit classes
    
    def forward(self, x):
        # BATCH x 1 x 28 x 28
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # BATCH x 32 x 14 x 14
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # BATCH x 64 x 7 x 7

        # now, flatten tensor for the linear layers
        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

""" Load partition MNIST data """
def load_data(partition_id: int, num_partitions: int):
    global fds

    if fds is None:
    # make num_partition amount of nodes
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist", 
            partitioners={"train": partitioner}
        )
    partition = fds.load_partition(partition_id=partition_id)
    
    # each node will 80/20 train test split
    partition_train_test = partition.train_test_split(test_size=0.2, seed=234)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # need to apply transformations on each PIL image wrapped inside the dataset obj
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    partition_train_test = partition_train_test.with_transform(transform=apply_transforms)
    trainloader = DataLoader(dataset=partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(dataset=partition_train_test['test'], batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, testloader

""" Train the model on the training set. """
def train(net, trainloader, valloader, epochs, device):
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
    train_loss, train_acc = test(net, trainloader, device)
    val_loss, val_acc = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


""" Validate model on the test set """
def test(net, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = net(images).to(device)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# transmisison of weights as numpy cpu arrays, as opposed to pytorch tensors
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# works on nvidia gpu, mac, or cpu only devices
def get_device():                                                                                                          
      if torch.cuda.is_available():                                                                                          
          return torch.device("cuda")                                                                                        
      elif torch.backends.mps.is_available():                                                                                
          return torch.device("mps")                                                                                         
      return torch.device("cpu")     