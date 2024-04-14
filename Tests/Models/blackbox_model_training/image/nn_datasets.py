from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PlainDataset(Dataset):
    def __init__(self, data, labels, transforms= None):
        super(PlainDataset, self).__init__()
        self.labels = labels
        self.data = data
        self.transforms = transforms
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input, label = self.data[index], self.labels[index]
        if not self.transforms is None:
            input = self.transforms(input)
        return input, label

def load_mnist(args, train_shuffle = True, flatten = True):
        # Define Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
    }


    # Training Data
    train_set = MNIST(
        root="./Files/Data", train=True, download=True, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=0)

    # Test Data
    test_set = MNIST(
        root="./Files/Data", train=False, download=True, transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # extract data
    tr_data_im, tr_data_lab = extract_data_labels(train_loader)
    te_data_im, te_data_lab = extract_data_labels(train_loader)

    if flatten:
        tr_data_im = torch.flatten(tr_data_im,1)
        te_data_im = torch.flatten(te_data_im,1)

    return tr_data_im, tr_data_lab, None, train_loader, None, None, None, None, te_data_im, te_data_lab, None, test_loader

def extract_data_labels(dataloader):
    x, y = [],[]
    for i, data in enumerate(dataloader):
        x.append(data[0])
        y.append(data[1])

    return torch.cat(x, dim = 0), torch.cat(y, dim = 0)

def load_cifar10(args, train_shuffle = False, flatten = True):
    
    x_mean = [0.507, 0.487, 0.441]
    x_std = [0.267, 0.256, 0.276]
    # Define Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=x_mean, std=x_std),
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=x_mean, std=x_std),
        ])
    }
    # Training Data
    train_set = CIFAR10(
        root="./Files/Data", train=True, download=True, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=0)

    # Test Data
    test_set = CIFAR10(
        root="./Files/Data", train=False, download=True, transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # extract data
    tr_data_im, tr_data_lab = extract_data_labels(train_loader)
    te_data_im, te_data_lab = extract_data_labels(train_loader)

    if flatten:
        tr_data_im = torch.flatten(tr_data_im,1)
        te_data_im = torch.flatten(te_data_im,1)

    return tr_data_im, tr_data_lab, None, train_loader, None, None, None, None, te_data_im, te_data_lab, None, test_loader

def load_fashionmnist_old(args, train_shuffle = True):
    print("HERE")
    tr_data = FashionMNIST(root = "./Files/Data", download = True, train = True) 
    te_data  = FashionMNIST(root = "./Files/Data", download = True, train = False)

    tr_data.data = torch.reshape(torch.tensor(tr_data.data), (len(tr_data.data), -1)) / 255
    te_data.data = torch.reshape(torch.tensor(te_data.data), (len(te_data.data), -1)) / 255

    tr_data_im   = tr_data.data[:-10000,:]
    tr_data_lab  = torch.tensor(tr_data.targets[:-10000])
    
    val_data_im   = tr_data.data[-10000:]
    val_data_lab  = torch.tensor(tr_data.targets[-10000:])

    te_data_im   = te_data.data 
    te_data_lab = te_data.targets

    tr_data = PlainDataset(tr_data_im, tr_data_lab, transforms = None)
    val_data = PlainDataset(val_data_im, val_data_lab, transforms = None)
    te_data = PlainDataset(te_data_im, te_data_lab, transforms = None)

    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle = train_shuffle, drop_last=False) ##
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##
    te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##

    return tr_data_im, tr_data_lab, tr_data, tr_loader, val_data_im, val_data_lab, val_data, val_loader, te_data_im, te_data_lab, te_data, te_loader

def load_fashionmnist(args, train_shuffle = True, flatten = True):
        # Define Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2855], std=[0.3528])
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2855], std=[0.3528])
        ])
    }


    # Training Data
    train_set = FashionMNIST(
        root="./Files/Data", train=True, download=True, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=0)

    # Test Data
    test_set = FashionMNIST(
        root="./Files/Data", train=False, download=True, transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # extract data
    tr_data_im, tr_data_lab = extract_data_labels(train_loader)
    te_data_im, te_data_lab = extract_data_labels(train_loader)

    if flatten:
        tr_data_im = torch.flatten(tr_data_im,1)
        te_data_im = torch.flatten(te_data_im,1)


    return tr_data_im, tr_data_lab, None, train_loader, None, None, None, None, te_data_im, te_data_lab, None, test_loader