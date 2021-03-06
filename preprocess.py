import torch

from torchvision import datasets, transforms

# NOTE: Mean and std used for normalization are known stats from the distribution of each dataset

def load_data(args):
    print('Load Dataset :: {}'.format(args.dataset))
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        train_len = int(len(train_data)*0.9)
        val_len = len(train_data) - train_len
        print('Len Train: {}, Len Valid: {}'.format(train_len,val_len))
        train_set, valid_set = torch.utils.data.random_split(train_data, [train_len, val_len])
        valid_set.transform = transform_test #Don't want to apply flips and random crops to this

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2673, 0.2564, 0.2762)
            ),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2673, 0.2564, 0.2762)
            ),
        ])

        train_data = datasets.CIFAR100('data', train=True, download=True, transform=transform_train)

        actual_to_be_used = int(len(train_data) * args.subset)
        train_data, _ = torch.utils.data.random_split(train_data, [actual_to_be_used, len(train_data) - actual_to_be_used])

        train_len = int(len(train_data)*0.9)
        val_len = len(train_data) - train_len
        print('Len Train: {}, Len Valid: {}'.format(train_len,val_len))
        train_set, valid_set = torch.utils.data.random_split(train_data, [train_len, val_len])
        valid_set.transform = transform_test #Don't want to apply flips and random crops to this

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'TinyImageNet':
        # We use the normalization stats of the full ImageNet dataset as an estimate for the stats of the
        # TinyImageNet dataset
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        # Only the training data has labels so we will split it up to make training, testing, and validation sets
        train_data = datasets.ImageFolder('./datasets/processed-tiny-imagenet', transform=transform_train)

        train_len = int(len(train_data)*0.8)
        val_len = int(len(train_data)*0.1)
        test_len = int(len(train_data)*0.1)
        train_set, valid_set, test_set = torch.utils.data.random_split(train_data, [train_len, val_len, test_len])
        
        # test_len = int(len(train_set)*0.1)
        # new_train_len = len(train_set) - test_len
        # train_set, test_set = torch.utils.data.random_split(train_set, [new_train_len, test_len])
        
        #Don't want to apply flips and random crops to this
        valid_set.transform = transform_test
        test_set.transform = transform_test

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        print('TinyImageNet Loader')
        print(train_loader)

    return train_loader, valid_loader, test_loader


#This is just for testing purposes
class Args:
    def __init__(self):
        self.batch_size = 32
        self.num_workers = 1
        self.dataset = 'CIFAR10'

if __name__ == '__main__':

    #need to split the training set into train/valid
    args = Args()
    train, valid, test = load_data(args)
