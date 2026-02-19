import os

import timm
import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms

from utils import build_model, build_dataset

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def parse_teacher_name(name: str):
    """
    ex)
    deit_tiny_16_cifar10  -> arch=deit_tiny, patch=16, dataset=cifar10
    deit_base_16_cifar100 -> arch=deit_base, patch=16, dataset=cifar100
    """
    parts = name.split('_')
    arch = f"{parts[0]}_{parts[1]}"   # deit_tiny / deit_base
    patch = int(parts[2])             # 16
    dataset = parts[3]                # cifar10 / cifar100
    return arch, patch, dataset

def get_teacher(name):
    _, _, dataset = parse_teacher_name(name)

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    teacher = build_model(teacher_name_list[name], Pretrained=True)
    for param in teacher.parameters():
        param.requires_grad = False

    teacher.reset_classifier(num_classes)
    return teacher


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def train(net, train_data, valid_data, num_epochs, optimizer,scheduler, criterion):
    best_val_acc = -100
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            im = im.to(device)  # (bs, 3, h, w)
            label = label.to(device)  # (bs, h, w)
            # forward
            output,_,_ = net(im,torch.LongTensor(list(range(num_patch))).repeat(im.shape[0], 1).to(device),torch.LongTensor(list(range(num_patch))).repeat(im.shape[0], 1).to(device))
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)
        if scheduler!=None:
            scheduler.step()

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                im = im.to(device)  # (bs, 3, h, w)
                label = label.to(device)  # (bs, h, w)
                output,_,_ = net(im,torch.LongTensor(list(range(num_patch))).repeat(im.shape[0], 1).to(device),torch.LongTensor(list(range(num_patch))).repeat(im.shape[0], 1).to(device))
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                        train_acc / len(train_data), valid_loss / len(valid_data),
                        valid_acc / len(valid_data)))
            print(epoch_str)
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                torch.save(net.state_dict(), './{}.pth'.format(teacher_name))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                            (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))

def validate(net,valid_data):
    valid_loss = 0
    valid_acc = 0
    net = net.eval()
    criterion = nn.CrossEntropyLoss()
    for im, label in valid_data:
        im = im.to(device)  # (bs, 3, h, w)
        label = label.to(device)  # (bs, h, w)
        output, _, _ = net(im,torch.LongTensor(list(range(num_patch))).repeat(im.shape[0], 1).to(device),torch.LongTensor(list(range(num_patch))).repeat(im.shape[0], 1).to(device))
        loss = criterion(output, label)
        valid_loss += loss.item()
        valid_acc += get_acc(output, label)
    epoch_str = (
            "Valid Loss: %f, Valid Acc: %f, "
            % (valid_loss / len(valid_data),
                valid_acc / len(valid_data)))
    print(epoch_str)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_name_list = {
        'deit_tiny_16_cifar10': 'deit_tiny_patch16_224',
        'deit_base_16_cifar10': 'deit_base_patch16_224',
        'deit_tiny_16_cifar100': 'deit_tiny_patch16_224',
        'deit_base_16_cifar100': 'deit_base_patch16_224',
    }

    evaluate_only=False
    teacher_name = 'deit_base_16_cifar10'

    arch, patch_size, dataset = parse_teacher_name(teacher_name)

    if patch_size == 16:
        num_patch = 197   # 14x14 + cls
    elif patch_size == 32:
        num_patch = 50
    else:
        raise NotImplementedError

    if evaluate_only == False:
        model = get_teacher(teacher_name)
        model = model.to(device)
        if 'swin'in teacher_name:
            batch_size=64
        else:
            batch_size=256

        dataset_output = build_dataset(
            arch.split('_')[0],  # deit
            dataset,
            batch_size,
            train_aug=True,
            keep_zero=True,
            train_inverse=False,
            dataset_path="/root/kadap/MyDisk/tools/ide/jhchoi/data/"
        )
        
        train_loader, val_loader = dataset_output[0], dataset_output[1]
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('trainable parameters:{}'.format(total_params))

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)  # optimizer
        # optimizer=torch.optim.SGD(model.parameters(), 0.1, weight_decay=1e-4, momentum=0.9)#same as test_finetune
        epoches=100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

        # train
        train(model, train_loader, val_loader, epoches, optimizer, None, criterion)
    else:
        model = get_teacher(teacher_name)
        model.load_state_dict(torch.load('PATH'))
        model = model.to(device)
        model.eval()
        dataset_output = build_dataset(
            teacher_name.split('_')[0],
            teacher_name.split('_')[-1],
            256,
            train_aug=True,
            keep_zero=True,
            train_inverse=False,
            dataset_path="/root/kadap/MyDisk/tools/ide/jhchoi/data/"
        )
        train_loader, val_loader = dataset_output[0], dataset_output[1]
        validate(model,val_loader)

