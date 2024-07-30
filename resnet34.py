import copy # someone on stack overflow says copy is built into python, and thus requires no installation.
import os
import shutil # fine
import time # fine
from PIL import Image # seems fine
import wandb
import pandas as pd


import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np

from collections import Counter

# Set the device
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(device1)

def set_parameter_requires_grad(model, feature_extracting, freeze_layers):
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False #I think this line was freezing a lot of parameters
    else:
        for param in model.parameters():
            param.requires_grad = True

# To initialize the model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, freeze_layers=True, classifier_layer_config=0):
    if model_name == "resnet18":
        model_ft = models.resnet18(weights='DEFAULT')
    if model_name == "resnet34":
        model_ft = models.resnet34(weights='DEFAULT')
    if model_name == "resnet50":
        model_ft = models.resnet50(weights='DEFAULT')
    if model_name == "resnet101":
        model_ft = models.resnet101(weights='DEFAULT')
    if model_name == "resnet152":
        model_ft = models.resnet152(weights='DEFAULT')
    if model_name == "convnext_base":
        model_ft = models.convnext_base(weights='DEFAULT')
    if model_name == "wideresnet50":
        model_ft = models.wide_resnet50_2(weights='DEFAULT')
    if model_name == "efficientnetv2_s":
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
    if model_name == "efficientnetv2_m":
        model_ft = models.efficientnet_v2_m(weights='DEFAULT')
    if model_name == "swin_v2_t":
        model_ft = models.swin_v2_t(weights='DEFAULT')
    if model_name == "swin_v2_s":
        model_ft = models.swin_v2_s(weights='DEFAULT')
    if model_name == 'convnext_small':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'convnext_tiny':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'maxvit':
        model_ft = models.maxvit_t(weights='DEFAULT')
    if model_name == 'mobilenet_large':
        model_ft = models.mobilenet_v3_large(weights='DEFAULT')
    if model_name == 'mobilenet_small':
        model_ft = models.mobilenet_v3_small(weights='DEFAULT')
    if model_name == 'vit_b_16':
        model_ft = models.vit_b_16(weights='DEFAULT')
    if model_name == 'vit_b_32':
        model_ft = models.vit_b_32(weights='DEFAULT')

    set_parameter_requires_grad(model_ft, feature_extract, freeze_layers)

    if model_name == "convnext_base":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((1024, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_small":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_tiny":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "maxvit":
        sequential_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers
    
    elif model_name == "mobilenet_large":
        sequential_layers = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "mobilenet_small":
        sequential_layers = nn.Sequential(
            nn.Linear(576, 1024, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_s":
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_m":
        
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == 'swin_v2_t':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()

        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers
    
    elif model_name == 'swin_v2_s':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers

    elif model_name == 'vit_b_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_b_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    else:       
        num_ftrs = model_ft.fc.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.fc = sequential_layers
        
    if classifier_layer_config != 1:
        sequential_layers.append(nn.LogSoftmax(dim=1))
        model_ft.fc = sequential_layers

    input_size = 224

    return model_ft, input_size

# To train the model
def train_model(model, dataloaders, image_datasets, criterion, optimizer, batch_size, num_epochs, log_interval=20):
    model.to(device1)

    since = time.time()
    val_acc_history = []
    train_loss_history = []
    best_acc = 0.
    best_loss = np.inf
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 30)
        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'
        for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs, labels = inputs.to(device1), labels.to(device1)

            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                    epoch,
                    batch_id * batch_size,
                    len(dataloaders['train'].dataset),
                    100. * batch_id / len(dataloaders['train']),
                    loss.item(),
                    int(running_corrects),
                    batch_id * batch_size
                ))
                epoch_loss, epoch_acc, pred_array, label_array = test_model(model, dataloaders, image_datasets, criterion, epoch)
                # if epoch_acc > best_acc:
                if epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    pred_array_best = pred_array
                    label_array_best = label_array
                    # class_names = Work on adding some kind of distinction based on location in the part
                val_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

        ##Log Validation Statistics
        val_loss, val_acc, pred_array_final, label_array_final = test_model(model, dataloaders, image_datasets, criterion, epoch, epoch_end=True)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)
        print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
        print()

        wandb.log({'epoch_train': epoch, 'train acc': epoch_acc, 'train loss': epoch_loss, 'val acc': val_acc, 'val loss': val_loss})

    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history, pred_array_best, label_array_best, best_loss

# To test the model during training
def test_model(model, dataloaders, image_datasets, criterion, epoch, epoch_end=False):
    model.to(device1)

    running_loss = 0.
    running_corrects = 0.
    model.eval()
    since = time.time()
    pred_array = torch.zeros(0, device=device1)
    label_array = torch.zeros(0, device=device1)

    for batch_id, (inputs, labels) in enumerate(dataloaders['val']):    #Added this line to evaluate model performance for multiple batches
    #for inputs, labels in dataloaders['val']:

        inputs, labels = inputs.to(device1), labels.to(device1)

        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
        pred_array = torch.cat((pred_array, preds.view(-1)), 0)
        label_array = torch.cat((label_array, labels.view(-1)), 0)

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects / len(dataloaders['val'].dataset)
    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))

    wandb.log({"conf_mat_" : wandb.plot.confusion_matrix( 
        preds=np.array(pred_array.cpu()), y_true=np.array(label_array.cpu()), class_names=['Good', 'Clog', 'Void', 'Elliptical', 'Non-Concentric'])})
    if epoch_end:
        df = pd.DataFrame(columns=['img', 'label', 'pred'])
        df.img = (np.array(image_datasets['val'].imgs)[:, 0])
        df.pred = pred_array.cpu()
        df.label = label_array.cpu()
        pd.DataFrame(df).to_csv("data/csv_outputs/" + wandb.run.name +'_'+ wandb.run.id + ".csv")

    # model.load_state_dict(best_model_wts)
    return epoch_loss, epoch_acc, pred_array, label_array
