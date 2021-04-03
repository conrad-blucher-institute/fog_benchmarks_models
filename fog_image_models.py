# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
# Torchsat
from torchsat.models.classification.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchsat.models.classification.vgg import vgg16, vgg19
from torchsat.models.classification.densenet import densenet121, densenet201
# System
import os
import time
import numpy as np
import pandas as pd
from optparse import OptionParser

###########
# Options #
###########
parser = OptionParser()
parser.add_option("-a", "--architecture",
	help="Selected architecture (resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, densenet121, densenet201)",
	default="resnet18",
)
parser.add_option("-f", "--file",
	help="Path to save trained model",
	default = None,
)
parser.add_option("-l", "--learn_rate",
	help="Learning rate",
	default = 0.1, type="float",
)
parser.add_option("-b", "--batch_size",
	help="Batch size",
	default=64, type="int",
)
parser.add_option("-e", "--epochs",
	help="Number of training epochs",
	default=50, type="int",
)
parser.add_option("-i", "--identifier",
	help="String to add to end of generated output filename",
	default = "test",
)
parser.add_option("--no_sst", 
	help="Do not use SST band",
	default = False, action = "store_true")
(options, args) = parser.parse_args()

# Device for training
device = torch.device("cuda")
# Rate to print training status
printFreq = 15

#########
# Model #
#########

# Selected architecture
architecture = options.architecture

# Use SST?
useSST = not options.no_sst

# Definitions
width    = 32
height   = 32
if useSST:
    channels = 385
else:
    channels = 384
classes  = 2

# Hyperparameters
learningRate = options.learn_rate
nEpochs      =  options.epochs
batchSize    = options.batch_size

# Path to save model
output_file = options.file
if output_file is None:
	# Path based on params
	output_file = "fog-{a}__lr{l}__e{e}__b{b}__{i}.pt".format(
		a=architecture, l=learningRate, e=nEpochs, b=batchSize, i=options.identifier)

# Load architecture
if architecture == "resnet18":
	model = resnet18(in_channels=channels, num_classes=classes)
elif architecture == "resnet34":
	model = resnet34(in_channels=channels, num_classes=classes)
elif architecture == "resnet50":
	model = resnet50(in_channels=channels, num_classes=classes)
elif architecture == "resnet101":
	model = resnet101(in_channels=channels, num_classes=classes)
elif architecture == "resnet152":
	model = resnet152(in_channels=channels, num_classes=classes)
elif architecture == "vgg16":
	model = vgg16(in_channels=channels, num_classes=classes)
elif architecture == "vgg19":
	model = vgg19(in_channels=channels, num_classes=classes)
elif architecture == "densenet121":
	model = densenet121(in_channels=channels, num_classes=classes)
elif architecture == "densenet201":
	model = densenet201(in_channels=channels, num_classes=classes)
elif architecture == "efficientnet":
	print("Not yet implemented efficiennet. Exiting...")
	exit(0)
else:
	print("Architecture {} not found. Exiting...".format(architecture))
	exit(0)

print("")
print("Training fog detection")
print("----------------------")
print("  Architecture:  {}".format(architecture))
print("  Learning rate: {}".format(learningRate))
print("  Epochs:        {}".format(nEpochs))
print("  Batch size:    {}".format(batchSize))
print("  Output path:   {}".format(output_file))
print("  Using SST:     {}".format(str(useSST)))
print("")

model = model.to(device)

# Learning components
criterion    = nn.CrossEntropyLoss()
optimizer    = optim.Adam(model.parameters(), lr=learningRate)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for idx, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device, dtype=torch.long)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(image), len(data_loader.dataset), 100. * idx / len(data_loader), loss.item()))

def evaluate(epoch, model, criterion, data_loader, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True, dtype=torch.long)
            output = model(image)
            loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(data_loader.dataset)/data_loader.batch_size

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

########
# Data #
########

# Directories
targetDir = "/data1/fog/fognn/Dataset/24HOURS/TARGET/"
dataDir   = "/data1/fog/fognn/Dataset/24HOURS/2D/"

# Key to extract numpy array from '.npz' data cube
cubeKey = "arr_0"      # EX: `trainCubesNAM[0][cubeKey]`

# Split years into train, validation, and test
initYear = 2009
lastYear = 2020
allYears = range(initYear, lastYear + 1)
trainYears = [4,  5,  6,  7,  8]
valYears   = [0,  1,  2,  3]
testYears  = [9, 10, 11, ]

# Filename templates
pathPatternMix = "NETCDF_MIXED_CUBE_{}_24.npz"
pathPatternNAM = "NETCDF_NAM_CUBE{}_24.npz"
pathPatternSST = "NETCDF_SST_CUBE_{}_24_scaled.npz"
pathPatternTargets = "target{}_24.csv"

# Generate list of target files
pathsTargets = [targetDir + "/" + pathPatternTargets.format(y) for y in allYears]

# Read targets
trainTargets = np.array(pd.concat([pd.read_csv(pathsTargets[y]) for y in trainYears])["VIS_Cat"]).astype(int)
valTargets   = np.array(pd.concat([pd.read_csv(pathsTargets[y]) for y in valYears])["VIS_Cat"]).astype(int)
testTargets  = np.array(pd.concat([pd.read_csv(pathsTargets[y]) for y in testYears])["VIS_Cat"]).astype(int)

def binarizeTargets(targets):
    targets[np.where(targets ==  0)] = -1
    targets[np.where(targets  >  0)] =  1
    targets[np.where(targets == -1)] =  0
    return targets

# Convert to binary targets
trainTargets = binarizeTargets(trainTargets)
valTargets   = binarizeTargets(valTargets)
testTargets  = binarizeTargets(testTargets)

print("Train targets: {} fog , {} no fog".format(len(trainTargets[np.where(trainTargets ==  0)]),
	len(trainTargets[np.where(trainTargets ==  1)])))
print("Validation targets: {} fog , {} no fog".format(len(valTargets[np.where(valTargets ==  0)]),
	len(valTargets[np.where(valTargets ==  1)])))
print("Test targets: {} fog , {} no fog".format(len(testTargets[np.where(testTargets ==  0)]),
	len(testTargets[np.where(testTargets ==  1)])))

# Print number of targets for each
print("Number targets, train = {}, validate = {}, test = {}".format(
    len(trainTargets), len(valTargets), len(testTargets)))

# Generate lists of data cube files
pathsMix = [dataDir + "/" + pathPatternMix.format(y) for y in allYears]
pathsNAM = [dataDir + "/" + pathPatternNAM.format(y) for y in allYears]
pathsSST = [dataDir + "/" + pathPatternSST.format(y) for y in allYears]

# Load cubes
trainCubesMix = [np.load(pathsMix[idx]) for idx in trainYears]
trainCubesNAM = [np.load(pathsNAM[idx]) for idx in trainYears]
trainCubesSST = [np.load(pathsSST[idx]) for idx in trainYears]
valCubesMix   = [np.load(pathsMix[idx]) for idx in valYears]
valCubesNAM   = [np.load(pathsNAM[idx]) for idx in valYears]
valCubesSST   = [np.load(pathsSST[idx]) for idx in valYears]
#testCubesMix  = [np.load(pathsMix[idx]) for idx in testYears]
#testCubesNAM  = [np.load(pathsNAM[idx]) for idx in testYears]
#testCubesSST  = [np.load(pathsSST[idx]) for idx in testYears]

# Open samples
idx = 0
sampleCubeMix = trainCubesMix[idx][cubeKey]
sampleCubeNAM = trainCubesNAM[idx][cubeKey]
sampleCubeSST = trainCubesSST[idx][cubeKey]

# Print shapes of each cube type
idx = 0
print("[Cube {i}] Mix: instances = {n}, height = {h}, width = {w}, depth = {d}".format(
    i = idx, n = sampleCubeMix.shape[0], h = sampleCubeMix.shape[1],
    w = sampleCubeMix.shape[2], d = sampleCubeMix.shape[3]))
print("[Cube {i}] NAM: instances = {n}, height = {h}, width = {w}, depth = {d}".format(
    i = idx, n = sampleCubeNAM.shape[0], h = sampleCubeNAM.shape[1],
    w = sampleCubeNAM.shape[2], d = sampleCubeNAM.shape[3]))
print("[Cube {i}] SST: instances = {n}, height = {h}, width = {w}, depth = {d}".format(
    i = idx, n = sampleCubeSST.shape[0], h = sampleCubeSST.shape[1],
    w = sampleCubeSST.shape[2], d = sampleCubeSST.shape[3]))


# Concatenate cubes
if useSST:
    trainCubesAll = [list(np.concatenate((trainCubesMix[i][cubeKey],
                                          trainCubesNAM[i][cubeKey], 
                                          trainCubesSST[i][cubeKey]),
                          axis = 3)) for i in range(len(trainCubesMix))]
    
    trainCubesAll = [item for sublist in trainCubesAll for item in sublist]
    valCubesAll   = [list(np.concatenate((valCubesMix[i][cubeKey],
                                          valCubesNAM[i][cubeKey],
                                          valCubesSST[i][cubeKey]),
                          axis = 3)) for i in range(len(valCubesMix))]
    valCubesAll   = [item for sublist in valCubesAll for item in sublist]
    #testCubesAll  = [list(np.concatenate((testCubesMix[i][cubeKey], 
    #                                      testCubesNAM[i][cubeKey], 
    #                                      testCubesSST[i][cubeKey]),
    #                      axis = 3)) for i in range(len(testCubesMix))]
    #testCubesAll  = [item for sublist in testCubesAll for item in sublist]
else:
    trainCubesAll = [list(np.concatenate((trainCubesMix[i][cubeKey],
                                          trainCubesNAM[i][cubeKey]), 
                          axis = 3)) for i in range(len(trainCubesMix))]
    
    trainCubesAll = [item for sublist in trainCubesAll for item in sublist]
    valCubesAll   = [list(np.concatenate((valCubesMix[i][cubeKey],
                                          valCubesNAM[i][cubeKey]),
                          axis = 3)) for i in range(len(valCubesMix))]
    valCubesAll   = [item for sublist in valCubesAll for item in sublist]
    #testCubesAll  = [list(np.concatenate((testCubesMix[i][cubeKey], 
    #                                      testCubesNAM[i][cubeKey]), 
    #                      axis = 3)) for i in range(len(testCubesMix))]
    #testCubesAll  = [item for sublist in testCubesAll for item in sublist]


# Print shape of combined cube
print("Combined: instances = {n}, height = {h}, width = {w}, depth = {d}".format(
    n = len(trainCubesAll), h = trainCubesAll[idx].shape[0],
    w = trainCubesAll[idx].shape[1], d = trainCubesAll[idx].shape[2]))

# Standardize cube 
train_x = np.array(trainCubesAll)
#train_x = zscore(train_x.reshape(-1, 3)).reshape(train_x.shape)
val_x   = np.array(valCubesAll)
#val_x   = zscore(val_x.reshape(-1, 3)).reshape(val_x.shape)

# Convert to pytorch dataset
train_x = np.moveaxis(train_x, (0, 1, 2, 3), (0, 2, 3, 1))
trainTensor_x = torch.Tensor(train_x)
trainTensor_y = torch.Tensor(trainTargets)
val_x = np.moveaxis(val_x, (0, 1, 2, 3), (0, 2, 3, 1))
valTensor_x = torch.Tensor(val_x)
valTensor_y = torch.Tensor(valTargets)

print(trainTensor_x.shape)
print(trainTensor_y.shape)
print(valTensor_x.shape)
print(valTensor_y.shape)

trainData = TensorDataset(trainTensor_x, trainTensor_y) 
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
valData = TensorDataset(valTensor_x, valTensor_y) 
valLoader = DataLoader(valData, batch_size=batchSize, shuffle=True)

#########
# Train #
#########

for epoch in range(nEpochs):
    train_one_epoch(model, criterion, optimizer, trainLoader, device, epoch, printFreq)
    lr_scheduler.step()
    evaluate(epoch, model, criterion, valLoader, device)

# Save model
torch.save(model, output_file)
