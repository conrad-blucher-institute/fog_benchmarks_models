import torch
import numpy as np
import pandas as pd
from optparse import OptionParser
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.special import softmax

def getPredictions(model, data_loader, device):
    model.eval()
    trueLabels = []
    predLabels = []
    rawOutputs = []
    model = model.to(device, dtype=torch.double)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.to(device, dtype=torch.double)
            output = model(image)
            pred = output.argmax(dim=1, keepdim=True)
            trueLabels.append(target)
            predLabels.append(pred)
            rawOutputs.append(output)
    return trueLabels, predLabels, rawOutputs


# Calculate metrics
def calcMetrics(y, ypred):
    metrics = {"tn" : None, "fp" : None, "fn" : None, "tp" : None,
               "POD" : None, "F" : None, "FAR" : None, "CSI" : None,
               "PSS" : None, "HSS" : None, "ORSS" : None, "CSS" : None}
    tn, fp, fn, tp = confusion_matrix(y, ypred).ravel()
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp
    a = tn  # Hit
    b = fn  # false alarm
    c = fp  # miss
    d = tp  # correct rejection
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics["POD"] = a/(a+c)
        metrics["F"]   = b/(b+d)
        metrics["FAR"]  = b/(a+b)
        metrics["CSI"] = a/(a+b+c)
        metrics["PSS"] = ((a*d)-(b*c))/((b+d)*(a+c))
        metrics["HSS"] = (2*((a*d)-(b*c)))/(((a+c)*(c+d))+((a+b)*(b+d)))
        metrics["ORSS"] = ((a*d)-(b*c))/((a*d)+(b*c))
        metrics["CSS"] = ((a*d)-(b*c))/((a+b)*(c+d))
    return metrics

def printMetrics(metrics, label="", header = True):
    if header:
        print("LABEL,TN,FN,FP,TP,POD,F,FAR,CSI,PSS,HSS,ORSS,CSS")
    print("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
        label,
        metrics["tn"], metrics["fn"], metrics["fp"], metrics["tp"],
        metrics["POD"], metrics["F"], metrics["FAR"], metrics["CSI"],
        metrics["PSS"], metrics["HSS"], metrics["ORSS"], metrics["CSS"]))

# Find optimal threshold
def findOptimalThreshold(y, yprob, baseMetrics):
    length = yprob.shape[0]
    results = np.empty(shape = (900, 8), dtype='float')
    ypred_ = np.ones(length)
    bestMetrics = baseMetrics
    bestThr = None

    # Evaluate each threshold
    for i, thr in enumerate(np.arange(0.1, 1, 0.001)):
        # Init probs to 0
        ypred_.fill(1)
        # Apply threshold
        ypred_[yprob > thr] = 0
        # Calculate metrics
        metrics = calcMetrics(y, ypred_)

        if metrics["HSS"] >= bestMetrics["HSS"]:
            bestMetrics = metrics
            bestThr = thr

    return bestMetrics, bestThr

# Options
parser = OptionParser()
parser.add_option("-m", "--model",
	help="Path to tested model",
	default=None)
parser.add_option("-t", "--target_dir",
	help="Path to directory with labeled target data",
	default="/data1/fog/fognn/Dataset/24HOURS/TARGET/",
)
parser.add_option("-d", "--data_dir",
	help="Path to directory with input rasters",
	default="/data1/fog/fognn/Dataset/24HOURS/2D/",
)
parser.add_option("-y", "--years",
	help="Comma-delimited list of test years",
	default="9,10,11",
)
parser.add_option("-p", "--predictions_file",
	help="Path to store predictions",
	default=None,
)
parser.add_option("--no_sst",
	help="Skip using SST band",
	default=False, action="store_true",
)
(options, args) = parser.parse_args()

# Load model
modelPath = options.model
model = torch.load(modelPath)

useSST = not options.no_sst

batchSize = 64

predsFile = options.predictions_file

# Directories
targetDir = options.target_dir
dataDir   = options.data_dir

# Key to extract numpy array from '.npz' data cube
cubeKey = "arr_0"      # EX: `testCubesNAM[0][cubeKey]`

# Test data cubes
initYear = 2009
lastYear = 2020
allYears = range(initYear, lastYear + 1)
testYears  = np.array(options.years.split(",")).astype(np.int)

# Filename templates
pathPatternMix = "NETCDF_MIXED_CUBE_{}_24.npz"
pathPatternNAM = "NETCDF_NAM_CUBE{}_24.npz"
pathPatternSST = "NETCDF_SST_CUBE_{}_24_scaled.npz"
pathPatternTargets = "target{}_24.csv"

# Generate list of target files
pathsTargets = [targetDir + "/" + pathPatternTargets.format(y) for y in allYears]
# Read targets
testTargets  = np.array(pd.concat([pd.read_csv(pathsTargets[y]) for y in testYears])["VIS_Cat"]).astype(int)

def binarizeTargets(targets):
    targets[np.where(targets ==  0)] = -1
    targets[np.where(targets  >  0)] =  1
    targets[np.where(targets == -1)] =  0
    return targets

# Convert to binary targets
testTargets  = binarizeTargets(testTargets)

#print("Test targets: {} fog , {} no fog".format(len(testTargets[np.where(testTargets ==  0)]),
#                                                len(testTargets[np.where(testTargets ==  1)])))

# Generate lists of data cube files
pathsMix = [dataDir + "/" + pathPatternMix.format(y) for y in allYears]
pathsNAM = [dataDir + "/" + pathPatternNAM.format(y) for y in allYears]
pathsSST = [dataDir + "/" + pathPatternSST.format(y) for y in allYears]
testCubesMix  = [np.load(pathsMix[idx]) for idx in testYears]
testCubesNAM  = [np.load(pathsNAM[idx]) for idx in testYears]
if useSST:
    testCubesSST  = [np.load(pathsSST[idx]) for idx in testYears]

# Concatenate cubes
if  useSST:
    testCubesAll  = [list(np.concatenate((testCubesMix[i][cubeKey],
                                          testCubesNAM[i][cubeKey],
                                          testCubesSST[i][cubeKey]),
                          axis = 3)) for i in range(len(testCubesMix))]
else:
    testCubesAll  = [list(np.concatenate((testCubesMix[i][cubeKey],
                                      testCubesNAM[i][cubeKey]),
                          axis = 3)) for i in range(len(testCubesMix))]
testCubesAll  = [item for sublist in testCubesAll for item in sublist]

# Convert to pytorch dataset
test_x = np.array(testCubesAll)
test_x = np.moveaxis(test_x, (0, 1, 2, 3), (0, 2, 3, 1))
testTensor_x = torch.Tensor(test_x)
testTensor_y = torch.Tensor(testTargets)
testData = TensorDataset(testTensor_x, testTensor_y)
testLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)

# Get predictions
device = torch.device("cuda")
trueLabels, predLabels, rawOutputs = getPredictions(model, testLoader, device)
y = np.array([int(item.item()) for sublist in trueLabels for item in sublist])
ypred = np.array([int(item.item()) for sublist in predLabels for item in sublist])
yraw = [t.cpu() for t in rawOutputs]
yraw = np.concatenate([t.numpy() for t in yraw])
yprobs = softmax(yraw, axis=1)
yprob = yprobs[:, 0]

if predsFile is not None:
    # Save the predictions

    # Get dates (identify data)
    dates = []
    for year in testYears:
        dates.extend(pd.read_csv(pathsTargets[year])["Date"].values)

    # Combine into csv
    dfPred = pd.DataFrame(
        np.column_stack([dates, yprobs[:, 0], yprobs[:, 1]]),
        columns=["date", "prob 0", "prob 1"]
    )

    dfPred.to_csv(predsFile, index=False, header=False)

# Calculate base metrics
baseMetrics = calcMetrics(y, ypred)
printMetrics(baseMetrics, label="base")

# Calculate optimal threshold & metrics
bestMetrics, bestThr = findOptimalThreshold(y, yprob, baseMetrics)
printMetrics(bestMetrics, label=str(bestThr), header=False)

# Calculate for a given threshold
thr = 0.343
ypred_ = np.ones(ypred.shape[0])
# Apply threshold
ypred_[yprob > thr] = 0
# Calculate metrics
testMetrics = calcMetrics(y, ypred_)
printMetrics(testMetrics, label=str(thr), header=False)
