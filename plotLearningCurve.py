from os import environ
import matplotlib as mpl
if environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from os import path
from matplotlib.lines import Line2D

red = "#c70039"
blue = "#996bb8"
purple = "#037535"
colors = {
    "resnet18" : blue,
    "resnet34" : blue,
    "resnet50" : blue,
    "resnet101" : blue,
    "resnet152" : blue,
    "vgg16" : red,
    "vgg19" : red,
    "densenet121" : purple,
    "densenet201" : purple,
}
lines = {
    "resnet18" : 0.8,
    "resnet34" : 0.8,
    "resnet50" : 0.8,
    "resnet101" : 0.8,
    "resnet152" : 0.8,
    "vgg16" : 2.2,
    "vgg19" : 2.2,
    "densenet121" : 2,
    "densenet201" : 2,
}

# Options
parser = OptionParser()
parser.add_option("-i", "--infile",
    help="Input file where each line is a file to plot",
    default="")
parser.add_option("-o", "--outfile",
    help="Output learning curve plot",
    default="")
(options, args) = parser.parse_args()

# InputL: List of files to plot
infile = options.infile
if not path.exists(infile):
    print("File not found: {}".format(infile))
    exit(1)
# Output: plot training curve
outfile = options.outfile

# Parse files to plot
with open(infile) as f:
    files = f.readlines()
files = [x.strip() for x in files]

# Extract models from filenames
models = [m.split("/")[-1] for m in files]
models = [m.split("_")[0] for m in models]

fig, ax = plt.subplots(1, figsize=(5,7))
ax.set_ylim(0, 0.2)
ax.set_ylabel("MSE")
ax.set_xlabel("Epoch")
custom_lines = [Line2D([0], [0], color = colors["resnet18"], lw = lines["resnet18"]),
                Line2D([0], [0], color = colors["vgg16"], lw = lines["vgg16"]),
                Line2D([0], [0], color = colors["densenet121"], lw = lines["densenet121"]),]
ax.legend(custom_lines, ["ResNet", "VGG", "DenseNet"], loc="lower left")

for i, f in enumerate(files):

    # Build grep command
    cmdGrep = 'grep "Average loss: [.0-9]*" -o {} | grep "[.0-9]*" -o'.format(f)

    # Get grep output -> training MSE curve
    try:
        mseCurve = np.array(subprocess.check_output(cmdGrep, shell=True).split("\n")[:-1]).astype("float")
    except:
        print("Failed shell cmd: {}".format(cmdGrep))
        print("  Skipping..")
        continue

    # Plot
    ax.plot(mseCurve, color = colors[models[i]], linewidth = lines[models[i]])

if outfile == "":
    plt.show()
else:
    plt.savefig(outfile)
