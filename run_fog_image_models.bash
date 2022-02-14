# Script to run experiment trials
# Generates the results used for comparison with FogNet 
# For an _under review_ publication

# Activate virtual environment
source fog-bmark/bin/activate

# Definitions
DATADIR="/data1/fog/fognn/Dataset/24HOURS/"  # See README to download data. Set path to local.
CMD_TRAIN="python fog_image_models.py "
CMD_TEST="python fog_image_metrics.py "
LOGDIR="out/logs/"
RESDIR="out/results/"
EPOCHS=100

#########
# TRAIN #
#########

# Resnet-18
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/resnet18_1.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/resnet18_2.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/resnet18_3.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/resnet18_4.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/resnet18_5.txt
# Resnet-34
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/resnet34_1.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/resnet34_2.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/resnet34_3.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/resnet34_4.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/resnet34_5.txt
# Resnet-50
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/resnet50_1.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/resnet50_2.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/resnet50_3.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/resnet50_4.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/resnet50_5.txt
# Resnet-101
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/resnet101_1.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/resnet101_2.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/resnet101_3.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/resnet101_4.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/resnet101_5.txt
# Resnet-152
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/resnet152_1.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/resnet152_2.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/resnet152_3.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/resnet152_4.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/resnet152_5.txt
# VGG-16
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/vgg16_1.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/vgg16_2.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/vgg16_3.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/vgg16_4.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/vgg16_5.txt
# VGG-19
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1 > $LOGDIR""/vgg19_1.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2 > $LOGDIR""/vgg19_2.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3 > $LOGDIR""/vgg19_3.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4 > $LOGDIR""/vgg19_4.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5 > $LOGDIR""/vgg19_5.txt
# Densenet-121
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1n > $LOGDIR""/densenet121_1.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2n > $LOGDIR""/densenet121_2.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3n > $LOGDIR""/densenet121_3.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4n > $LOGDIR""/densenet121_4.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5n > $LOGDIR""/densenet121_5.txt
# DenseNet-201
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 1n > $LOGDIR""/densenet201_1.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 2n > $LOGDIR""/densenet201_2.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 3n > $LOGDIR""/densenet201_3.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 4n > $LOGDIR""/densenet201_4.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -d $DATADIR/2D/ -t $DATADIR/TARGET/ -i 5n > $LOGDIR""/densenet201_5.txt

########
# TEST #
########

# Resnet-18
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-34
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-50
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-101
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-152
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__5.csv
# VGG-16
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__5.csv
# VGG-19
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__5.csv
# Densenet-121
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__5.csv
# DenseNet-201
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__1.pt > $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__2.pt > $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__3.pt > $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__4.pt > $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__5.pt > $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__5.csv

#########################
# Store all predictions #
#########################
# Multiple runs per model because of RAM needed to open all train, validation, and test at once
# Resnet-18
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet18__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet18__lr0.1__e$EPOCHS""__b64__5.csv.b

# Resnet-34
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet34__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet34__lr0.1__e$EPOCHS""__b64__5.csv.b

# Resnet-50
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet50__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet50__lr0.1__e$EPOCHS""__b64__5.csv.b

# Resnet-101
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet101__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet101__lr0.1__e$EPOCHS""__b64__5.csv.b

# Resnet-152
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-resnet152__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-resnet152__lr0.1__e$EPOCHS""__b64__5.csv.b

# VGG-16
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg16__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-vgg16__lr0.1__e$EPOCHS""__b64__5.csv.b

# VGG-19
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-vgg19__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-vgg19__lr0.1__e$EPOCHS""__b64__5.csv.b

# Densenet-121
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet121__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-densenet121__lr0.1__e$EPOCHS""__b64__5.csv.b

# Densenet-201
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__1.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__1.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv.b
cat $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv.b > \
	$RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv
rm $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__1.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__2.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__2.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv.b
cat $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv.b > \
	$RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv
rm $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__2.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__3.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__3.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv.b
cat $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv.b > \
	$RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv
rm $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__3.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__4.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__4.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv.b
cat $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv.b > \
	$RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv
rm $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__4.csv.b

$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__5.pt -y 0,1,2,3,4,5   -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv.a
$CMD_TEST -d $DATADIR/2D/ -t $DATADIR/TARGET/ -m $RESDIR""/fog-densenet201__lr0.1__e$EPOCHS""__b64__5.pt -y 6,7,8,9,10,11 -p $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv.b
cat $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv.b > \
	$RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv
rm $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv.a $RESDIR""/fog-preds-densenet201__lr0.1__e$EPOCHS""__b64__5.csv.b






