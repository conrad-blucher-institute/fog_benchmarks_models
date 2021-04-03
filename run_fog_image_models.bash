# Script to run experiment trials
# Generates the results used for comparison with FogNet 
# to be reported in a publication

# Activate virtual environment
source fog-bmark/bin/activate

# Definitions
CMD_TRAIN="python fog_image_models.py "
CMD_TEST="python fog_image_metrics.py "
LOGDIR="logs/"
EPOCHS=100

#########
# TRAIN #
#########

# Resnet-18
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -i 1 > $LOGDIR""/resnet18_1.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -i 2 > $LOGDIR""/resnet18_2.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -i 3 > $LOGDIR""/resnet18_3.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -i 4 > $LOGDIR""/resnet18_4.txt
$CMD_TRAIN -a resnet18 -e $EPOCHS"" -i 5 > $LOGDIR""/resnet18_5.txt
# Resnet-34
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -i 1 > $LOGDIR""/resnet34_1.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -i 2 > $LOGDIR""/resnet34_2.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -i 3 > $LOGDIR""/resnet34_3.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -i 4 > $LOGDIR""/resnet34_4.txt
$CMD_TRAIN -a resnet34 -e $EPOCHS"" -i 5 > $LOGDIR""/resnet34_5.txt
# Resnet-50
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -i 1 > $LOGDIR""/resnet50_1.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -i 2 > $LOGDIR""/resnet50_2.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -i 3 > $LOGDIR""/resnet50_3.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -i 4 > $LOGDIR""/resnet50_4.txt
$CMD_TRAIN -a resnet50 -e $EPOCHS"" -i 5 > $LOGDIR""/resnet50_5.txt
# Resnet-101
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -i 1 > $LOGDIR""/resnet101_1.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -i 2 > $LOGDIR""/resnet101_2.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -i 3 > $LOGDIR""/resnet101_3.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -i 4 > $LOGDIR""/resnet101_4.txt
$CMD_TRAIN -a resnet101 -e $EPOCHS"" -i 5 > $LOGDIR""/resnet101_5.txt
# Resnet-152
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -i 1 > $LOGDIR""/resnet152_1.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -i 2 > $LOGDIR""/resnet152_2.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -i 3 > $LOGDIR""/resnet152_3.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -i 4 > $LOGDIR""/resnet152_4.txt
$CMD_TRAIN -a resnet152 -e $EPOCHS"" -i 5 > $LOGDIR""/resnet152_5.txt
# VGG-16
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -i 1 > $LOGDIR""/vgg16_1.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -i 2 > $LOGDIR""/vgg16_2.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -i 3 > $LOGDIR""/vgg16_3.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -i 4 > $LOGDIR""/vgg16_4.txt
$CMD_TRAIN -a vgg16 -e $EPOCHS"" -i 5 > $LOGDIR""/vgg16_5.txt
# VGG-19
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -i 1 > $LOGDIR""/vgg19_1.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -i 2 > $LOGDIR""/vgg19_2.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -i 3 > $LOGDIR""/vgg19_3.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -i 4 > $LOGDIR""/vgg19_4.txt
$CMD_TRAIN -a vgg19 -e $EPOCHS"" -i 5 > $LOGDIR""/vgg19_5.txt
# Densenet-121
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -i 1n > $LOGDIR""/densenet121_1.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -i 2n > $LOGDIR""/densenet121_2.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -i 3n > $LOGDIR""/densenet121_3.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -i 4n > $LOGDIR""/densenet121_4.txt
$CMD_TRAIN -a densenet121 -e $EPOCHS"" -i 5n > $LOGDIR""/densenet121_5.txt
# DenseNet-201
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -i 1n > $LOGDIR""/densenet201_1.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -i 2n > $LOGDIR""/densenet201_2.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -i 3n > $LOGDIR""/densenet201_3.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -i 4n > $LOGDIR""/densenet201_4.txt
$CMD_TRAIN -a densenet201 -e $EPOCHS"" -i 5n > $LOGDIR""/densenet201_5.txt

########
# TEST #
########

# Resnet-18
$CMD_TEST -m fog-resnet18__lr0.1__e$EPOCHS""__b64__1.pt > fog-resnet18__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-resnet18__lr0.1__e$EPOCHS""__b64__2.pt > fog-resnet18__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-resnet18__lr0.1__e$EPOCHS""__b64__3.pt > fog-resnet18__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-resnet18__lr0.1__e$EPOCHS""__b64__4.pt > fog-resnet18__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-resnet18__lr0.1__e$EPOCHS""__b64__5.pt > fog-resnet18__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-34
$CMD_TEST -m fog-resnet35__lr0.1__e$EPOCHS""__b64__1.pt > fog-resnet35__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-resnet35__lr0.1__e$EPOCHS""__b64__2.pt > fog-resnet35__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-resnet35__lr0.1__e$EPOCHS""__b64__3.pt > fog-resnet35__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-resnet35__lr0.1__e$EPOCHS""__b64__4.pt > fog-resnet35__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-resnet35__lr0.1__e$EPOCHS""__b64__5.pt > fog-resnet35__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-50
$CMD_TEST -m fog-resnet50__lr0.1__e$EPOCHS""__b64__1.pt > fog-resnet50__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-resnet50__lr0.1__e$EPOCHS""__b64__2.pt > fog-resnet50__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-resnet50__lr0.1__e$EPOCHS""__b64__3.pt > fog-resnet50__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-resnet50__lr0.1__e$EPOCHS""__b64__4.pt > fog-resnet50__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-resnet50__lr0.1__e$EPOCHS""__b64__5.pt > fog-resnet50__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-101
$CMD_TEST -m fog-resnet101__lr0.1__e$EPOCHS""__b64__1.pt > fog-resnet101__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-resnet101__lr0.1__e$EPOCHS""__b64__2.pt > fog-resnet101__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-resnet101__lr0.1__e$EPOCHS""__b64__3.pt > fog-resnet101__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-resnet101__lr0.1__e$EPOCHS""__b64__4.pt > fog-resnet101__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-resnet101__lr0.1__e$EPOCHS""__b64__5.pt > fog-resnet101__lr0.1__e$EPOCHS""__b64__5.csv
# Resnet-152
$CMD_TEST -m fog-resnet152__lr0.1__e$EPOCHS""__b64__1.pt > fog-resnet152__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-resnet152__lr0.1__e$EPOCHS""__b64__2.pt > fog-resnet152__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-resnet152__lr0.1__e$EPOCHS""__b64__3.pt > fog-resnet152__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-resnet152__lr0.1__e$EPOCHS""__b64__4.pt > fog-resnet152__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-resnet152__lr0.1__e$EPOCHS""__b64__5.pt > fog-resnet152__lr0.1__e$EPOCHS""__b64__5.csv
# VGG-16
$CMD_TEST -m fog-vgg16__lr0.1__e$EPOCHS""__b64__1.pt > fog-vgg16__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-vgg16__lr0.1__e$EPOCHS""__b64__2.pt > fog-vgg16__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-vgg16__lr0.1__e$EPOCHS""__b64__3.pt > fog-vgg16__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-vgg16__lr0.1__e$EPOCHS""__b64__4.pt > fog-vgg16__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-vgg16__lr0.1__e$EPOCHS""__b64__5.pt > fog-vgg16__lr0.1__e$EPOCHS""__b64__5.csv
# VGG-19
$CMD_TEST -m fog-vgg19__lr0.1__e$EPOCHS""__b64__1.pt > fog-vgg19__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-vgg19__lr0.1__e$EPOCHS""__b64__2.pt > fog-vgg19__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-vgg19__lr0.1__e$EPOCHS""__b64__3.pt > fog-vgg19__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-vgg19__lr0.1__e$EPOCHS""__b64__4.pt > fog-vgg19__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-vgg19__lr0.1__e$EPOCHS""__b64__5.pt > fog-vgg19__lr0.1__e$EPOCHS""__b64__5.csv
# Densenet-121
$CMD_TEST -m fog-densenet121__lr0.1__e$EPOCHS""__b64__1.pt > fog-densenet121__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-densenet121__lr0.1__e$EPOCHS""__b64__2.pt > fog-densenet121__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-densenet121__lr0.1__e$EPOCHS""__b64__3.pt > fog-densenet121__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-densenet121__lr0.1__e$EPOCHS""__b64__4.pt > fog-densenet121__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-densenet121__lr0.1__e$EPOCHS""__b64__5.pt > fog-densenet121__lr0.1__e$EPOCHS""__b64__5.csv
# DenseNet-201
$CMD_TEST -m fog-densenet201__lr0.1__e$EPOCHS""__b64__1.pt > fog-densenet201__lr0.1__e$EPOCHS""__b64__1.csv
$CMD_TEST -m fog-densenet201__lr0.1__e$EPOCHS""__b64__2.pt > fog-densenet201__lr0.1__e$EPOCHS""__b64__2.csv
$CMD_TEST -m fog-densenet201__lr0.1__e$EPOCHS""__b64__3.pt > fog-densenet201__lr0.1__e$EPOCHS""__b64__3.csv
$CMD_TEST -m fog-densenet201__lr0.1__e$EPOCHS""__b64__4.pt > fog-densenet201__lr0.1__e$EPOCHS""__b64__4.csv
$CMD_TEST -m fog-densenet201__lr0.1__e$EPOCHS""__b64__5.pt > fog-densenet201__lr0.1__e$EPOCHS""__b64__5.csv
