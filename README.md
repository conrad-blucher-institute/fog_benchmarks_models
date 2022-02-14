# Fog Detection with 2D CNNs
## Benchmarks for comparison with FogNet (3D CNN)

[FogNet](https://gridftp.tamucc.edu/fognet/) is a deep learning architecture for fog prediction. The input data is a raster cube where each band is a 32x32 meteorological or oceanic variable over a spatial region. The number of variable channels ranges from 288-385, depending on the target lead time. Models has been trained with lead times of 6, 12, and 24 hours. 
For clarity, here _architecture_ is a network and _model_ is a specific trained instance. 

FogNet achieves good predictive performance using:

- Dense blocks
- Attention maps
- Grouped correlated features
- 3D, dilated convolution

The purpose of this repo is to train several popular 2D CNN architectures for 24-hour fog prediction. The input rasters are 32x32x385. Given the large size of the training, validation, and testing data these are not included in the repo, but will be provided when FogNet is published. As expected, the models were not able to match the performance of FogNet. In fact, the models often failed to learn at all. That is, 98% accuracy was achieved by simply always predicting no fog since fog is underrepresented compared to no fog. Training was performed on architectures implemented in the [TorchSat](https://github.com/sshuair/torchsat) python library because it's implementations allow for an arbitrary number of bands. Typical implementations assume RGB or grayscale input images. 

The following architectures were trained (use these exact names with the `-a` option to train the selected model):

- `VGG16`, `VGG19`
- `ResNet18`, `ResNet34`, `ResNet50`, `ResNet101`, `ResNet152`
- `DenseNet101`, `DenseNet152`

## Note

Due to the file sizes, `out/results` has each trained model's metric CSV, but not the saved model weights. 
The saved weights, along with all other outputs generated for the **FogNet Ablation Study Paper (Under Review)**, are archived [here](https://gridftp.tamucc.edu/fognet/datashare/archive/2D_benchmarks/fog_benchmark_models_outputs-02132022.tar.gz).

## Repo organization

- fog-benchmark-models:
	- `fog_image_models.py`: training script
	- `fog_image_metrics.py`: testing (metrics) script
	- `shrink_SST.py`: resize 128x12 Sea Surface Temperature raster to 32x32
	- `run_fog_image_models.bash`: run all trials for FogNet comparison
	- `plotLearningCurve.py`: generates learning curve plot for the paper
	- `README.md`: this document
	- `venv`: python2.7.6 virtual environment
	- `out`: output directoryy
		- `out/fog_benchmark_runs.csv`: metrics comparison of various models/trials for FogNet comparison
		- `out/logs`: directory for piping the training script output
		- `out/results`: directory for model weights and metrics for each model trained


## Fog dataset

The FogNet data is available at https://gridftp.tamucc.edu/fognet/.
The following instructions show how to download the 24-hour lead time dataset that is compatable with the benchmark architectures. 
The [main FogNet repository](https://github.com/conrad-blucher-institute/FogNet) has more information on the source of the dataset. 

First, choose where to install the data: we will refer to this directory as `$DATASETS`.
    
    # Go to desired download location
    cd $DATASETS

    # Download dataset
    wget -m https://gridftp.tamucc.edu/fognet/datashare/archive/datasets/24HOURS


## Model training & testing

**Train a model:**

    python fog_image_models.py -a $ARCHITECTURE -e $EPOCHS -i $UNIQ_ID -d $DATA_DIR -t $TARGET_DIR

    # Example:
    python fog_image_models.py -a resnet34 -e 100 -i 1 -d $DATASETS/24HOURS/2D/ -t $DATASETS/24HOURS/TARGET/
    
    # Saved model path:  out/results/fog-resnet34__lr0.1__e100__b64__1.pt

**Test a model:**

    python fog_image_metrics.py -m $MODEL 

    # Example:
    python fog_image_metrics.py fog-resnet34__lr0.1__e100__b64__1.pt


**Resize sea surface temperature (SST) band:**

Each FogNet band is a 32x32 grid. However, the SST data begins as a high resolution 128x128 image.
A component of FogNet is dimension reduction using convolution to resize the SST to 32x32 for inclusion with the other channels. 
Since this is not part of the benchmark 2D CNNs, the `shrink_SST.py` script is used to simply scale the image to 32x32.
You will need to run this if the resized SST is not already available.

    python shrink_SST.py

## Todo 

- [X] Add final benchmark result spreadsheet to repo
- [X] Remove hard-coded paths to the FogNet data folders
- [ ] Make repo public when appropriate to do so

