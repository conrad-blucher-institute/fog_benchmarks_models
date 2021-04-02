# Fog Detection with 2D CNNs
## Benchmarks for comparison with FogNet (3D CNN)

FogNet is a deep learning architecture for fog prediction by Hamid Kamangir. The input data is a raster cube where each band is a 32x32 meteorological or oceanic variable over a spatial region. The number of variable channels ranges from 288-385, depending on the target lead time. Models has been trained with lead times of 6, 12, and 24 hours. 
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

## Guide

**Active python3 virtual environment**

    source fog-bmark/bin/activate

**Train a model:**

    python fog_image_models.py -a $ARCHITECTURE -e $EPOCHS -i $UNIQ_ID 

    # Example:
    python fog_image_models.py -a resnet34 -e 100 -i 1
    
    # Saved model path:  fog-resnet34__lr0.1__e100__b64__1.pt

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

- [ ] Add final benchmark result spreadsheet to repo
- [ ] Remove hard-coded paths to the FogNet data folders
- [ ] Add references to the FogNet papers, when published
- [ ] Make repo public when appropriate to do so















