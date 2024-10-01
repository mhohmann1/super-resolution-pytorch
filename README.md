# Super-Resolution Network
Our PyTorch implementation of [1], please visit their repository for more details. 

# Content
- [Installation](#installation)
- [Data](#data)
- [Super-Resolution](#super-resolution)
- [Sources](#sources)

# Installation

```
conda create -n super-resolution-pytorch
conda activate super-resolution-pytorch
conda env update -n super-resolution-pytorch --file environment.yml
```

# Data

The ShapeNet dataset can be found at [2] and must be placed in the directory `data/ShapeNetCorev1`.

# Super-Resolution

Please be sure, that the data is inside the respective directory. For training the Super-Resolution-Network, first pre-process the data with:

```
python prepare_SR.py
```
A new folder with the different voxel grid resolutions and ODMs will be created in `data/ShapeNetCoreSR`. If you want, you can remove the folder `data/ShapeNetCorev1` manually. 

Then train the depth map model with:

```
python train_SR.py --model_type "depth"
```

Followed with the occupancy map model:

```
python train_SR.py --model_type "occupancy"
```

If you use our PyTorch implementation, please refer to our repository `https://github.com/mhohmann1/super-resolution-pytorch` and of course to [1].


# Sources

`[1] https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation`

`[2] ShapeNetCorev1 - https://shapenet.org/`