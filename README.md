# Super-Resolution Network

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

The ShapeNet dataset can be found at [1].

# Super-Resolution

Own PyTorch implementation of [2], please visit their repository for more details. Please be sure, that the data is inside the respective directory. For training the Super-Resolution-Network, first pre-process the data with:

```
python prepare_SR.py
```

Then train the depth map model with:

```
python train_SR.py --model_type "depth"
```

Followed with the occupancy map model:

```
python train_SR.py --model_type "occupancy"
```

If you use our PyTorch implementation, please refer to our repository `https://github.com/mhohmann1/super-resolution-pytorch` and of course to [3].


# Sources

`[1] ShapeNet`

`[2] https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation`
