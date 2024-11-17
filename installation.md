# Installation
Start by cloning the repo:
```bash
git clone git@github.com:Yanbo-23/Proto-Comp.git
cd Proto-Comp
```

Then you can create an anaconda environment called `protocomp` as below. 

```bash
conda create -n protocomp python=3.9
conda activate protocomp
```

Step 1: install PyTorch (we tested on 2.3.0, but the following versions should also work):

```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Step 2: install the dependences:

```bash

pip install -e .
```

Step 3: Building Pytorch Extensions for Chamfer Distance and PointNet++:
```bash
# Chamfer Distance
cd ./extensions/chamfer_dist
python setup.py install
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"


```