from setuptools import setup

setup(
    name="ProtoComp",
    packages=[
        "models.point_e",
        "models.point_e.diffusion",
        "models.point_e.models",
        "models.point_e.util",
    ],
    install_requires=[
        "argparse",
        "easydict",
        "h5py",
        "matplotlib",
        "numpy==1.24.3",
        "open3d==0.14.1",
        "opencv-python",
        "pyyaml",
        "scipy",
        "tensorboardX",
        "timm==0.4.5",
        "tqdm",
        "transforms3d",
        "einops",
        "ipdb",
        "filelock",
        "Pillow",
        "fire",
        "humanize",
        "requests",
        "scikit-image",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
    author="Yanbo Wang",
)
