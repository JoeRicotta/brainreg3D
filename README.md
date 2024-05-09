# brainreg3D
A pipeline for manual registration of cortical regions using 3D projection onto an experimentally obtained 2D image. 

## Installation
Installation can be done in one of three ways: install using pip, or through repo cloning and manual install. The latter is recommended for the example scripts and data to be included in the install.

1. Clone repository (Windows)
```
python -m venv .venv
./.venv/Scripts/Activate.ps1

git clone https://github.com/JoeRicotta/brainreg3D.git
cd brainreg3D

python -m pip install -r requirements.txt
python example.py

```

1. Clone repository (Mac)
```
python3 -m venv .venv
source .venv/bin/activate

git clone https://github.com/JoeRicotta/brainreg3D.git
cd brainreg3D

pip install -r requirements.txt

python example.py
```

2. Install from pip
```
python -m pip install -i https://test.pypi.org/simple/ brainreg3D
```

