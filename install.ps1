# ensure scripts are allowed in powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# open a virtual environment and run the activation script
python -m venv .venv
./.venv/Scripts/Activate.ps1
git clone https://github.com/JoeRicotta/brainreg3D
python -m pip install -r brainreg3D/requirements.txt

# try using brainreg
python -m brainreg3D