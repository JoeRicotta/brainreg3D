import argparse
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

from .brainreg3D import BrainReg3D

# command line interface
parser = argparse.ArgumentParser(
    prog="brainreg3D",
    description="For segmenting 2D images using projections onto 3D brain structures.",
    epilog="Written by Joe Ricotta, DPT PhD @ PSU 06/2024."
)

# parsing input arguments
parser.add_argument('-tiff_path', type=Path, help='The tiff file to be registered.')

# results
results = parser.parse_args()
tiff_path = results.tiff_path

# if no argument in the command line, open a dialog asking for tiff files only
if not tiff_path:
    root = Tk()
    root.withdraw()
    tiff_path = askopenfilename(title="Select tiff file to be registered", filetypes=[('Tiff Files', '*.tif')])

# if a file was selected, process it
if tiff_path != '':
    reg = BrainReg3D(tiff_path)
    reg.run()    
else:
    raise(ValueError("No file was selected."))







