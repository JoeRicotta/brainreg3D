from brainreg3D import BrainReg3D

# load the sample image
reg = BrainReg3D('./resources/sample_image.tif', image_dims_mm=[6.25,4])

# run the registration
reg.run()

# the results of the analysis are saved as a pickle file
# at this directory
obj_path = reg.pickle_path

# To match a second registration to the first registration results,
# use the match_to method on the previous results
reg2 = BrainReg3D('./resources/sample_image.tif', image_dims_mm=[6.25,4])
reg2 = reg2.match_to(obj_path) # template matching to previous registration results
reg2.run()