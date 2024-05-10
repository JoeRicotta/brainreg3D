# brainreg3D workflow
brainreg3D is based off of the vedo and brainrender framework to allow registration of a 2D image onto a 3D brain. It uses perspectival projections of the brain onto the field of view to map cortical regions to image pixels.

The following images were obtained with the below code:
```
from brainreg3D import BrainReg3D

# by specifying the image_dims_mm parameter, we ensure the loaded image
# is to scale
reg = BrainReg3D('./resources/sample_image.tif', image_dims_mm=[6.25,4])

# if verbose, intermediate processing steps will be plotted and shown
reg.verbose = True

# run the pipeline
reg.run()
```

The first step is to manually align the image over the Allen atlas.

![Registration image](resources/_images/reg_image.png)

After hitting enter, an image with the perspectival projections of the underlying brain regions will appear. These brain regions can be identified (and modified) within the class parameter `BrainReg3D._DEFAULT_BRAIN_REGIONS`.

![Projection image](resources/_images/proj_image.png)

The underlying volumes of these specified regions are then plotted:

![Regional volume](resources/_images/projected_volume.png)

The 2D projections of these regions are shown, without regard for their depth in the brain.

![Unresolved projections](resources/_images/perspectival_projections.png)

Superficial structures mask deep structures. To account for this, the image space is quantized and projections are preserved based on the length of these projections:

![Quantized projections](resources/_images/quantized_projections.png)
![Resolved projections](resources/_images/depth_resolved.png)

A k-nearest neigbors algorithm is then used to label each pixel identity based on the identity of it's neigbors:

![Labeled pixels](resources/_images/pixel_labels.png)

Finally, the pixel labels are converted to masks (images of 0 and 1 values) and saved in the image directory.

![Image masks](resources/_images/pixel_masks.png)

