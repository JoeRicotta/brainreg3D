from collections.abc import Iterable
import time
from typing import Type, Self
from pathlib import Path
import pickle
from warnings import warn

from brainrender import Scene
from brainrender.actors import Point
from brainrender.render import mtx_swap_x_z, mtx
from brainrender.settings import DEFAULT_ATLAS

import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

from tifffile import imread, imwrite
from vedo import Plane, Image, Mesh



class BrainReg3D(object):
    """
    Main brain registration object.
    """

    # default parameters
    _DEFAULT_BREGMA = np.array([(6333+5066)/2, 440, 5700])
    _DEFAULT_BRAIN_REGIONS = ["MOp", "MOs", "SSp", "SSs", "PTLp", "VIS", "RSP", "ACA", "ORB", "OLF"]
    _DEFAULT_FOCAL_POINT = np.array([0,-23000000,0]) # distance from cranial window center to camera, ~23 cm in my case
    _DEFAULT_CAM = dict(
        pos=(7347.14, -46026.8, -4821.09),
        focal_point=(7347.14, 3337.84, -4821.09),
        viewup=(0, 0, -1.00000),
        roll=180.000,
        distance=49364.6,
        clipping_range=(40460.4, 60692.2),
    )
    _DEFAULT_ATLAS = DEFAULT_ATLAS # renaming for access with only class import
    _DEFAULT_AUTO_CONTRAST = True # to use for automatically adjusting tiff contrast
    _DEFAULT_IMG_DIMS_MM = [] # default image dimensions in mm, width * height.


    def __init__(self, 
                 tiff_path: Path = Path(), 
                 brain_regions: list = _DEFAULT_BRAIN_REGIONS, 
                 focal_point: np.ndarray = _DEFAULT_FOCAL_POINT,
                 cam: dict = _DEFAULT_CAM,
                 image_dims_mm: list[int] = _DEFAULT_IMG_DIMS_MM,
                 verbose: bool = False,
                 _tiff_frame: int = 0,
                 _auto_contrast: bool = _DEFAULT_AUTO_CONTRAST) -> None:
        
        self.tiff_path = self._check_path(tiff_path)
        self.brain_regions = brain_regions
        self.focal_point = focal_point
        self.cam = cam
        self.image_dims_mm = image_dims_mm
        self.verbose = verbose

        # initializing parameters
        self.tiff = imread(self.tiff_path, key=_tiff_frame)

        # if auto contrast is selected, address the image to make it
        # look better during manipulation.
        if _auto_contrast:
            self._enhance_tiff_contrast()

        # flag to change for created objects
        # if an object is loaded from a pickled object,
        # then this is false and nothing is draggable
        self._EDITABLE = True

        # flag to indicate image was matched to another
        self._MATCHED = False

    @staticmethod
    def _check_path(path) -> None:        
        """
        Ensures self.tiff_path is a Path() object.
        Converts a string to a Path() object if necessary.
        Raises error if input is neither string nor path.
        """
        if isinstance(path, str):
            return Path(path).resolve()
        elif isinstance(path, Path):
            return path.resolve()
        else:
            raise(ValueError(f'Variable tiff_path required to be string or pathlib.Path object-- could not recognize {path}'))
     
    @staticmethod
    def _mask_subplot_size(n) -> tuple:
        if n % 2 == 0:
            return int(n/2), 2
        return int((n+1)/2), 2

    def _make_scene(self) -> list[Scene, list]:
        """
        Creates a scene. Returns a Scene object and a list of brain regions
        """

        # create a brainrender scene
        scene = Scene(title="", atlas_name=self._DEFAULT_ATLAS)    

        region_actors = []
        for region in self.brain_regions:
            region_actors.append(scene.add_brain_region(region,alpha=0.9, hemisphere='right'))

        # add points to help register the image
        # the default points are bregma, one mm rostral and one mm lateral
        points = [self._DEFAULT_BREGMA,
                self._DEFAULT_BREGMA + np.array([0,0,-1000]),
                self._DEFAULT_BREGMA + np.array([-1000,0,0])]

        # add coordinate points to the scene
        for point in points:
            scene.add(Point(point))

        return scene, region_actors

    def _enhance_tiff_contrast(self) -> None:
        """
        Enhances the contrast of the tiff object.
        """
        # taken from ImageJ auto enhance algorithm
        # Python code found here:
        # https://forum.image.sc/t/macro-for-image-adjust-brightness-contrast-auto-button/37157/3

        # image datatype
        im_type = self.tiff.dtype
        # minimum and maximum of image
        im_min = np.min(self.tiff)
        im_max = np.max(self.tiff)

        # converting image =================================================================================================

        # case of color image : contrast is computed on image cast to grayscale
        if len(self.tiff.shape) == 3 and self.tiff.shape[2] == 3:
            # depending on the options you chose in ImageJ, conversion can be done either in a weighted or unweighted way
            # go to Edit > Options > Conversion to verify if the "Weighted RGB conversion" box is checked.
            # if it's not checked, use this line
            # im = np.mean(im, axis = -1)
            # instead of the following
            self.tiff = 0.3 * self.tiff[:,:,2] + 0.59 * self.tiff[:,:,1] + 0.11 * self.tiff[:,:,0]
            self.tiff = self.tiff.astype(im_type)

        # histogram computation =============================================================================================

        # parameters of histogram computation depend on image dtype.
        # following https://imagej.nih.gov/ij/developer/macro/functions.html#getStatistics
        # 'The histogram is returned as a 256 element array. For 8-bit and RGB images, the histogram bin width is one.
        # for 16-bit and 32-bit images, the bin width is (max-min)/256.'
        if im_type == np.uint8:
            hist_min = 0
            hist_max = 256
        elif im_type in (np.uint16, np.int32):
            hist_min = im_min
            hist_max = im_max
        else:
            raise NotImplementedError(f"Not implemented for dtype {im_type}")            
        
        # compute histogram
        histogram = np.histogram(self.tiff, bins = 256, range = (hist_min, hist_max))[0]
        bin_size = (hist_max - hist_min)/256

        # compute output min and max bins =================================================================================

        # various algorithm parameters
        h, w = self.tiff.shape[:2]
        pixel_count = h * w
        # the following values are taken directly from the ImageJ file.
        limit = pixel_count/10
        const_auto_threshold = 5000
        auto_threshold = 0

        auto_threshold = const_auto_threshold if auto_threshold <= 10 else auto_threshold/2
        threshold = int(pixel_count/auto_threshold)

        # setting the output min bin
        i = -1
        found = False
        # going through all bins of the histogram in increasing order until you reach one where the count if more than
        # pixel_count/auto_threshold
        while not found and i <= 255:
            i += 1
            count = histogram[i]
            if count > limit:
                count = 0
            found = count > threshold
        hmin = i
        found = False

        # setting the output max bin : same thing but starting from the highest bin.
        i = 256
        while not found and i > 0:
            i -= 1
            count = histogram[i]
            if count > limit:
                count = 0
            found = count > threshold
        hmax = i

        # compute output min and max pixel values from output min and max bins ===============================================
        if hmax >= hmin:
            min_ = hist_min + hmin * bin_size
            max_ = hist_min + hmax * bin_size
            # bad case number one, just return the min and max of the histogram
            if min_ == max_:
                min_ = hist_min
                max_ = hist_max
        # bad case number two, same
        else:
            min_ = hist_min
            max_ = hist_max

        # apply the contrast ================================================================================================
        imr = (self.tiff-min_)/(max_-min_) * 255

        # reassign the tiff image
        self.tiff = imr

    @staticmethod
    def _make_unit_vector(x) -> list:
        x_ = np.array(x)
        norm = np.sqrt(np.sum(x_**2))
        x = x_ / norm
        return x.tolist()

    @classmethod
    def _normcross(cls,x,y) -> list:
        # return normalized cross product
        z_ = np.cross(x,y)
        z_norm = cls._make_unit_vector(z_)
        return z_norm

    @staticmethod
    def _myround(x: Iterable, base=5) -> Iterable:
        # custom rounding function:
        # https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
        return base * np.round(x/base)

    @staticmethod
    def _sort_lists(val) -> list:
        # sorts the z-distances from smallest to highest.
        # resolves duplicate distances based on list position.

        depths = [x[3] for x in val]
        sorted_depths = sorted(depths)
        ranks = [sorted_depths.index(x) for x in depths]

        # ocassionally, the same value will occur twice. In this case, we have
        # to resolve arbitrary rankings.
        if len(set(ranks)) < len(ranks):
            # find which ranks aren't unique
            for i in range(len(ranks)):
                # rank is i
                rank_inds = [ind for ind,j in enumerate(ranks) if i==j]
                if len(rank_inds) > 1:
                    for j,ind_ in enumerate(rank_inds):
                        if j==0:
                            continue
                        ranks[ind_] = ranks[ind_] + j
        return [val[ranks.index(i)] for i in range(len(ranks))]

    @property
    def mask_path(self) -> str:
        out_path = self.tiff_path.parent / (self.tiff_path.stem + '_masks.tif')
        return str(out_path)

    @property
    def pickle_path(self) -> str:
        out_path = self.tiff_path.parent / (self.tiff_path.stem + '_registration_results.pickle')
        return str(out_path)
    
    def match_to(self, reg_path) -> None:
        """
        Specify pickled object to which to match the registration
        """
        self.match_path = reg_path
        self._MATCHED = True

        # test function
        _ = self.match_actor

        return self

    @property
    def match_actor(self) -> Type[Mesh]:
        """
        Produces the matching image template to which to match the current
        active image to. Useful for matching image locations across repeated
        measurements.
        """

        if not self._MATCHED:
            return None
        
        match_path = self._check_path(self.match_path)
        if not match_path.exists():
            print(f"Registration file not identified-- {str(match_path)} not a valid path to an object")

        # load saved object
        reg = load_registration_result(str(match_path))
        matrices = reg.transforms

        # get image matching
        img3D_match = Image(reg.tiff).tomesh().clone().cmap("gray").alpha(.7)

        # looping through all transformations except for the first: this
        # is already present in img3D_match when the tomesh() method is called
        for m in matrices[1:]:
            img3D_match = img3D_match.apply_transform(m)

        # add negligible shift in z to put template and image on different planes
        img3D_match = img3D_match.shift([0, -10, 0])

        # reversing the automatic coordinate transformation which is used during
        # brainrender's scene.render method. This line causes the final two transformations
        # to cancel out into identities, as transformation matrices without translations are their
        # own inverses.
        img3D_match = img3D_match.apply_transform(mtx).apply_transform(np.array(mtx_swap_x_z).T)

        # make unmovable
        img3D_match.draggable(False)
        img3D_match.pickable(False)

        return img3D_match


    def run(self) -> Self:
        """
        Runs the registration pipeline.
        match: location of a prior registration to match to. 
        """

        # make the scene and the regional actors
        scene1, region_actors = self._make_scene()

        # converting the image to mesh
        img = Image(self.tiff)
        img3D = img.tomesh().clone().cmap("gray").alpha(0.85)

        # identifying scale factor from img3D
        if self.image_dims_mm:
            x_scale = self.image_dims_mm[0] * 1000 / np.diff(img3D.xbounds())[0]
            y_scale = self.image_dims_mm[1] * 1000 / np.diff(img3D.ybounds())[0]
            scale_factor = [x_scale, y_scale, 1]
        else:
            scale_factor = 15

        img3D = img3D.scale(scale_factor).rotate_x(-90).rotate_y(90).pos(self._DEFAULT_BREGMA + np.array([0,-400,0]))
        img3D = img3D.draggable(True)

        # add matched template if it exists
        if self._MATCHED:
            scene1.add(self.match_actor)

        # adding the image                     
        scene1.add(img3D)
        scene1.plotter += "Press 'a' to manipulate image.\nshift: translate\nctrl: rotate \nright click: scale"
        pre_coords = scene1.actors[-1].coordinates

        # render the scene and interact
        scene1.render(camera=self.cam)
    
        # ensure that the position of the image has changed
        post_actor = scene1.actors[-1].copy()

        # getting transforms
        post_coords = scene1.actors[-1].coordinates

        # raise error if no movement is detected
        if np.all(pre_coords == post_coords):
            raise(ValueError("No manipulation of the loaded image was detected. Did you hit enter to accept your changes?"))

        # transform moved image to starting coordinate system
        moved_image = scene1.actors[-1].copy().apply_transform(mtx).apply_transform(mtx_swap_x_z)

        # initialize second scene
        scene2, _ = self._make_scene()

        # making image plane
        extent = img.extent()
        starting_extent = [extent[1],extent[3]]
        img_plane = Plane(s=starting_extent).apply_transform(post_actor.transform).apply_transform(mtx).apply_transform(mtx_swap_x_z)

        # get image boundary
        img_boundaries = img_plane.silhouette("2d")
        scene2.add(img_boundaries)
        scene2.plotter += "Imaged brain volume with planar projection"

        # Solving for the rectangular edge norms
        # TODO: clean this up
        origins = img_boundaries.vertices.tolist()
        reorder_list = [0,1,3,2]
        origins = [origins[i] for i in reorder_list]
        origins2 = origins[1:] + [origins[0]]

        # midpoints along frame coordinates
        midpoints = [(np.array(x) + np.array(y)) / 2 for x,y in zip(origins, origins2)]

        # computing normals
        persp_vecs = [moved_image.pos() + self.focal_point - x for x in midpoints]
        normals = [np.array(x) - np.array(y) for x,y in zip(origins,origins2)] #directed along frame axes

        # computing perspectival normals
        persp_norms = [self._normcross(x,-y) for x,y in zip(persp_vecs, normals)]

        # plotting the planes
        planes = [Plane(x,y,s=(1000,1000)) for x,y in zip(midpoints, persp_norms)]
        cols = ['red', 'orange', 'yellow', 'green']

        # adding perspectival cutting planes to the image for viewing
        for pl, cl in zip(planes, cols):
            scene2.add(pl.c(cl))

        # identify different projections
        cut_projections = []
        proj_distances = []

        for i, actor in enumerate(region_actors):
            
            # update the current actor
            cur_mesh = actor.mesh.clone()
            cur_mesh.apply_transform(mtx_swap_x_z)

            # cutting actor along projection planes
            # formed by image surfaces
            cur_mesh.cut_closed_surface(midpoints, persp_norms)
            scene2.add(cur_mesh)

            # get projections to image plane
            try:
                proj_distance = cur_mesh.distance_to(img_plane, signed=True)
            except:
                print(f'{actor.name} does not project onto the image plane; removing from brain region list')
                del self.brain_regions[i]
                continue

            proj_distances.append(proj_distance)

            # define the current full projection
            cur_proj = cur_mesh.copy().project_on_plane(img_plane, point=moved_image.pos() + self.focal_point)
            cur_proj = cur_proj.c(actor.color()).alpha(.8)

            # trying to cut the projection to the image plane
            img_proj = cur_proj.copy().cut_with_box(img_boundaries.bounds())
            cut_projections.append(cur_proj)

            # adding the cut projection to scene 2
            scene2.add(img_proj.copy())

        if self.verbose:
            scene2.render(camera=self.cam)

        scene2.close()

        # plotting the projection coordinates
        # include the 3d distance from the scene
        if self.verbose:
            fig = plt.figure()
            fig.suptitle("Brain volume visible in image")
            ax = fig.add_subplot(projection='3d')

            # plotting the cut projections
            for i,proj in enumerate(cut_projections):
                x = -np.array(proj.coordinates[:,2]).T
                y = np.array(proj.coordinates[:,0]).T
                z = -proj_distances[i]
                ax.scatter(x, y, z)
            plt.show()
            plt.close()


        ############################################
        ### Resolving to image coordinate system ###
        ############################################
        nrows, ncols = self.tiff.shape

        # get the transformation to the current imaging plane
        inv_trans = img_plane.transform.compute_inverse()
        img_scale = img_plane.transform.get_concatenated_transform(0)
        new_projs = []
        fig = plt.figure()
        fig.suptitle("Region projections onto image plane")
        for proj in cut_projections:
            new_proj = proj.copy().apply_transform(inv_trans).apply_transform(img_scale)
            x,y,z = new_proj.coordinates.T
            plt.scatter(x,-y)
            new_projs.append(new_proj)

        if self.verbose:
            plt.show()
        plt.close()

        # quantizing coordinates
        quantized_coords = []
        fig = plt.figure()
        fig.suptitle('Quantized projections onto image plane')

        # get pixel counter
        counts = dict()

        # quantizing coordinate projections onto the image plane
        for i, proj in enumerate(new_projs):
            x,y,_ = proj.coordinates.T
            z = proj_distances[i]
            y = -y # switch sign of y naturally
            new_x = np.array(x, dtype=np.int64)
            new_y = np.array(y, dtype=np.int64)

            # get quantized coordinates
            low_x = self._myround(new_x, base=10)
            low_y = self._myround(new_y, base=10)

            # with quantized coordinates
            for j, xyvals in enumerate(zip(low_x, low_y)):
                xval, yval = xyvals
                xval = int(xval)
                yval = int(yval)

                key = f'{xval},{yval}'
                val = [self.brain_regions[i], x[j], y[j], z[j]]
                if not counts.get(key):
                    counts.update({key : []})
                counts[key].append(val)

            quantized_coords.append([low_x, low_y, z])
            plt.scatter(low_x, low_y)
        
        if self.verbose:
            plt.show()
        plt.close()

        # convert back to coordinates
        ordered_counts = {k : self._sort_lists(v) for k,v in counts.items()}
        recovered = dict()
        for k, v in ordered_counts.items():
            nearest = v[0]
            region = nearest[0]
            if not recovered.get(region):
                recovered.update({region : []})
            recovered[region].append(nearest[1:])

        # recover the values
        if self.verbose:
            fig = plt.figure()
            fig.suptitle("Resolved regional projections onto image plane")
            ax = fig.add_subplot()
            for k,v in recovered.items():
                x,y,z = np.array(v).T
                ax.scatter(x,y,label=k)
            ax.legend()
            plt.show()
            plt.close()

        # converting brain regions to integers
        region_to_int = {self.brain_regions[i] : i for i in range(len(self.brain_regions))}

        # now, trying find_boundaries by support vector machine
        label_vec = []
        coord_vecs = []
        for k,v in recovered.items():
            label_int = region_to_int[k]
            label_vec += ([label_int]*len(v))
            nv = np.array(v)
            coord_vecs += nv[:,:2].tolist()

        # converting to numpy array
        label_vec_ = np.array(label_vec)
        coord_vecs_ = np.array(coord_vecs)

        # k-nearest-neighbors algorithm to determine boundaries
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
        )

        # now run estimates on all values
        x_offset = round(ncols - ncols/2)
        y_offset = round(nrows - nrows/2)
        min_x, max_x = -x_offset, x_offset
        min_y, max_y = -y_offset, y_offset

        # correcting for integer bias
        max_x_ = max_x + ncols - (max_x - min_x)
        max_y_ = max_y + nrows - (max_y - min_y)

        # getting list of pixels in image
        xx , yy = np.mgrid[min_x:max_x_, min_y:max_y_]
        pixel_list = np.c_[xx.ravel(), yy.ravel()]

        # using the estimator to guage boundaries
        clf.set_params(knn__weights="uniform").fit(coord_vecs_, label_vec_)
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            pixel_list,
            response_method="predict",
            plot_method="pcolormesh"
        )
        disp.figure_.suptitle("Pixelwise decision boundary using k-nearest neigbors")
        plt.show()

        # getting pixelwise predictions
        preds = clf.predict(pixel_list)

        # generating masks
        n_masks = len(self.brain_regions)
        masks = np.zeros((n_masks, nrows, ncols), dtype=np.int16)

        # adjust pixel list
        pixel_adj = pixel_list + np.abs(pixel_list.min(0))

        # plotting mask outputs
        srows, scols = self._mask_subplot_size(n_masks)
        ids = [[a,b] for a in range(srows) for b in range(scols)]

        fig, axs = plt.subplots(srows, scols)
        fig.suptitle("Pixelwise masks")
        for i in range(n_masks):

            # getting array of zeros
            mask = masks[i]

            # find indices which are positive
            region_pixels = pixel_adj[preds==i]
            xinds, yinds = region_pixels.T
            mask[yinds, xinds] = 1

            # reassigning the mask
            masks[i] = mask

            # plotting the mask
            srow, scol = ids[i]
            axs[srow,scol].imshow(mask, origin='lower')
            axs[srow,scol].set_title(self.brain_regions[i])

        plt.show()
        plt.close()


        if self._EDITABLE:
            # if the image is editable, write all of the results to file

            # now writing the masks to a tiff file for use in imageJ
            imwrite(self.mask_path, 
                    masks, 
                    photometric='minisblack',
                    imagej=True,
                    metadata={'axes' : 'ZYX', 
                            'Labels' : self.brain_regions})
            
            print(f"Wrote {self.mask_path}")

            # if image wasn't editable, don't save any of the results 
            # assigning variable to object prior to write
            self.registered_date = time.ctime()
            self.quantized_coords = quantized_coords
            self.recovered = recovered
            self.masks = masks
            self.clf = clf
            self.pixel_list = pixel_list
            ntransforms = scene1.actors[-1]._mesh.transform.ntransforms
            self.transforms = [scene1.actors[-1]._mesh.transform.get_concatenated_transform(i).matrix for i in range(ntransforms)]

            # writing results to pickled object
            stream = open(self.pickle_path, 'wb')
            pickle.dump(self, stream, pickle.HIGHEST_PROTOCOL)

            # saving transformation matrix
            print(f"Wrote {self.pickle_path}")
        
        return self


def load_registration_result(pickle_path) -> Type[BrainReg3D]:
    """
    Loads a former registration result
    """

    # finding pickle_path
    stream = open(pickle_path, 'rb')

    # loading object as non-editable
    obj = pickle.load(stream)
    obj._EDITABLE = False

    return obj

        

if __name__ == "__main__":
    reg = BrainReg3D('./resources/sample_image.tif', image_dims_mm=[6.25,4])
    reg.run()
    obj_path = reg.pickle_path

    # trying to match to template
    reg2 = BrainReg3D('./resources/sample_image.tif', image_dims_mm=[6.25,4])
    reg2 = reg2.match_to(obj_path) # template matching to previous registration results
    reg2.run()