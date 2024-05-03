from collections.abc import Iterable
from pathlib import Path

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

from tifffile import imread, imwrite, TiffFile
from vedo import Plane, Image


###########################
##### PARAMETERS ##########
###########################



class BrainReg3D():
    """
    TODO document register 3D class
    """

    # defining defaults
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


    def __init__(self, 
                 tiff_path: Path = Path(), 
                 brain_regions: list = _DEFAULT_BRAIN_REGIONS, 
                 focal_point: np.ndarray = _DEFAULT_FOCAL_POINT,
                 cam: dict = _DEFAULT_CAM,
                 _tiff_frame = 0) -> None:
        
        self.tiff_path = self._check_path(tiff_path).resolve()
        self.brain_regions = brain_regions
        self.focal_point = focal_point
        self.cam = cam

        # initializing parameters
        self.tiff = imread(self.tiff_path, key=_tiff_frame)
        

    @staticmethod
    def _check_path(path):
        """
        Ensures self.tiff_path is a Path() object.
        Converts a string to a Path() object if necessary.
        Raises error if input is neither string nor path.
        """
        if isinstance(path, str):
            return Path(path)
        elif isinstance(path, Path):
            return path
        else:
            raise(ValueError(f'Variable tiff_path required to be string or pathlib.Path object-- could not recognize {path}'))

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


    @staticmethod
    def make_unit_vector(x) -> list:
        x_ = np.array(x)
        norm = np.sqrt(np.sum(x_**2))
        x = x_ / norm
        return x.tolist()

    @classmethod
    def normcross(cls,x,y) -> list:
        # return normalized cross product
        z_ = np.cross(x,y)
        z_norm = cls.make_unit_vector(z_)
        return z_norm

    @staticmethod
    def myround(x: Iterable, base=5) -> Iterable:
        # custom rounding function:
        # https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
        return base * np.round(x/base)

    @staticmethod
    def sort_lists(val) -> list:
        # sorts the z-distances from smallest to highest.
        # resolves duplicate distances based on list position.

        depths = [x[3] for x in val]
        sorted_depths = sorted(depths)
        ranks = [sorted_depths.index(x) for x in depths]

        # ocassionally, the same value will occur twice. In this case, we have
        # to resolve arbitrary distances.

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
    def tiff_out(self) -> str:
        out_path = self.tiff_path.parent / (self.tiff_path.stem + '_masks.tif')
        return str(out_path)

    def run(self):
        """
        Runs the registration pipeline.
        """

        # make the scene and the regional actors
        scene1, region_actors = self._make_scene()

        # converting the image to mesh
        img = Image(self.tiff)
        img3D = img.tomesh().clone().cmap("gray").alpha(0.85)
        img3D = img3D.scale(15).rotate_x(-90).rotate_y(90).pos(self._DEFAULT_BREGMA + np.array([0,-400,0]))
        img3D = img3D.draggable(True)

        # adding the image                     
        scene1.add(img3D)
        pre_coords = scene1.actors[-1].coordinates

        # render the scene and interact
        scene1.render(camera=self.cam)
        scene1.close()

        # ensure that the position of IOS image has changed
        post_actor = scene1.actors[-1].copy()
        post_coords = scene1.actors[-1].coordinates

        # raise error if no movement is detected
        if np.all(pre_coords == post_coords):
            raise(ValueError("No manipulation of the IOS image was detected. Did you hit enter to accept your changes?"))

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
        persp_norms = [self.normcross(x,-y) for x,y in zip(persp_vecs, normals)]

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
            cut_proj = cur_proj.copy().cut_with_box(img_boundaries.bounds())
            cut_projections.append(cur_proj)

            # adding the cut projection to scene 2
            scene2.add(cut_proj.copy())

        # render the scene
        scene2.render(camera=self.cam)
        scene2.close()

        # plotting the projection coordinates
        # include the 3d distance from the scene
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # plotting the cut projections
        for i,proj in enumerate(cut_projections):
            origin = np.zeros(proj.coordinates.shape[0])
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
        # take projection coordinates and perform inverse transform on the pixels?
        inv_trans = img_plane.transform.compute_inverse()
        img_scale = img_plane.transform.get_concatenated_transform(0)
        new_projs = []
        fig = plt.figure()
        for proj in cut_projections:
            new_proj = proj.copy().apply_transform(inv_trans).apply_transform(img_scale)
            x,y,z = new_proj.coordinates.T
            plt.scatter(x,-y)
            new_projs.append(new_proj)
        plt.show()
        plt.close()

        # quantizing coordinates
        quantized_coords = []
        plt.figure()

        # get pixel counter
        counts = dict()

        # 
        for i, proj in enumerate(new_projs):
            x,y,_ = proj.coordinates.T
            z = proj_distances[i]
            y = -y # switch sign of y naturally
            new_x = np.array(x, dtype=np.int64)
            new_y = np.array(y, dtype=np.int64)

            # get quantized coordinates
            low_x = self.myround(new_x, base=10)
            low_y = self.myround(new_y, base=10)

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
        plt.show()


        # convert back to coordinates
        ordered_counts = {k : self.sort_lists(v) for k,v in counts.items()}
        recovered = dict()
        for k, v in ordered_counts.items():
            nearest = v[0]
            region = nearest[0]
            if not recovered.get(region):
                recovered.update({region : []})
            recovered[region].append(nearest[1:])

        # recover the values
        fig = plt.figure()
        ax = fig.add_subplot()
        for k,v in recovered.items():
            x,y,z = np.array(v).T
            ax.scatter(x,y,label=k)
        ax.legend()
        plt.show()

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

        # now running k-nearest-neighbors algorithm to determine boundaries
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
        DecisionBoundaryDisplay.from_estimator(
            clf,
            pixel_list,
            response_method="predict",
            plot_method="pcolormesh"
        )
        plt.show()

        # getting pixelwise predictions
        preds = clf.predict(pixel_list)

        # I now need to organize these into masks.
        # This is preferable to ROIs because the masks need not be continuous, such as if
        # regions stretched across hemispheres.
        n_masks = len(self.brain_regions)
        masks = np.zeros((n_masks, nrows, ncols), dtype=np.int16)

        # adjust pixel list
        pixel_adj = pixel_list + np.abs(pixel_list.min(0))

        # identifying mask values
        fig, axs = plt.subplots(n_masks, 1)
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
            axs[i].imshow(mask, origin='lower')
            axs[i].set_title(self.brain_regions[i])

        plt.show()

        # now writing the masks to a tiff file for use in imageJ
        imwrite(self.tiff_out, 
                masks, 
                photometric='minisblack',
                imagej=True,
                metadata={'axes' : 'ZYX', 
                        'Labels' : self.brain_regions})
        
        print(f"Wrote {self.tiff_out}")
        

if __name__ == "__main__":
    reg = BrainReg3D('./resources/sample_image.tif')
    reg.run()