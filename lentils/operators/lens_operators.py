

import numpy as np
from scipy import sparse
from lentils.operators import Operator
from lentils.common import ImageSpace, DelaunaySpace
from lentils.backend import libraster, libtriangles
from lentils.models import GlobalLensModel


# base class for lens operator types
class LensOperator(Operator):
    def __init__(self, space_left, space_right, lensmodel):
        self._lensmodel = lensmodel
        super().__init__(space_left, space_right)
    def get_casted_points():
        raise NotImplementedError
    def compute_entries():
        raise NotImplementedError


class ManifoldLensOperator(LensOperator):

    def __init__(self, image_space, source_space, lensmodel, z_src, **superargs):

        # compute z-planes, etc
        lensmodel.setup_raytracing(z_src)

        # make the matrix
        num_rows = image_space.size
        num_vals = 50*num_rows # a conservative guess 
        row_inds = np.zeros(num_rows+1, dtype=np.int32) 
        cols = np.zeros(num_vals, dtype=np.int32) 
        vals = np.zeros(num_vals, dtype=np.float64) 
        libraster.manifold_lens_matrix_csr(image_space, source_space,
                lensmodel, row_inds, cols, vals)
        self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(image_space.size,source_space.size))

        # finish up and pass along supers
        super().__init__(image_space, source_space, lensmodel)

    
    def rasterize(self, triangles):

        source_space = self.space_right
        rastered = source_space.new_vector()
        b = source_space._bounds.flatten()
        libraster.rasterize_triangle(
                triangles, None, source_space.shape[0], source_space.shape[1], 
                b[0], b[1], b[2], b[3], rastered)
        return rastered 

    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        out_masked = out[self._image_mask]
        out_masked[self._casted_inds] = vec
        out_masked[self._uncasted_inds] = np.sum(self._uncasted_weights*vec[self._uncasted_tris], axis=-1)
        out[self._image_mask] = out_masked
        return out

    def _matrixfree_transpose(self, vec):
        #print('Lensop adjoint shape =', vec.shape)
        #out = vec[self._image_mask][self._casted_inds]
        out = vec[self._image_mask][self._casted_inds]
        #out = vec[self._image_mask][self._casted_inds]
        #out_masked = out[self._image_mask]
        #out_masked[self._casted_inds] = vec
        #out_masked[self._uncasted_inds] = np.sum(self._uncasted_weights*vec[self._uncasted_tris], axis=-1)
        #out[self._image_mask] = out_masked
        return out




# Vegetti and Koopmans delaunay tessellation
class DelaunayLensOperator(LensOperator):

    def __init__(self, image_space, lensmodel, z_src, ncasted=1, **superargs):

        if not isinstance(image_space, ImageSpace):
            raise TypeError("image_space must be ImageSpace")
        if not isinstance(lensmodel, GlobalLensModel):
            raise TypeError("lensmodel must be GlobalLensModel")

        # make image-plane masks containing "casted" and "uncasted" points
        image_mask = image_space.mask
        image_points = image_space.points
        casted_mask = np.zeros_like(image_mask)
        uncasted_mask = np.ones_like(image_mask)
        casted_mask[::ncasted,::ncasted] = True
        uncasted_mask[::ncasted,::ncasted] = False 
        casted_mask *= image_mask
        uncasted_mask *= image_mask

        # deflect the rays
        # Get Delaunay triangulation and triangle indices of uncasted rays
        self.casted_points = lensmodel.deflect(image_points[casted_mask], z_s=z_src)
        self.uncasted_points = lensmodel.deflect(image_points[uncasted_mask], z_s=z_src)
        source_space = DelaunaySpace(self.casted_points) 
        uncasted_tri_inds = source_space.delaunay.find_simplex(self.uncasted_points)

        # make the matrix
        numcasted = np.sum(casted_mask)
        numuncasted = np.sum(uncasted_mask)
        num_rows = image_space.size
        num_vals = numcasted + 3*numuncasted
        row_inds = np.zeros(num_rows+1, dtype=np.int32) 
        cols = np.zeros(num_vals, dtype=np.int32) 
        vals = np.zeros(num_vals, dtype=np.float64) 
        libtriangles.delaunay_lens_matrix_csr(image_space, source_space,
                ncasted, self.uncasted_points, uncasted_tri_inds, row_inds, cols, vals)
        self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(image_space.size,source_space.size))

        # finish up and pass along supers
        super().__init__(image_space, source_space, lensmodel)



