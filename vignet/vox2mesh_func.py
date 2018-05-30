"""
================
Voxel to Mesh Converter
================
Written by: Ferdinand John Briones and Norman David Medina

vox2mesh is a simple function that utilizes the marching cube algorithm implemented by Lewiner
to convert a 3D voxel into a mesh and outputs it into an OBJ File Format.

Required Packages:
    skimage
    NumPy
    vispy.io

Parameters:
            voxel: <numpy.ndarray>
                    3D Volume array
            fname: <str>
                    Filename to write. Must end in  '.obj'
"""

from skimage import measure
import numpy as np
from vispy.io import write_mesh


def vox2mesh(fname, voxel):

    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel, 0)
    write_mesh(fname, verts, faces, normals, None, overwrite=True, reshape_faces=True)
