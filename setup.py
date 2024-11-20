from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import cv2

# Get the OpenCV library path and include directories
opencv_libs = cv2.getBuildInformation().split('Install path: ')[1].split('\n')[0] + '/lib'
opencv_includes = cv2.getBuildInformation().split('Install path: ')[1].split('\n')[0] + '/include'

ext_modules = [
    Extension(
        name="face_detector",
        sources=["face_detector.pyx"],
        libraries=["opencv_core", "opencv_imgproc", "opencv_objdetect", "opencv_imgcodecs"],
        library_dirs=[opencv_libs],
        include_dirs=[opencv_includes, np.get_include()],
    )
]

setup(
    name="face_detector",
    ext_modules=cythonize(ext_modules),
)

