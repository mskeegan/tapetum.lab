import ctypes
import numpy as np

_mod = ctypes.cdll.LoadLibrary("./libtap.so")

_denoise = _mod.denoise
_denoise.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.c_double) 
_denoise.restype = None

_segment = _mod.segment_1c
_segment.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)) 
_segment.restype = None

_discrete_image = _mod.discrete_image
_discrete_image.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)) 
_discrete_image.restype = None

# First version of this function. Assumes that image input is 2D numpy array.
def denoise(image,smooth_parameter=0.1*255):
    
    dims = image.shape

    cast_uint_double = image.dtype == np.uint8

    # copy the image into 2D array of type double and normalize, get pointer
    if(cast_uint_double):
        img_d = image.astype(np.double)/255.0
    else:
        ## assuming of float/double type
        img_d = image
    p_src = img_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # initialize arrays for output, get pointer
    p_dest = np.zeros(dims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    #denoise
    _denoise(p_src,p_dest,dims[0],dims[1],smooth_parameter)
    
    # normalize output to image data types
    out_array = np.ctypeslib.as_array(p_dest,dims)
    if(cast_uint_double):
        out_array = np.round(255*out_array)
        out_image = out_array.astype(np.uint8)
    else:
        out_image = out_array

    return out_image
        
def segment(image):

    dims = image.shape
    
    cast_uint_double = image.dtype == np.uint8
    if(cast_uint_double):
        img_d = image.astype(np.double)/255.0
    else:
        ## assuming of float/double type
        img_d = image
    p_src = img_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # initialize arrays for output, get pointer
    p_dest = np.zeros(dims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_segs = np.zeros(dims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    p_colors = np.array([0.0,0.6]).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    _segment(p_src,p_segs,dims[0],dims[1],p_colors)
    _discrete_image(p_segs,p_dest,dims[0],dims[1],p_colors)

    # normalize output to image data types
    out_segments = np.ctypeslib.as_array(p_segs,dims)
    out_array = np.ctypeslib.as_array(p_dest,dims)

    if(cast_uint_double):
        out_array = np.round(255*out_array)
        out_image = out_array.astype(np.uint8)
    else:
        out_image = out_array

    return (out_segments,out_image)
