import ctypes
import numpy as np

_mod = ctypes.cdll.LoadLibrary("./libtap.so")

_denoise = _mod.denoise
_denoise.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.c_double) 
_denoise.restype = None

_segment = _mod.segment
_segment.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
_segment.restype = None

_segment_mp = _mod.segmentmp
_segment_mp.argtypes =(ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
_segment_mp.restype = None

_discrete_image = _mod.discrete_image
_discrete_image.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)) 
_discrete_image.restype = None

# First version of this function. Assumes that image input is 2D numpy array.
def denoise(image,smooth_parameter=0.1*255):
    
    dims = image.shape

    # the imaging libraries assume that images are of type double with range [0,1]
    cast_uint_double = image.dtype == np.uint8
    if(cast_uint_double):
        # copy the image into 2D array of type double and normalize
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
        
def segment(image, classes=2, means=None):

    imgdims = image.shape
    pixdims = image.shape[:2]  ## pixel dimensions

    if len(imgdims) == 2:
        clrdim = 1
    elif len(imgdims) == 3:
        clrdim = imgdims[2]
    else:
        return

    # the imaging libraries assume that images are of type double with range [0,1]
    # cast the image into correct format and get pointer to image data
    cast_uint_double = image.dtype == np.uint8
    if(cast_uint_double):
        # copy the image into 2D array of type double and normalize
        img_d = image.astype(np.double)/255.0
    else:
        ## assuming float/double type
        img_d = image
    p_src = img_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # initialize array for output and get pointer
    p_dest = np.zeros(imgdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if not means:
        # automatically determine means
        
        # check whether cv2 is installed
        import imp
        try:
            imp.find_module('cv2')
            foundcv2 = True
        except ImportError:
            foundcv2 = False

        if foundcv2:
            # use cv2.kmeans to find initial clusters

            # convergence criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # set flag for random initialization
            flags = cv2.KMEANS_RANDOM_CENTERS

            # Apply KMeans
            z = img_d.reshape(np.prod(pixdims),-1).astype(np.float32)
            compactness,labels,means = cv2.kmeans(z,classes,None,criteria,10,flags)
        elif clrdim == 1:
            centers = np.array(np.xrange(0,classes).as_type(float)/classes).reshape(-1,1)
        else:
            # not yet implemented
            return
    else:
        # TODO: check whether means are inputted as uint8 or as float type and should normalize accordingly

        means = np.array(means) # cast in case inputted as list format
        if len(means.shape) < 2 and clrdim == 1:
            # cast to 2d - useful if image is grayscale and means input is 1d list of grays
            means = means.reshape(-1,1)
        elif len(means.shape) != 2:
            return

        if means.shape[1] != clrdim:
            # check that means has the correct color dimensions
            return

        classes = means.shape[0] # number of elements in means overrules classes value
    p_colors = np.array(means).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    segdims = pixdims + list(classes)
    if classes == 2:
        p_segs = np.zeros(imgdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _segment(p_src,p_segs,imgdims[0],imgdims[1],p_colors)
    else:
        p_segs = np.zeros(segdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _segment_mp(p_src,p_segs,imgdims[0],imgdims[1],classes,p_colors)
    _discrete_image(p_segs,p_dest,imgdims[0],imgdims[1],classes,p_colors)

    # normalize output to image data types
    out_segments = np.ctypeslib.as_array(p_segs,dims)
    out_array = np.ctypeslib.as_array(p_dest,dims)

    if(cast_uint_double):
        out_array = np.round(255*out_array)
        out_image = out_array.astype(np.uint8)
    else:
        out_image = out_array

    return (out_segments,out_image)
