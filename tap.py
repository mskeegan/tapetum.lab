import ctypes
import numpy as np
import numpy.ctypeslib as npct
import time

_mod = npct.load_library('libtap','.')
#_mod = ctypes.cdll.LoadLibrary('./libtap.so')

# Note: so far, the time differences I've seen between C and Fortran contiguous matrices
# has been negligible. 
# TODO: come up with a better test for this.
c_double_1d_ptr = npct.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')
c_double_2d_ptr = npct.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')
c_double_3d_ptr = npct.ndpointer(dtype=np.double, ndim=3, flags='C_CONTIGUOUS')
c_double_ptr = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')

_denoise = _mod.denoise
#_denoise.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.c_double)
_denoise.argtypes = (c_double_2d_ptr,c_double_2d_ptr,ctypes.c_int,ctypes.c_int,ctypes.c_double) 
_denoise.restype = None

_segment = _mod.segment
_segment.argtypes = (c_double_2d_ptr,c_double_2d_ptr,ctypes.c_int,ctypes.c_int,c_double_1d_ptr)
_segment.restype = None

_segment_mp = _mod.segmentmp
#_segment_mp.argtypes =(ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
_segment_mp.argtypes =(c_double_2d_ptr,c_double_3d_ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,c_double_1d_ptr)
_segment_mp.restype = None

_discrete_image = _mod.discrete_image
_discrete_image.argtypes = (c_double_ptr,c_double_2d_ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,c_double_1d_ptr) 
_discrete_image.restype = None

# First version of this function. Assumes that image input is 2D numpy array.
def denoise(image,smooth_parameter=0.1*255):
    
    totalstart=time.time()

    dims = image.shape

    # the imaging libraries assume that images are of type double with range [0,1]
    cast_uint_double = image.dtype == np.uint8
    if(cast_uint_double):
        # copy the image into 2D array of type double and normalize
        img_d = image.astype(np.double)/255.0
    else:
        ## assuming of float/double type
        img_d = image
    #p_src = img_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #p_src = np.asfortranarray(img_d)
    p_src = img_d

    # initialize arrays for output, get pointer
    # p_dest = np.zeros(dims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #p_dest = np.asfortranarray(np.zeros(dims))
    p_dest = np.zeros(dims)

    start=time.time()
    #denoise
    _denoise(p_src,p_dest,dims[1],dims[0],smooth_parameter)
    end=time.time()

    # normalize output to image data types
    #out_array = np.ctypeslib.as_array(p_dest,dims)
    out_array=p_dest
    if(cast_uint_double):
        out_array = np.round(255*out_array)
        out_image = out_array.astype(np.uint8)
    else:
        out_image = out_array

    totalend=time.time()

    print "Call time:", end-start
    print "Total time:", totalend-totalstart

    return out_image
        
def segment2(image, classes=2, means=None):

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
            import cv2

            # convergence criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # set flag for random initialization
            flags = cv2.KMEANS_RANDOM_CENTERS

            # Apply KMeans
            z = img_d.reshape(np.prod(pixdims),-1).astype(np.float32)
            compactness,labels,means = cv2.kmeans(z,classes,criteria,10,flags)
        elif clrdim == 1:
            means = np.array(np.xrange(0,classes).as_type(float)/classes).reshape(-1,1)
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

    if classes == 2:
        segdims = pixdims
        print "Segmenting image into",classes,"classes. Dimensions:",segdims
        p_segs = np.zeros(segdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _segment(p_src,p_segs,segdims[0],segdims[1],p_colors)
    else:
        segdims = tuple(list(pixdims) + [classes])
        print "Segmenting image into",classes,"classes. Dimensions:",segdims
        p_segs = np.zeros(segdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _segment_mp(p_src,p_segs,segdims[0],segdims[1],classes,p_colors)

    _discrete_image(p_segs,p_dest,imgdims[0],imgdims[1],classes,p_colors)

    # normalize output to image data types
    out_segments = np.ctypeslib.as_array(p_segs,segdims)
    out_array = np.ctypeslib.as_array(p_dest,imgdims)

    if(cast_uint_double):
        out_array = np.round(255*out_array)
        out_image = out_array.astype(np.uint8)
    else:
        out_image = out_array

    return (out_segments,out_image)

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
        # copy the image into double array and normalize
        img_d = image.astype(np.double)/255.0
    else:
        ## assuming float/double type
        img_d = image

    ###p_src = img_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_src = img_d

    # initialize array for output and get pointer
    ###p_dest = np.zeros(imgdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_dest = np.zeros(imgdims)

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
            import cv2

            # convergence criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # set flag for random initialization
            flags = cv2.KMEANS_RANDOM_CENTERS

            # Apply KMeans
            print "Find centers by k-means."
            z = img_d.reshape(np.prod(pixdims),-1).astype(np.float32)
            compactness,labels,means = cv2.kmeans(z,classes,criteria,10,flags)
        elif clrdim == 1:
            means = np.array(np.xrange(0,classes).as_type(double)/classes).reshape(-1,1)
        else:
            # not yet implemented
            return
    else:
        # TODO: check whether means are inputted as uint8 or as float type and should normalize accordingly

        means = np.array(means,dtype=np.double) # cast in case inputted as list format
        if len(means.shape) < 2 and clrdim == 1:
            # cast to 2d - useful if image is grayscale and means input is 1d list of grays
            means = means.reshape(-1,1)
        elif len(means.shape) != 2:
            return

        if means.shape[1] != clrdim:
            # check that means has the correct color dimensions
            return

        classes = means.shape[0] # number of elements in means overrules classes value
    ###p_colors = np.array(means).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_colors = np.array(means).astype(np.double)

    if classes == 2:
        segdims = pixdims
        print "Segmenting image into",classes,"classes. Dimensions:",segdims
        ###p_segs = np.zeros(segdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_segs = np.zeros(segdims)
        _segment(p_src,p_segs,segdims[1],segdims[0],p_colors)
    else:
        segdims = tuple(list(pixdims) + [classes])
        print "Segmenting image into",classes,"classes. Dimensions:",segdims
        #p_segs = np.zeros(segdims).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_segs = np.zeros(segdims)
        _segment_mp(p_src,p_segs,segdims[1],segdims[0],classes,p_colors)

    _discrete_image(p_segs,p_dest,imgdims[1],imgdims[0],classes,p_colors)

    # normalize output to image data types
    ###out_segments = np.ctypeslib.as_array(p_segs,segdims)
    ###out_array = np.ctypeslib.as_array(p_dest,imgdims)
    out_segments=p_segs
    out_array=p_dest


    if(cast_uint_double):
        out_array = np.round(255*out_array)
        out_image = out_array.astype(np.uint8)
    else:
        out_image = out_array

    return (out_segments,out_image)
