import ctypes
import numpy as np
import time
from ctypes import cdll

lib = cdll.LoadLibrary("/app/libclassifierv2trtlib.so")


# init func
lib.InitializeGPUMemory.argtypes = [
    ctypes.c_int, # batch size
    ctypes.c_int, # InputW
    ctypes.c_int, # InputH 
    ctypes.c_int, # InputChannel
    ctypes.c_int, # OutputChannel
    ctypes.c_int, # DebugLevel
    ctypes.c_int, # DebugImage
]

lib.InitializeGPUMemory.restype = None

# load onnx function
lib.load_onnx.argtypes = [
    ctypes.c_char_p, # onnxPath
    ctypes.c_char_p, # trtPath
    ctypes.c_int, # isFP16
]
lib.load_onnx.restype = ctypes.c_int

# load engine function
lib.load_engine.argtypes = [
    ctypes.c_char_p, # trtPath
]
lib.load_engine.restype = ctypes.c_int

# Initialize the batch
lib.batchInitialize.argtypes = None
lib.batchInitialize.restype = None


##
class Metdata_for_Crop(ctypes.Structure):
    _fields_ = [("tlx", ctypes.c_int), # top left x
                ("tly", ctypes.c_int), # top left y
                ("brx", ctypes.c_int), # bottom left x
                ("bry", ctypes.c_int), # bottom right y
                ("width", ctypes.c_int),
                ("height", ctypes.c_int)]

class Metdata_for_Mask(ctypes.Structure):
    _fields_ = [("tlx", ctypes.c_int),
                ("tly", ctypes.c_int),
                ("brx", ctypes.c_int),
                ("bry", ctypes.c_int)]

class Metdata_for_Resize(ctypes.Structure):
    _fields_ = [("tlx", ctypes.c_int),
                ("tly", ctypes.c_int),
                ("brx", ctypes.c_int),
                ("bry", ctypes.c_int),
                ("scale", ctypes.c_float)] # scale
##

# Adding images and boxes
class BoundingBox(ctypes.Structure):
    _fields_ = [("bbox_id", ctypes.c_int),  # bbox_id
                ("tlx", ctypes.c_int),  # top left x
                ("tly", ctypes.c_int),  # top left y
                ("brx", ctypes.c_int),  # bottom left x
                ("bry", ctypes.c_int)]  # bottom right y

class ImageWithBoundingBoxes(ctypes.Structure):
    _fields_ = [("image_id", ctypes.c_int),
                ("image_data", ctypes.POINTER(ctypes.c_ubyte)),
                ("image_width", ctypes.c_int),
                ("image_height", ctypes.c_int),
                ("num_bounding_boxes", ctypes.c_int),
                ("bounding_boxes", ctypes.POINTER(BoundingBox)),
                ("crop_metadata", ctypes.POINTER(Metdata_for_Crop)), # meta data for cropping
                # ("mask_metadata", ctypes.POINTER(Metdata_for_Mask)), # meta data for masking
                ("resize_metadata", ctypes.POINTER(Metdata_for_Resize))] # meta data for resizing

add_images_with_bounding_boxes = lib.add_images_with_bounding_boxes
add_images_with_bounding_boxes.argtypes = [ctypes.POINTER(ImageWithBoundingBoxes), ctypes.c_int]
add_images_with_bounding_boxes.restype = ctypes.c_int
##

# Output data
do_inference = lib.do_inference
do_inference.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # Output data
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # Meta ids 
]
do_inference.restype = ctypes.c_int
##