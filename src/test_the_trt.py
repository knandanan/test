# Import the ctypes library
import ctypes
import time

import numpy as np

# import preprocess_lib as preprocess
import code_deploy_v6 as preprocess

#Not to be used
import cv2
from PIL import Image, ImageDraw


lib = ctypes.CDLL("/app/src/build/libclassifierv2trtlib.so")

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

# HyperParameters setting
BatchSize = 2
InputH = 128
InputW = 128
InputChannel = 3
OutputChannel = 2
isFP16 = 0
DebugLevel = 10


# Initialize the trt
lib.InitializeGPUMemory(BatchSize, InputW, InputH, InputChannel, OutputChannel, DebugLevel, DebugImage)

# Set the onnx and trt engine path
onnxPath = "/app/onnx_models/fp_tp_128.onnx".encode('utf-8')
trtPath = "/app/onnx_models/fp_tp_128.trt".encode('utf-8')

# Model builders
status = lib.load_onnx(onnxPath, trtPath, isFP16)
if status:
    print("Onnx build done")

status = lib.load_engine(trtPath)
if status:
    print("TRT build done")


## Running for 2 times


# Initialize the batch
lib.batchInitialize.argtypes = None
lib.batchInitialize.restype = None
lib.batchInitialize()

##
class Metdata_for_Crop(ctypes.Structure):
    _fields_ = [("tlx", ctypes.c_int),
                ("tly", ctypes.c_int),
                ("brx", ctypes.c_int),
                ("bry", ctypes.c_int),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int)]

# class Metdata_for_Mask(ctypes.Structure):
#     _fields_ = [("tlx", ctypes.c_int),
#                 ("tly", ctypes.c_int),
#                 ("brx", ctypes.c_int),
#                 ("bry", ctypes.c_int)]

class Metdata_for_Resize(ctypes.Structure):
    _fields_ = [("tlx", ctypes.c_int),
                ("tly", ctypes.c_int),
                ("brx", ctypes.c_int),
                ("bry", ctypes.c_int),
                ("scale", ctypes.c_float)]

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
                ("crop_metadata", ctypes.POINTER(Metdata_for_Crop)),
                # ("mask_metadata", ctypes.POINTER(Metdata_for_Mask)),
                ("resize_metadata", ctypes.POINTER(Metdata_for_Resize))]

add_images_with_bounding_boxes = lib.add_images_with_bounding_boxes
add_images_with_bounding_boxes.argtypes = [ctypes.POINTER(ImageWithBoundingBoxes), ctypes.c_int]
add_images_with_bounding_boxes.restype = ctypes.c_int

def convert_image_to_byte_array(image_path):
    img = cv2.imread(image_path)
    jpg_np_img = np.array(img, dtype='<u2')
    jpg_np_img = cv2.convertScaleAbs(jpg_np_img)
    print("shape is ", jpg_np_img.shape)
    print("dtype is ", jpg_np_img.dtype)
    print("ndim is ", jpg_np_img.ndim)
    print("itemsize is ", jpg_np_img.itemsize) # size in bytes of each array element
    print("nbytes is ", jpg_np_img.nbytes) # size in bytes of each array element
    return jpg_np_img.tobytes()

# Create some test data.
im = "/app/sample_images/image1.jpg"
im1 = convert_image_to_byte_array(im)

image1_data = im1  # b"\x01\x02\x03\x04\x05\x06"
image1_width = 1920
image1_height = 1080
image1_num_bounding_boxes = 2
image1_data_array = (ctypes.c_ubyte * len(image1_data)).from_buffer_copy(image1_data)

# im = "/app/images/image2.jpg"
# im2 = convert_image_to_byte_array(im)

# image2_data = im2 # b"\x07\x08\x09\x10\x11\x12"
# image2_width = 640
# image2_height = 360
# image2_num_bounding_boxes = 1
# image2_data_array = (ctypes.c_ubyte * len(image2_data)).from_buffer_copy(image2_data)

# im = "/app/images/image3.jpg"
# im3 = convert_image_to_byte_array(im)

# image3_data = im3 # b"\x07\x08\x09\x10\x11\x12"
# image3_width = 640
# image3_height = 360
# image3_num_bounding_boxes = 1
# image3_data_array = (ctypes.c_ubyte * len(image3_data)).from_buffer_copy(image3_data)

## Define the bounding boxes for the image.
# image1 -> [1661,305 1770, 535] [530, 735, 610, 925] [1380, 120, 1420, 278]  
bounding_box1 = BoundingBox(bbox_id = 1, tlx=1661, tly=305, brx=1770, bry=535)
# bounding_box1 = BoundingBox(bbox_id = 1, tlx=1700, tly=400, brx=1731, bry=430)
bounding_box2 = BoundingBox(bbox_id = 2, tlx=530,  tly=735, brx=610,  bry=925)
# bounding_box3 = BoundingBox(bbox_id = 3, tlx=1380, tly=120, brx=1420, bry=278)

# image2 -> 460,58 550, 210
# bounding_box4 = BoundingBox(bbox_id = 4, tlx=460,  tly=58,  brx=550,  bry=210)
## image3 -> 390,18 465,195
# bounding_box5 = BoundingBox(bbox_id = 5, tlx=390,  tly=18,  brx=465,  bry=195)

image1_bounding_boxes = (BoundingBox * image1_num_bounding_boxes)(
    bounding_box1, bounding_box2,
)


# image2_bounding_boxes = (BoundingBox * image2_num_bounding_boxes)(
#     bounding_box4,
# )

# image3_bounding_boxes = (BoundingBox * image3_num_bounding_boxes)(
#     bounding_box5,
# )

full_img_shape = {}
full_img_shape["width"] = image1_width
full_img_shape["height"] = image1_height

box_coordinates = {}
box_coordinates["tlx"] = bounding_box1.tlx
box_coordinates["tly"] = bounding_box1.tly
box_coordinates["brx"] = bounding_box1.brx
box_coordinates["bry"] = bounding_box1.bry

pred_boxes = [ ["person", [bounding_box1.tlx, bounding_box1.tly, bounding_box1.brx, bounding_box1.bry]], 
               ["person", [bounding_box2.tlx, bounding_box2.tly, bounding_box2.brx, bounding_box2.bry]]
            ]
pred_box_index = 0
extended_box, new_shape, padding, person_box, scale = preprocess.get_box_info((full_img_shape["height"], full_img_shape["width"]), pred_boxes, pred_box_index)

print(f"Context area boundary : {extended_box}")
print(f"Scale: {scale}")
print(f"Resized image coord: {person_box}")

crop_metadata1 = Metdata_for_Crop(tlx=extended_box[0],
                                tly=extended_box[1],
                                brx=extended_box[2],
                                bry=extended_box[3],
                                width=full_img_shape["width"],
                                height=full_img_shape["height"] )

resize_metadata1 = Metdata_for_Resize(tlx=person_box[0],  
                                    tly=person_box[1],  
                                    brx=person_box[2],  
                                    bry=person_box[3], 
                                    scale=scale)

full_img_shape = {}
full_img_shape["width"] = image1_width
full_img_shape["height"] = image1_height

box_coordinates = {}
box_coordinates["tlx"] = bounding_box2.tlx
box_coordinates["tly"] = bounding_box2.tly
box_coordinates["brx"] = bounding_box2.brx
box_coordinates["bry"] = bounding_box2.bry

pred_boxes = [ ["person", [bounding_box1.tlx, bounding_box1.tly, bounding_box1.brx, bounding_box1.bry]], 
               ["person", [bounding_box2.tlx, bounding_box2.tly, bounding_box2.brx, bounding_box2.bry]]
            ]
pred_box_index = 1
extended_box, new_shape, padding, person_box, scale = preprocess.get_box_info((full_img_shape["height"], full_img_shape["width"]), pred_boxes, pred_box_index)

print(f"Context area boundary : {extended_box}")
print(f"Scale: {scale}")
print(f"Resized image coord: {person_box}")

crop_metadata2 = Metdata_for_Crop(tlx=extended_box[0],
                                tly=extended_box[1],
                                brx=extended_box[2],
                                bry=extended_box[3],
                                width=full_img_shape["width"],
                                height=full_img_shape["height"] )

resize_metadata2 = Metdata_for_Resize(tlx=person_box[0],  
                                    tly=person_box[1],  
                                    brx=person_box[2],  
                                    bry=person_box[3], 
                                    scale=scale)


image1_metadata_for_crop = (Metdata_for_Crop * image1_num_bounding_boxes)(
    crop_metadata1, crop_metadata2,
)
image1_metadata_for_resize = (Metdata_for_Resize * image1_num_bounding_boxes)(
    resize_metadata1, resize_metadata2
)
# image1_metadata_for_mask = (Metdata_for_Mask * image1_num_bounding_boxes)(
#     mask_metadata1, mask_metadata2
# )

# Create the ImageWithBoundingBoxes structure.
image1 = ImageWithBoundingBoxes(
    image_id = 0,
    image_data=image1_data_array,
    image_width=image1_width,
    image_height=image1_height,
    num_bounding_boxes=image1_num_bounding_boxes,
    bounding_boxes=image1_bounding_boxes,
    crop_metadata=image1_metadata_for_crop,
    # mask_metadata=image1_metadata_for_mask,
    resize_metadata=image1_metadata_for_resize
)

# image2 = ImageWithBoundingBoxes(
#     image_id = 2,
#     image_data=image2_data_array,
#     image_width=image2_width,
#     image_height=image2_height,
#     num_bounding_boxes=image2_num_bounding_boxes,
#     bounding_boxes=image2_bounding_boxes
# )


# image3 = ImageWithBoundingBoxes(
#     image_id = 3,
#     image_data=image3_data_array,
#     image_width=image3_width,
#     image_height=image3_height,
#     num_bounding_boxes=image3_num_bounding_boxes,
#     bounding_boxes=image3_bounding_boxes
# )

# Pass the ImageWithBoundingBoxes structure to the C function.
print("Adding images")
result = add_images_with_bounding_boxes(ctypes.byref(image1),0)
if(result):
    print("Added a image with boxes")
# result = add_images_with_bounding_boxes(ctypes.byref(image2),1)
# if(result):
#     print("Added a image with boxes")
# result = add_images_with_bounding_boxes(ctypes.byref(image3),2)
# if(result):
#     print("Added a image with boxes")

TotalSize = 2

# Inferencing
print("Define the output shape")
outdata = np.zeros((TotalSize, OutputChannel), dtype=np.float32)

meta_ids = np.zeros((TotalSize,2), dtype=np.float32)

# Output data
lib.do_inference.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
]
lib.do_inference.restype = ctypes.c_int

print("Inference")
lib.do_inference(outdata, meta_ids)


print("Output:")
print(outdata)

print("Meta ids:")
print(meta_ids)

# Segmentation model utils
# print("Saving:")
# COLOURS = [
#         (0, 0, 0),  # Black
#         (230, 159, 0),  # Orange
#         (86, 180, 233),  # Sky blue
#         (0, 158, 115),  # Bluish green
#         (240, 228, 66),  # yellow
#         (0, 114, 178),  # blue
#         (213, 94, 0),  # vermillion
#         (204, 121, 167),  # reddish purple
# ]
# for idx in range(0,TotalSize):
#     single_outdata = outdata[idx,:]
#     single_outdata = single_outdata.reshape(InputH, InputW)
    
#     pr_image = Image.new("RGB", (InputW, InputH))
#     pr_image_arr = np.array(pr_image)

#     for cls_idx in range(4):
#         cls_pixels = np.where(single_outdata[:, :] == cls_idx)
#         if cls_idx == 0:
#             pr_image_arr[cls_pixels] = COLOURS[0]  # black
#         elif cls_idx == 1:
#             pr_image_arr[cls_pixels] = COLOURS[6]  # orange
#         elif cls_idx == 2:
#             pr_image_arr[cls_pixels] = COLOURS[3]  # bluish green
#         elif cls_idx == 3:
#             pr_image_arr[cls_pixels] = COLOURS[5]  # blue

#     Image.fromarray(pr_image_arr).save('/app/output'+str(idx)+'.jpg')
###
lib.batchInitialize()


# Create some test data.
im = "/app/sample_images/image1.jpg"
im1 = convert_image_to_byte_array(im)

image1_data = im1  # b"\x01\x02\x03\x04\x05\x06"
image1_width = 1920
image1_height = 1080
image1_num_bounding_boxes = 2
image1_data_array = (ctypes.c_ubyte * len(image1_data)).from_buffer_copy(image1_data)

# im = "/app/images/image2.jpg"
# im2 = convert_image_to_byte_array(im)

# image2_data = im2 # b"\x07\x08\x09\x10\x11\x12"
# image2_width = 640
# image2_height = 360
# image2_num_bounding_boxes = 1
# image2_data_array = (ctypes.c_ubyte * len(image2_data)).from_buffer_copy(image2_data)

# im = "/app/images/image3.jpg"
# im3 = convert_image_to_byte_array(im)

# image3_data = im3 # b"\x07\x08\x09\x10\x11\x12"
# image3_width = 640
# image3_height = 360
# image3_num_bounding_boxes = 1
# image3_data_array = (ctypes.c_ubyte * len(image3_data)).from_buffer_copy(image3_data)

## Define the bounding boxes for the image.
# image1 -> [1661,305 1770, 535] [530, 735, 610, 925] [1380, 120, 1420, 278]  
bounding_box1 = BoundingBox(bbox_id = 1, tlx=1661, tly=305, brx=1770, bry=535)
# bounding_box1 = BoundingBox(bbox_id = 1, tlx=1700, tly=400, brx=1731, bry=430)
bounding_box2 = BoundingBox(bbox_id = 2, tlx=530,  tly=735, brx=610,  bry=925)
# bounding_box3 = BoundingBox(bbox_id = 3, tlx=1380, tly=120, brx=1420, bry=278)

# image2 -> 460,58 550, 210
# bounding_box4 = BoundingBox(bbox_id = 4, tlx=460,  tly=58,  brx=550,  bry=210)
## image3 -> 390,18 465,195
# bounding_box5 = BoundingBox(bbox_id = 5, tlx=390,  tly=18,  brx=465,  bry=195)

image1_bounding_boxes = (BoundingBox * image1_num_bounding_boxes)(
    bounding_box1, bounding_box2,
)


# image2_bounding_boxes = (BoundingBox * image2_num_bounding_boxes)(
#     bounding_box4,
# )

# image3_bounding_boxes = (BoundingBox * image3_num_bounding_boxes)(
#     bounding_box5,
# )

full_img_shape = {}
full_img_shape["width"] = image1_width
full_img_shape["height"] = image1_height

box_coordinates = {}
box_coordinates["tlx"] = bounding_box1.tlx
box_coordinates["tly"] = bounding_box1.tly
box_coordinates["brx"] = bounding_box1.brx
box_coordinates["bry"] = bounding_box1.bry

pred_boxes = [ ["person", [bounding_box1.tlx, bounding_box1.tly, bounding_box1.brx, bounding_box1.bry]], 
               ["person", [bounding_box2.tlx, bounding_box2.tly, bounding_box2.brx, bounding_box2.bry]]
            ]
pred_box_index = 0
extended_box, new_shape, padding, person_box, scale = preprocess.get_box_info((full_img_shape["height"], full_img_shape["width"]), pred_boxes, pred_box_index)

print(f"Context area boundary : {extended_box}")
print(f"Scale: {scale}")
print(f"Resized image coord: {person_box}")

crop_metadata1 = Metdata_for_Crop(tlx=extended_box[0],
                                tly=extended_box[1],
                                brx=extended_box[2],
                                bry=extended_box[3],
                                width=full_img_shape["width"],
                                height=full_img_shape["height"] )

resize_metadata1 = Metdata_for_Resize(tlx=person_box[0],  
                                    tly=person_box[1],  
                                    brx=person_box[2],  
                                    bry=person_box[3], 
                                    scale=scale)

full_img_shape = {}
full_img_shape["width"] = image1_width
full_img_shape["height"] = image1_height

box_coordinates = {}
box_coordinates["tlx"] = bounding_box2.tlx
box_coordinates["tly"] = bounding_box2.tly
box_coordinates["brx"] = bounding_box2.brx
box_coordinates["bry"] = bounding_box2.bry

pred_boxes = [ ["person", [bounding_box1.tlx, bounding_box1.tly, bounding_box1.brx, bounding_box1.bry]], 
               ["person", [bounding_box2.tlx, bounding_box2.tly, bounding_box2.brx, bounding_box2.bry]]
            ]
pred_box_index = 1
extended_box, new_shape, padding, person_box, scale = preprocess.get_box_info((full_img_shape["height"], full_img_shape["width"]), pred_boxes, pred_box_index)

print(f"Context area boundary : {extended_box}")
print(f"Scale: {scale}")
print(f"Resized image coord: {person_box}")

crop_metadata2 = Metdata_for_Crop(tlx=extended_box[0],
                                tly=extended_box[1],
                                brx=extended_box[2],
                                bry=extended_box[3],
                                width=full_img_shape["width"],
                                height=full_img_shape["height"] )

resize_metadata2 = Metdata_for_Resize(tlx=person_box[0],  
                                    tly=person_box[1],  
                                    brx=person_box[2],  
                                    bry=person_box[3], 
                                    scale=scale)


image1_metadata_for_crop = (Metdata_for_Crop * image1_num_bounding_boxes)(
    crop_metadata1, crop_metadata2,
)
image1_metadata_for_resize = (Metdata_for_Resize * image1_num_bounding_boxes)(
    resize_metadata1, resize_metadata2
)
# image1_metadata_for_mask = (Metdata_for_Mask * image1_num_bounding_boxes)(
#     mask_metadata1, mask_metadata2
# )

# Create the ImageWithBoundingBoxes structure.
image1 = ImageWithBoundingBoxes(
    image_id = 0,
    image_data=image1_data_array,
    image_width=image1_width,
    image_height=image1_height,
    num_bounding_boxes=image1_num_bounding_boxes,
    bounding_boxes=image1_bounding_boxes,
    crop_metadata=image1_metadata_for_crop,
    # mask_metadata=image1_metadata_for_mask,
    resize_metadata=image1_metadata_for_resize
)

# image2 = ImageWithBoundingBoxes(
#     image_id = 2,
#     image_data=image2_data_array,
#     image_width=image2_width,
#     image_height=image2_height,
#     num_bounding_boxes=image2_num_bounding_boxes,
#     bounding_boxes=image2_bounding_boxes
# )


# image3 = ImageWithBoundingBoxes(
#     image_id = 3,
#     image_data=image3_data_array,
#     image_width=image3_width,
#     image_height=image3_height,
#     num_bounding_boxes=image3_num_bounding_boxes,
#     bounding_boxes=image3_bounding_boxes
# )

# Pass the ImageWithBoundingBoxes structure to the C function.
print("Adding images")
result = add_images_with_bounding_boxes(ctypes.byref(image1),0)
if(result):
    print("Added a image with boxes")
# result = add_images_with_bounding_boxes(ctypes.byref(image2),1)
# if(result):
#     print("Added a image with boxes")
# result = add_images_with_bounding_boxes(ctypes.byref(image3),2)
# if(result):
#     print("Added a image with boxes")

TotalSize = 2

# Inferencing
print("Define the output shape")
outdata = np.zeros((TotalSize, OutputChannel), dtype=np.float32)

meta_ids = np.zeros((TotalSize,2), dtype=np.float32)

# Output data
lib.do_inference.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
]
lib.do_inference.restype = ctypes.c_int

print("Inference")
lib.do_inference(outdata, meta_ids)


print("Output:")
print(outdata)

print("Meta ids:")
print(meta_ids)
