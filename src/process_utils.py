import numpy as np
import base64
import os

# For RUN_MODE
import cv2

# For dependencies
from ctype_utils import *

#For debug
from PIL import Image

## helper
def check_run(rules_activation, rules, service_name):
    rule_dict = {r['ID']: r for r in rules}
    return any(
        rule_dict[r]['advanced'].get(service_name, {}).get('enable')
        for ra in rules_activation
        for r in ra['rules']
    )
##
def process_batchInit():
    lib.batchInitialize()

def process_InitGPUMemory(batchSize, _width, _height, InputChannel, OutputChannel, debug_level, debug_image):
    lib.InitializeGPUMemory(batchSize, _width, _height, InputChannel, OutputChannel, debug_level, debug_image)

def process_loadModel(_onnxpath, _enginepath, model_type, batchSize):
    _enginepath_left = _enginepath.split('.trt')[0]
    _enginepath = _enginepath_left + '_{}_{}.trt'.format(batchSize,model_type)
    _enginepath = _enginepath.encode('utf-8')
    if not os.path.exists(_enginepath):
        lib.load_onnx(_onnxpath, _enginepath, model_type)
    lib.load_engine(_enginepath)

def process_msg_images(input_json, mqData, img_list_tuples, objYaml, RUN_MODE): 
    dict_of_images = {}

    # get image count
    no_of_images = input_json["moduleData"]["imageCount"]
    # get jsons
    jsons = input_json["moduleData"]["imageList"]
    # For each image
    for i in range(no_of_images):
        one_json = jsons[i]
        if not RUN_MODE:
            imagedata = mqData[0].data[i + 1]
            image_string = imagedata[: mqData[0].length[i + 1]]
            buf = memoryview(image_string)
            jpg_as_np = np.frombuffer(buf, dtype="uint8")
            jpg_as_np = jpg_as_np.reshape([one_json["imageMeta"]["height"],one_json["imageMeta"]["width"],3,])
            # jpg_as_np = cv2.convertScaleAbs(jpg_as_np)
            height = one_json["imageMeta"]["height"]
            width = one_json["imageMeta"]["width"]
        else:
            image_string = img_list_tuples[i]["image_string"]
            jpg_original = base64.b64decode(image_string)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            jpg_as_np = cv2.imdecode(jpg_as_np, flags=1)
            height,width,_ = jpg_as_np.shape
        
        dict_image_data = { "image_string":image_string, "np_data" : jpg_as_np, "len": len(image_string), "height": height, "width": width }
        dict_of_images[i] = dict_image_data
    return dict_of_images, no_of_images

def process_msg_json(input_json, objYaml, RUN_MODE, SERVICE_NAME, LOGGER):
    dict_of_json = {}

    # get image count
    no_of_images = input_json["moduleData"]["imageCount"]
    # get jsons
    jsons = input_json["moduleData"]["imageList"]
    # For each image
    for i in range(no_of_images):
        one_json = jsons[i]
        dict_of_json[i] = []
        process_image = False

        roi_process_image = False
        if ("rulesActivation" in one_json["imageMeta"] and one_json["imageMeta"]["rulesActivation"] is not None):
            roi_process_image = check_run(  
                                            one_json["imageMeta"]["rulesActivation"],
                                            one_json["imageMeta"]["rules"],
                                            SERVICE_NAME,
                                        )

        if "detections" in one_json["inference"] and roi_process_image:
            for className in objYaml.values["bboxType"]:
                if "detections" in one_json["inference"]:
                    map_id = -1
                    if className in one_json["inference"]["detections"]:
                        for classDetections in one_json["inference"]["detections"][className]:
                            map_id += 1
                            if "stationary" in classDetections:
                                if classDetections["stationary"] != "stationary":
                                    continue
                            if "coordinates" in classDetections and "id" in classDetections:
                                coord = classDetections["coordinates"]
                                bbox_id = classDetections["id"]
                                tlx = int(coord["xmin"])
                                tly = int(coord["ymin"])
                                brx = int(coord["xmax"])
                                bry = int(coord["ymax"])
                                dict_json_data = {"bbox_id": bbox_id, "map_id": map_id, "tlx": tlx, "tly": tly, "brx": brx, "bry":bry, "class":className } 
                                if i in dict_of_json:
                                    dict_of_json[i].append(dict_json_data)
                                else:
                                    dict_of_json[i] = []
                                    dict_of_json[i].append(dict_json_data)
    return dict_of_json

def prepare_pred_boxes(one_json):
    '''
    prepare_pred_boxes
    '''

    pred_boxes = []
    start_index_dict = {}
    start_index = 0
    if "detections" in one_json["inference"]:
        for className in one_json["inference"]["detections"].keys():
            if className not in start_index_dict:
                start_index_dict[className] = {}
            start_index_dict[className]['startIndex'] = start_index

            for classDetections in one_json["inference"]["detections"][className]:
                if "coordinates" in classDetections:
                    obj_boxes = []
                    obj_boxes.append(className)

                    coord = classDetections["coordinates"]
                    obj_coor = []
                    obj_coor.append(int(coord["xmin"]))
                    obj_coor.append(int(coord["ymin"]))
                    obj_coor.append(int(coord["xmax"]))
                    obj_coor.append(int(coord["ymax"]))

                    obj_boxes.append(obj_coor)
                    pred_boxes.append(obj_boxes)
                    start_index += 1
    
    # pred_boxes: [[class_id, [tlx, tly, brx, bry]], [class_id, [tlx, tly, brx, bry]], [class_id, [tlx, tly, brx, bry]]]
    return pred_boxes, start_index_dict



def process_preprocess(LOGGER, input_json, dict_of_images, dict_of_json, preprocess):

    TotalSize = 0
    print(len(dict_of_images))
    print(len(dict_of_json))
    mapper_totalsize_and_object_id = {}
    for image_id in dict_of_images:
        dict_image_data = dict_of_images[image_id]
        if image_id in dict_of_json:
            list_of_json_data  = dict_of_json[image_id]

            if image_id not in mapper_totalsize_and_object_id:
                mapper_totalsize_and_object_id[image_id] = {}

            #call to prepare pred_boxes list for all the classes
            one_json = input_json["moduleData"]["imageList"][image_id]
            pred_boxes, start_index_dict = prepare_pred_boxes(one_json)

            full_img_shape = {}
            full_img_shape["width"] = dict_image_data["width"]
            full_img_shape["height"] = dict_image_data["height"]

            data_array = dict_image_data["np_data"].tobytes()
            
            image_data_array = (ctypes.c_ubyte * len(data_array)).from_buffer_copy(data_array)

            image_bounding_boxes = []
            image_metadata_for_crop = []
            image_metadata_for_resize = []

            for index, dict_json_data in enumerate(list_of_json_data):
                box_coordinates = {}
                box_coordinates["tlx"] = dict_json_data["tlx"]
                box_coordinates["tly"] = dict_json_data["tly"]
                box_coordinates["brx"] = dict_json_data["brx"]
                box_coordinates["bry"] = dict_json_data["bry"]
                
                bounding_box = BoundingBox(bbox_id = dict_json_data["bbox_id"],
                                        tlx=dict_json_data["tlx"],
                                        tly=dict_json_data["tly"],
                                        brx=dict_json_data["brx"],
                                        bry=dict_json_data["bry"])

                pred_box_index = start_index_dict[dict_json_data['class']]['startIndex'] + dict_json_data["map_id"]
                LOGGER.debug("Cross check current bbox: {}  pred_bbox: {}".format(dict_json_data, pred_boxes[pred_box_index]))
                extended_box, new_shape, padding, person_box, scale = preprocess.get_box_info((full_img_shape["height"], full_img_shape["width"]), pred_boxes, pred_box_index)
                LOGGER.debug("extended_box {}  and person_box: {}".format(extended_box, person_box))

                if extended_box is None:
                    continue

                if dict_json_data['class'] not in mapper_totalsize_and_object_id[image_id]:
                    mapper_totalsize_and_object_id[image_id][dict_json_data['class']] = {}
                mapper_totalsize_and_object_id[image_id][dict_json_data['class']][TotalSize] = dict_json_data["bbox_id"]
                TotalSize = TotalSize + 1

                crop_metadata = Metdata_for_Crop(tlx=extended_box[0],
                                                tly=extended_box[1],
                                                brx=extended_box[2],
                                                bry=extended_box[3],
                                                width=full_img_shape["width"],
                                                height=full_img_shape["height"] )

                resize_metadata = Metdata_for_Resize(tlx=person_box[0],  
                                                    tly=person_box[1],  
                                                    brx=person_box[2],  
                                                    bry=person_box[3], 
                                                    scale=scale)

                image_bounding_boxes.append(bounding_box)
                image_metadata_for_crop.append(crop_metadata)
                image_metadata_for_resize.append(resize_metadata)

            #convert python list type to ctype array
            image_bounding_boxes_ctype_obj = (BoundingBox * len(image_bounding_boxes))(*image_bounding_boxes)
            image_metadata_for_crop_ctype_obj = (Metdata_for_Crop * len(image_metadata_for_crop))(*image_metadata_for_crop)
            image_metadata_for_resize_ctype_obj = (Metdata_for_Resize * len(image_metadata_for_resize))(*image_metadata_for_resize)

            # Create the ImageWithBoundingBoxes structure.
            image = ImageWithBoundingBoxes(
                image_id = image_id,
                image_data=image_data_array,
                image_width=full_img_shape["width"],
                image_height=full_img_shape["height"],
                num_bounding_boxes=len(image_bounding_boxes_ctype_obj),
                bounding_boxes= image_bounding_boxes_ctype_obj,
                crop_metadata=  image_metadata_for_crop_ctype_obj,
                resize_metadata=image_metadata_for_resize_ctype_obj,
            )

            start = time.time()
            result = add_images_with_bounding_boxes(ctypes.byref(image),image_id)
            end = time.time()

            print(f"Added image {result} : {end - start} image_id: {image_id}")
    return TotalSize, mapper_totalsize_and_object_id

def process_inference(outdata, meta_ids):
    lib.do_inference(outdata, meta_ids)

def debug_image_segmentation(TotalSize, outdata, InputH, InputW):
    COLOURS = [
        (0, 0, 0),  # Black
        (230, 159, 0),  # Orange
        (86, 180, 233),  # Sky blue
        (0, 158, 115),  # Bluish green
        (240, 228, 66),  # yellow
        (0, 114, 178),  # blue
        (213, 94, 0),  # vermillion
        (204, 121, 167),  # reddish purple
    ]
    for idx in range(0,TotalSize):
        single_outdata = outdata[idx,:]
        single_outdata = single_outdata.reshape(InputH, InputW)
        
        pr_image = Image.new("RGB", (InputW, InputH))
        pr_image_arr = np.array(pr_image)

        for cls_idx in range(4):
            cls_pixels = np.where(single_outdata[:, :] == cls_idx)
            if cls_idx == 0:
                pr_image_arr[cls_pixels] = COLOURS[0]  # black
            elif cls_idx == 1:
                pr_image_arr[cls_pixels] = COLOURS[6]  # orange
            elif cls_idx == 2:
                pr_image_arr[cls_pixels] = COLOURS[3]  # bluish green
            elif cls_idx == 3:
                pr_image_arr[cls_pixels] = COLOURS[5]  # blue
        Image.fromarray(pr_image_arr).save('/app/output'+str(idx)+'.jpg')

def process_postprocess(outdata, meta_ids, TotalSize, mapper_totalsize_and_object_id,input_json, objYaml, labels_map):

    for idx in range(0,TotalSize):
        single_outdata = outdata[idx,:]
        class_index = np.argmax(single_outdata)
        confidence_value = single_outdata[class_index]
        pred_class_name = labels_map[class_index]

        meta_id = meta_ids[idx,:]
        image_id, bbox_id = int(meta_id[0]), int(meta_id[1])
        one_json = input_json["moduleData"]["imageList"][image_id]['inference']['detections']
        for className in objYaml.values["bboxType"]:
            if className not in mapper_totalsize_and_object_id[image_id] and className not in one_json:
                continue

            mapped_bbox_id = mapper_totalsize_and_object_id[image_id][className][idx]
            classDetection = one_json[className]
            detection_to_update = next((x for x in classDetection if x['id'] == mapped_bbox_id), None)
            if detection_to_update is not None:
                if 'attributes' not in detection_to_update:
                    detection_to_update['attributes'] = {}
                    detection_to_update['attributes']['fptp'] = {}

                if 'fptp' not in detection_to_update['attributes']:
                    detection_to_update['attributes']['fptp'] = {}

                detection_to_update['attributes']['fptp'] = {
                    "class": pred_class_name,
                    "confidence": confidence_value.tolist()
                }

    return input_json
