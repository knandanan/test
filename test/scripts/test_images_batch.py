"""
@author: nitin@uncannyvision.com
This scripts works as sender. Its send API request to the ENTRYPOINT of ovClassifier and receive back response.

command to run classifier container independently:

sudo docker run -it -e RUN_MODE=1 -e NODENAME=perimeter3-headgear_classifier -e ORCHESTRATOR_PORT=5000 -e ORCHESTRATOR_IP=10.16.239.10 -e SERVICE_NAME=headgear_classifier -v /home/uncanny/GenenTech/classifier_testing/mounts/:/app/mounts/ -p 5011:5011 trt_classifier:prod

"""
import sys
import json
import time
import cv2
import numpy as np
import requests
import base64
import json
import os

data_dir = "../data/test_folder"
# data_dir = "../data/LM_folder"

images_dir = os.path.join(data_dir, "single_images")
jsons_dir = os.path.join(data_dir, "jsons")
results_dir = os.path.join(data_dir, "results")

image_files = os.listdir(images_dir)
image_files.sort()


#Template
batch_number = 5
batch_counter = 0
module_data = {
    "moduleData": {
        "dryRun": False,
        "imageCount": 0,
        "imageList": [
        ]
    }
}
module_img_list = []
result_path_list = []
image_counter = 0


for image in image_files:
    json_name = image.replace(".jpeg", ".json")
    # json_name = image.replace(".jpg", ".json")
    
    image_path = os.path.join(images_dir, image)
    json_path  = os.path.join(jsons_dir, json_name)
    result_path= os.path.join(results_dir, json_name)

    image_counter += 1
    DATA = json.load(open(json_path, "r"))

    IMG = cv2.imread(image_path)
    #IMG = cv2.resize(IMG, (640, 480))
    # print(IMG.shape)
    SEND_IMAGE = IMG.flat
    # SEND_IMAGE_STR = (np.array(SEND_IMAGE)).tostring()
    # print(len(SEND_IMAGE_STR), type(SEND_IMAGE_STR))

    IMAGE_STRING = base64.b64encode(cv2.imencode(".bmp", IMG)[1]).decode()
    # print(IMG)
    # print(len(IMAGE_STRING), type(IMAGE_STRING))

    module_data['moduleData']['imageList'].append(DATA['moduleData']['imageList'][0])
    module_img_list.append({"image_string": IMAGE_STRING, "len": len(IMAGE_STRING)})
    result_path_list.append(result_path)
    batch_counter += 1

    print("Image_counter: ", image_counter, " length_image_files: ", len(image_files), " . json_name: ", json_name)
    if batch_counter == batch_number or image_counter == len(image_files):
        batch_counter = 0

        module_data['moduleData']['imageCount'] = len(module_data['moduleData']['imageList'])
        MD = {
            "metadata": module_data,
            "images": module_img_list,
        }

        START = time.time() * 1000
        response = requests.post("http://10.16.239.20:5011/process", json=MD)
        END = time.time() * 1000
        print("time taken END - START,",(END - START),  " msec.")

        print(response.status_code)
        result_json = json.loads(response.content.decode("utf-8"))

        for i in range(len(result_json["moduleData"]["imageList"])):
            rp = result_path_list[i]

            print(result_json["moduleData"]["imageList"][i]["inference"])
            # print(result_json)

            f = open(rp, "w")
            json.dump(result_json["moduleData"]["imageList"][i], f, indent= 4)
            f.close()

        #default values
        module_data['moduleData']['imageList'] = []
        module_data['moduleData']['imageCount'] = 0
        module_img_list = []
        result_path_list = []
