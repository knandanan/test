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
images_dir = os.path.join(data_dir, "single_images")
jsons_dir = os.path.join(data_dir, "jsons")
results_dir = os.path.join(data_dir, "results")

image_files = os.listdir(images_dir)
image_files.sort()

for image in image_files:
	json_name = image.replace(".jpeg", ".json")
	image_path = os.path.join(images_dir, image)
	json_path  = os.path.join(jsons_dir, json_name)
	result_path= os.path.join(results_dir, json_name)
	

	DATA = json.load(open(json_path, "r"))
	
	IMG = cv2.imread(image_path)
	#IMG = cv2.resize(IMG, (640, 480))
	print(IMG.shape)
	SEND_IMAGE = IMG.flat
	# SEND_IMAGE_STR = (np.array(SEND_IMAGE)).tostring()
	# print(len(SEND_IMAGE_STR), type(SEND_IMAGE_STR))
	
	IMAGE_STRING = base64.b64encode(cv2.imencode(".bmp", IMG)[1]).decode()
	# print(IMG)
	print(len(IMAGE_STRING), type(IMAGE_STRING))
	for request_ in range(1):
	    print("Sending request %s …" % request_)
	    #DATA["moduleData"]["timeLogs"][-1]["recvTime"] = int(time.time() * 1000)
	    MD = {
	        "metadata": DATA,
	        "images": [{"image_string": IMAGE_STRING, "len": len(IMAGE_STRING)}],
	    }
	    # response = requests.post('https://10.16.239.1:5011/ovc_input', json=data_)
	    START = time.time() * 1000
	    response = requests.post("http://10.16.239.20:5011/process", json=MD)
	    END = time.time() * 1000
	    print("time taken for request ", request_, ": ", END - START, " msec.")
	    print(response.status_code)
	    result_json = json.loads(response.content.decode("utf-8"))
	    print(result_json["moduleData"]["imageList"][0]["inference"])
	    # print("Send completed")
	    f = open(result_path, "w")
	    json.dump(result_json, f, indent= 4)
	    f.close()
