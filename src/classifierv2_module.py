#!/usr/bin/python3
"""
@author: prince@een.com
main code for signature module
"""
#################
# Template imports for api
import logging
import time
import argparse
import json
import threading
import os
import requests
import numpy as np
import sys
import math
import cv2
import random
import copy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.path.insert(0, '/opt/SimpleMQ/bindings')

from mqcsimplemq import *
from mqConnections import MqConnections
from configReader import YamlReader
from observerFile import observerChange
from extended_apilayer import ExtApiLayer, add_method
SCRIPT = "classifierv2_module.py"
#################

#################
# process utils 
from process_utils import *
##################

#################
# Module specifc
import code_deploy_v6 as preprocess
##################

LOGGER = None
NODENAME = os.environ["NODENAME"]
ORCHESTRATOR_PORT = os.environ["ORCHESTRATOR_PORT"]
ORCHESTRATOR_IP = os.environ["ORCHESTRATOR_IP"]
SERVICE_NAME = os.environ["SERVICE_NAME"]

# For production
monitoring_ip = str(os.environ.get("MONITORING_IP", "127.0.0.1"))
monitoring_port = int(os.environ.get("MONITORING_PORT", 5011))
api_port = int(os.environ.get("API_PORT", 5011))

try:
    RUN_MODE = int(os.environ["RUN_MODE"])
except KeyError:
    RUN_MODE = 0

MD = None
IMG_STRING_TUPLES = None

API_REQUEST_ADD_QUEUE = []
API_RESPONSE = {}

LOCK = threading.Lock()

@add_method(ExtApiLayer)
def process_input(md, img_string_tuples):
    global API_REQUEST_ADD_QUEUE, API_RESPONSE
    LOCK.acquire()
    unique_id = random.random()
    API_REQUEST_ADD_QUEUE.append([md, img_string_tuples, unique_id])
    LOCK.release()

    while unique_id not in API_RESPONSE:
        time.sleep(.001)

    return API_RESPONSE.pop(unique_id, {})
##

if __name__ == "__main__":
    main()

def main():
    global LOGGER, API_REQUEST_ADD_QUEUE, API_RESPONSE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-externalconfig', help='External config file path')
    args = parser.parse_args()
    if not args.externalconfig:
        return
    external_config_path = args.externalconfig.strip()

    objYaml = YamlReader()
    objYaml.yamlReader("config/internalconfig.yaml", "classifierv2")
    objYaml.yamlReader(external_config_path, SERVICE_NAME)
    initial_external_config_hash = objYaml.getHash(external_config_path, SERVICE_NAME)
    apil = ExtApiLayer(name=NODENAME, port=5011, max_hung_time=30, log_level=objYaml.values["debug_level"],
                       config_manager_addr="monitoring_engine:5011")
    if RUN_MODE != 1:
        apil = ExtApiLayer(name=NODENAME, port=api_port, max_hung_time=30,\
            log_level=objYaml.values["debug_level"],\
            config_manager_addr=monitoring_ip+":"+str(monitoring_port))
    else:
        apil = ExtApiLayer(name=NODENAME, port=5011, max_hung_time=5,\
            log_level=objYaml.values["debug_level"])

    apil.run()
    LOGGER = apil.logger
    LOGGER.info(
        json.dumps({"NODENAME": NODENAME}).encode('utf-8'),
        method="main()",
        script=SCRIPT
    )

    LOGGER.info(
        json.dumps({"msg": "Read internal and externalconfig"}).encode('utf-8'),
        method="main()",
        script=SCRIPT
    )
    
    LOGGER.debug(
        json.dumps({"msg": str(objYaml.values)}).encode('utf-8'),
        method="main()",
        script=SCRIPT
    )

    ## Load the labels file
    labels_map = None
    if objYaml.values["labels"]:
        with open(objYaml.values["labels"], "r") as f:
            labels_map = [x.split(sep=" ", maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None


    batchSize = objYaml.values["batch_size"]
    model_type = 1 if objYaml.values["model_type"] == 'FP16' else 0  
    
    _width = objYaml.values["width"]
    _height = objYaml.values["height"]
    _onnxpath = objYaml.values["onnxmodel"]
    _enginepath = objYaml.values["trtengine"]

    InputChannel = 3
    if 'input_channel' in objYaml.values:
        InputChannel = objYaml.values["input_channel"]
    
    debug_image = 0
    if 'debug_image' in objYaml.values:
        debug_image = objYaml.values["debug_image"]

    OutputChannel = len(labels_map)

    # Initialize the trt
    process_InitGPUMemory(batchSize, _width, _height, InputChannel, OutputChannel, int(objYaml.values["debug_level"]), debug_image)
    
    #Initialize and clear the batch
    process_batchInit()

    # Model builders
    if not os.path.exists(_onnxpath):
        LOGGER.error(
        json.dumps({"msg": "Onnx file doesn't exist."}).encode('utf-8'),
        method="main()",
        script=SCRIPT,
        )
        exit(1)

    process_loadModel(_onnxpath.encode('utf-8'), _enginepath, model_type, batchSize)
    
    LOGGER.info(
        json.dumps({"msg": "Model loading completed"}).encode('utf-8'),
        method="main()",
        script=SCRIPT,
    )

    if 'frame_debug_interval' in objYaml.values:
        frame_debug_interval = int(objYaml.values.get("frame_debug_interval"))
    else:
        frame_debug_interval = 1000
    
    frame_debug_sum = 0
    frame_debug_counter = 0

    if not RUN_MODE:
        LOGGER.debug(
            json.dumps({"ORCHESTRATOR_IP": ORCHESTRATOR_IP}).encode('utf-8'),
            method="main()",
            script=SCRIPT,
        )
        LOGGER.debug(
            json.dumps({"ORCHESTRATOR_PORT": ORCHESTRATOR_PORT}).encode('utf-8'),
            method="main()",
            script=SCRIPT,
        )
        ports = ["Input", "Output", "Monitoring"]
        objMQ = MqConnections(ports, LOGGER)
        METRIC_INTERVAL = 10000
        objMQ.setupMQAndConnection(
            NODENAME,
            ip=ORCHESTRATOR_IP,
            orchestrator_port=ORCHESTRATOR_PORT,
            METRIC_INTERVAL=10000,
        )


    if not RUN_MODE:
        observerChange(
            external_config_path,
            objMQ,
            objYaml,
            SERVICE_NAME,
            initial_external_config_hash,
            LOGGER,
        )
    
    while True:
        dry_run = False
        
        LOGGER.debug(
            json.dumps({"msg": "inside loop"}).encode('utf-8'),
            method="main()",
            script=SCRIPT
        )
        mqData = None
        array_of_tuple = []
        img_list_tuples = []

        if not RUN_MODE:
            mqData = NewSimpleMQData(1)
            LOGGER.info(
                json.dumps({"msg": "SimpleMQ msg JSON is loaded"}).encode('utf-8'),
                method="main()",
                script=SCRIPT
            )

            receive_status = simpleMQRecv(
                objMQ.g_mqAppCtxt.mqCtxt, objMQ.ports["Input"], ctypes.byref(mqData)
            )
            LOGGER.info(
                json.dumps({"msg": "Receive Status(" + str(NODENAME) + ") is " + str(receive_status)}).encode('utf-8'),
                method="main()",
                script=SCRIPT,
            )
            jsondata = mqData[0].data[0]
            json_length = mqData[0].length[0]
            json_string = jsondata[:json_length]
            input_json = json.loads(json_string.decode("utf-8"))
            LOGGER.debug(input_json)
        else:
            request_data = None
            try:
                request_data = API_REQUEST_ADD_QUEUE.pop(0)
            except:
                pass
            if request_data is None:
                LOGGER.info(
                    json.dumps({"msg":  "Waiting for API request going to sleep for 1 sec."}).encode('utf-8'),
                    method="main()",
                    script=SCRIPT,
                )
                time.sleep(1)
                continue
            else:
                MD = request_data[0]
                IMG_STRING_TUPLES = request_data[1]
                LOGGER.info(
                    json.dumps({"msg":  "Receive Status from API for" + NODENAME + ")"}).encode('utf-8'),
                    method="main()",
                    script=SCRIPT,
                )
                input_json = MD
                img_list_tuples = IMG_STRING_TUPLES
                unique_id = request_data[2]
            
        apil.ping()
        start = int(time.time() * 1000)
        if "dryRun" in input_json["moduleData"]:
            if input_json["moduleData"]["dryRun"] == True:
                dry_run = True
        if dry_run:
            LOGGER.debug(
                json.dumps({"msg":  "Dry run on"}).encode('utf-8'),
                method="main()",
                script=SCRIPT,
            )
            end = int(time.time() * 1000)
            input_json["moduleData"]["timeLogs"].append(
                {"recvTime": start, "sendTime": end, "moduleName": NODENAME}
            )
            _final_send_json = json.dumps(input_json)
            LOGGER.debug(
                str(_final_send_json), method="main()", script=SCRIPT
            )
            _final_send_json = _final_send_json.encode("utf-8")
            if not RUN_MODE:
                image_sendstatus = simpleMQSendBuffers(
                    objMQ.g_mqAppCtxt.mqCtxt,
                    objMQ.ports["Output"],
                    1,
                    _final_send_json,
                    len(_final_send_json),
                )
                LOGGER.info(
                    json.dumps({"msg": "Send Status(" + NODENAME + ")" + str(image_sendstatus)}).encode('utf-8'),
                    method="main()",
                    script=SCRIPT,
                )
                DeleteSimpleMQData(mqData)
            else:
                LOGGER.info(
                    json.dumps({"msg": "Sending response back to client from ("
                    + NODENAME + ")"}).encode('utf-8'),
                    method="main()",
                    script=SCRIPT,
                )
                API_RESPONSE[unique_id] = _final_send_json 
            continue

        ## Process every message now
        start = int(time.time() * 1000)
        
        dict_of_images, no_of_images = process_msg_images(input_json, mqData, img_list_tuples, objYaml, RUN_MODE)
        dict_of_json = process_msg_json(input_json, objYaml, RUN_MODE, SERVICE_NAME, LOGGER)

        TotalSize, mapper_totalsize_and_object_id = process_preprocess(LOGGER, input_json, dict_of_images, dict_of_json, preprocess)
        
        LOGGER.debug(
            json.dumps({"msg": "Total Bounding boxes to process "
            + str(TotalSize) + ")"}).encode('utf-8'),
            method="main()",
            script=SCRIPT,
        )

        # inferenced results
        outdata =  np.zeros((TotalSize, OutputChannel), dtype=np.float32)
        meta_ids = np.zeros((TotalSize, 2 ),  dtype=np.float32)
        
        process_inference(outdata, meta_ids)

        input_json = process_postprocess(outdata, meta_ids, TotalSize, mapper_totalsize_and_object_id, input_json, objYaml, labels_map)
        
        #Initialize and clear the batch
        process_batchInit()

        end = int(time.time() * 1000)

        if not RUN_MODE:
            DeleteSimpleMQData(mqData)

        images_args = tuple()
        for image_id in dict_of_images:
            tuple_image_data = (dict_of_images[image_id]["image_string"], dict_of_images[image_id]["len"])
            images_args = images_args + tuple_image_data

        frame_debug_counter += 1
        frame_debug_sum += (end-start)

        if(frame_debug_counter == frame_debug_interval):
            mean = round(float(frame_debug_sum)/frame_debug_interval)
            LOGGER.info(
                json.dumps({"Received to send completed in : ": str(mean) }).encode('utf-8'),
                method="main()",
                script=SCRIPT,
            )
            frame_debug_counter = 0
            frame_debug_sum = 0
        
        input_json["moduleData"]["timeLogs"].append(
            {"recvTime": start, "sendTime": end, "moduleName": NODENAME}
        )
        send_json = json.dumps(input_json)
        send_json = send_json.encode("utf-8")
        
        LOGGER.debug(
            str(len(send_json)),
            method="main()",
            script=SCRIPT,
        )
        LOGGER.debug(
            str(len(images_args)),
            method="main()",
            script=SCRIPT,
        )

        if not RUN_MODE:
            image_sendstatus = simpleMQSendBuffers(
                objMQ.g_mqAppCtxt.mqCtxt,
                objMQ.ports["Output"],
                no_of_images + 1,
                send_json,
                len(send_json),
                *images_args
            )
            LOGGER.debug(
                json.dumps({"msg": "Send Status(" + str(NODENAME) + ")" + str(image_sendstatus)}).encode('utf-8'),
                method="main()",
                script="trtClassifier.py",
            )
        else:
            API_RESPONSE[unique_id] = send_json
            LOGGER.debug(json.dumps({"msg": "Sending response back to client from("
                + NODENAME + ")"}).encode('utf-8'),
                method="main()",
                script="trtClassifier.py",
            )
    observer.join()