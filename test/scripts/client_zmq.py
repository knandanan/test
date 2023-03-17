'''
@author: nitin@uncannyvision.com
This scripts is working as client it sends input directly to ovClassifier INPUT PORT and receive response from OUTPUT PORT
'''
import sys
import json
import time
import cv2
import zmq
import numpy as np
#  Socket to talk to server
CONTEXT = zmq.Context()
DATA = json.load(open(sys.argv[1], "r"))
print("Connecting to hello world server…")
SOCKET = CONTEXT.socket(zmq.PUSH)
SOCKET.bind("tcp://10.16.239.1:1233")

IMG = cv2.imread(sys.argv[2])
# IMG = cv2.resize(IMG, (640, 480))
print(IMG.shape)
SEND_IMAGE = IMG.flat
SEND_IMAGE_STR = np.array(SEND_IMAGE).tostring()
# print(IMG)
SOCKET_OUT = CONTEXT.socket(zmq.PULL)
SOCKET_OUT.connect("tcp://10.16.239.20:49152")
SOCKET_MON = CONTEXT.socket(zmq.PULL)
SOCKET_MON.connect("tcp://10.16.239.20:1235")

#  Do 10 requests, waiting each time for a response
for request in range(1):
    print("Sending request %s …" % request)
    DATA["moduleData"]["timeLogs"][-1]["recvTime"] = int(time.time()*1000)
    MD = json.dumps(DATA)
    START = time.time()*1000
    SOCKET.send_multipart([MD.encode("utf-8"), SEND_IMAGE_STR])
    #print("Send completed")
    #  Get the reply.
    RESPONSE = SOCKET_OUT.recv_multipart()
    END = time.time()*1000
    print("Time Taken to process request "+str(request)+ ": "+str(END-START)+" msec.")
    RESULT_JSON = json.loads(RESPONSE[0].decode('utf-8'))
    print(RESULT_JSON["moduleData"]["imageList"][0]["inference"])
