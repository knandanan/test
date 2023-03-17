#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/opt/SimpleMQ/bindings
exec /app/run_classifierv2.py -externalconfig=mounts/config/externalconfig/externalconfig.yaml