#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # INTERNAL REPO # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
STR_INTERNAL="192.168.1.167    uvDockInternal"
echo "Appending internal's ip in /etc/hosts/"
if ! grep -q "$STR_INTERNAL" /etc/hosts; then
    echo $STR_INTERNAL | sudo tee -a /etc/hosts
else
    echo "Entry already exists in /etc/hosts"
fi

INTERNAL_DIR="/etc/docker/certs.d/uvDockInternal:8899"
echo "Creating internal directory at $INTERNAL_DIR"
if ! sudo test -d "$INTERNAL_DIR"; then
    sudo mkdir -p $INTERNAL_DIR
else
    echo "Directory already exists"
fi

INTERNAL_CRT_PATH="/etc/docker/certs.d/uvDockInternal:8899/uvDockInternal.crt"
echo "Downloading crt file at $INTERNAL_CRT_PATH"
if ! sudo test -f "$INTERNAL_CRT_PATH"; then
    sudo curl -L https://www.dropbox.com/s/wtu876wfdbfpc9v/uvDockInternal.crt?dl=1 -o $INTERNAL_CRT_PATH
else
    echo "crt file already exists."
fi

STR_BASE="192.168.1.167    uvDockBase"
echo "Appending base's ip in /etc/hosts/"
if ! grep -q "$STR_BASE" /etc/hosts; then
    echo $STR_BASE | sudo tee -a /etc/hosts
else
    echo "Entry already exists in /etc/hosts"
fi

BASE_DIR="/etc/docker/certs.d/uvDockBase:3333/"
echo "Creating base directory at $BASE_DIR"
if ! sudo test -d $BASE_DIR; then
    sudo mkdir -p $BASE_DIR
else
    echo "Directory already exists"
fi

BASE_CRT_PATH="/etc/docker/certs.d/uvDockBase:3333/uvDockBase.crt"
echo "Downloading crt file at $BASE_CRT_PATH"
if ! sudo test -f $BASE_CRT_PATH; then
    sudo curl -L https://www.dropbox.com/s/9q5nhbl33ck3p8c/uvDockBase.crt?dl=1 -o $BASE_CRT_PATH
else
    echo "crt file already exists."
fi
