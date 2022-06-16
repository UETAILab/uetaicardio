#!/bin/bash
echo "Building docker"
docker build -t echodeploy .

dirpath=/tmp/echo_deploy/

docker run -it\
    --gpus all\
    --mount type=bind,source="${dirpath}"/deploy_weight,target=/deploy_weight \
    -p 0.0.0.0:8089:8080/tcp \
     echodeploy:latest
