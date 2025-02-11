#!/bin/bash

echo -n "Project mount path: "
read -r PROJECT_PATH
echo -n "Datasets mount path: "
read -r DATASETS_PATH
echo -n "Image tag: "
read -r tag

docker create --name reconet --ipc host -it --gpus all \
    -v $PROJECT_PATH:/root/project \
    -v $DATASETS_PATH:/root/datasets \
    $tag