#!/bin/sh
#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
BUILD_DIR=$1
SRC_DIR=`pwd`
VERSION=$2
TYPE=$3

for os in `ls docker`; do
    docker build -t eshkol_$os_$TYPE -f `pwd`/docker/$os/$TYPE/Dockerfile .
    container=`docker container create eshkol_$os_$TYPE:latest`
    docker container start $container
    mkdir -p $BUILD_DIR/$os/$TYPE
    docker cp "$container:/app/build/_packages/eshkol_${VERSION}_amd64.deb" $BUILD_DIR/$os/$TYPE
    docker container stop $container
done
