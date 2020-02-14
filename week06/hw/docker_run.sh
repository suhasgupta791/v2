#!/bin/sh 

nvidia-docker run --rm \
	-v $(pwd):/root/v2/week06/hw \
	--name hw06 \
	-p 8888:8888 \
	-d w251/hw06:x86-64
