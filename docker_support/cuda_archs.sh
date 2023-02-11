#!/bin/bash

if [ "`echo $CUDA_VERSION | cut -d. -f-2`" == "11.8" ] ; then 
    if [ "$1" == "long" ] ; then
        echo "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"
    else
        echo "6.0;7.0;7.5;8.0;8.6;8.9"
    fi
else
    if [ "$1" == "long" ] ; then
        echo "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62  -gencode arch=compute_70,code=sm_70  -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
    else
        echo "6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"
    fi
fi


