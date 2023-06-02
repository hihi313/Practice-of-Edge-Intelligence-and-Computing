#!/bin/bash

# VSCode's launch.json doesn't support setting & getting (env) variable immedietly
# So use this shell script to test superpoint conviniently

TEST_PY="superpoint_testing.py"
MODEL="superpoint_35k_finetune"
DATASET="/datasets/VSLAM/HISLAB/EE7/ee_map_good_01"
H=240
W=320
SHOW="--show_extra"
GPU=""

set -x
while getopts "m:sciot:" opt
do
  case $opt in
    # Model
    m)
        MODEL=$OPTARG
        ;;
    # Show
    s)
        SHOW="--show_extra"
        ;;
    # OpenCV SIFT
    c)
        python3 $TEST_PY $DATASET --H $H --W $W $SHOW --cv_kp
        ;;
    # OpenVINO IR
    i)
        python3 $TEST_PY $DATASET --H $H --W $W $SHOW --ir_path "./models/${MODEL}/superpoint.xml"
        ;;
    # ONNX
    o)
        python3 $TEST_PY $DATASET --H $H --W $W $SHOW --onnx_path "./models/${MODEL}.onnx"
        ;;
    # Torch
    t)
        # GPU
        if [[ $OPTARG == *"g"* ]]; then
            GPU="--cuda"
        fi
        python3 $TEST_PY $DATASET --H $H --W $W $SHOW --weights_path "./models/${MODEL}.pth.tar" $GPU
        ;;
    \?) 
        echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    *)
        echo "*"
        ;;
  esac
done
set +x

# python3 superpoint_testing.py             
                # // "/datasets/VSLAM/HISLAB/EE7/ee_map_good_01",
#                 "/datasets/COCO/test",
#                 "--H", "240",
#                 "--W", "320",
#                 // "--cuda",
#                 // "--colab",
#                 // "--cv_kp",
#                 "--show_extra",
#                 "--weights_path", "/app/models/superpoint_35k_finetune.pth.tar",
#                 // "--onnx_path", "/app/models/superpoint.onnx",
#                 // "--ir_path", "/app/models/superpoint_openvino/superpoint.xml",
