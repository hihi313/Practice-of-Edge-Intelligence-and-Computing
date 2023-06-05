import argparse
import os
from pathlib import Path
import sys
from models.superpoint.models.SuperPointNet_gauss2 import SuperPointNet_gauss2
import torch


def get_args():
    parser = argparse.ArgumentParser(description="HRL")
    parser.add_argument("--weights",
                        default=None,
                        required=True,
                        help='weights of pretrained model (default: "")')
    parser.add_argument("--output",
                        default=None,
                        help='output path/name of onnx model (with .onnx ext.)')
    parser.add_argument("--W",
                        default=640,
                        help='model input image width (int)')
    parser.add_argument("--H",
                        default=480,
                        help='model input image height (int)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()


    default_fname="ir_output"
    if os.path.isfile(args.weights):
        filenameExt = os.path.basename(args.weights)
        default_fname = f"models/{filenameExt.rsplit('.')[0]}.onnx"
    else:
        print("The path is not a file")
        sys.exit()

    ONNX_PATH = Path(args.output) if args.output is not None else default_fname
    print(f"Output file name: {ONNX_PATH}")
    device = torch.device('cpu')

    # Load network
    net = SuperPointNet_gauss2()
    # Load weights
    checkpoint = torch.load(f"{args.weights}", map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    # Eval mode
    net.eval()

    dummy_input = torch.randn(1, 1, int(args.H), int(args.W))
    torch.onnx.export(
        net,
        dummy_input,
        f"{ONNX_PATH}",
        input_names=["image"],
        output_names=["semi", "desc"],
        dynamic_axes={
            # dict value: manually named axes
            "image": {2: "height",
                      3: "width"},
            "semi": {2: "height",
                     3: "width"},
            "desc": {2: "height",
                     3: "width"}
        },
        verbose=True
    )
