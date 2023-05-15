
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from models.superpoint.VideoStreamer import VideoStreamer
from openvino.runtime import Core

def get_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--img_glob', type=str, default='*.png',
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
                        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=120,
                        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
                        help='Input image width (default:160).')
    parser.add_argument('--display_scale', type=int, default=2,
                        help='Factor to scale output visualization (default: 2).')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
                        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
                        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed (default: False)')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write', action='store_true',
                        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)
    return opt


def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
        3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
        0 : Empty or suppressed.
        1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
        in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
        nmsed_corners - 3xN numpy matrix with surviving corners.
        nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0]+pad, rc[1]+pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds



if __name__ == '__main__':
    args = get_args()

    # This class helps load input images from different sources.
    vs = VideoStreamer(args.input, args.camid, args.H,
                        args.W, args.skip, args.img_glob)
    
    device = torch.device('cpu')
    # Load network to Inference Engine
    ONNX_PATH = Path("./models/tmp.onnx")
    ie = Core()
    model_onnx = ie.read_model(model=f"{ONNX_PATH}")
    compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

    # output_layer_onnx = compiled_model_onnx.output(0)

    while True:

        start = time.time()

        # Get a new image.
        img, status = vs.next_frame()
        if status is False:
            break
        # Image preprocess
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = inp.reshape(1, 1, H, W)        

        # Get points and descriptors.
        start1 = time.time()
        output = compiled_model_onnx([inp])
        val = tuple(output.values())
        semi = val[0]
        desc = val[1]
        end1 = time.time()

        end = time.time()
        net_t = (1. / float(end1 - start))
        total_t = (1. / float(end - start))
        print(f"Processed image {vs.i} (net+post_process: {net_t:.2f} FPS, total: {total_t:.2f} FPS).")
