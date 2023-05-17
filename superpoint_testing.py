
import argparse
from pathlib import Path
import time
import cv2

import numpy as np
import torch
import torch.nn.functional as F
import sys
print(sys.path)
from models.superpoint.PointTracker import PointTracker, myjet
from models.superpoint.VideoStreamer import VideoStreamer
from models.superpoint.models.SuperPointNet_gauss2 import SuperPointNet_gauss2
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

def simple_nms(scores: torch.Tensor, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return F.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def remove_borders(keypoints: torch.Tensor, scores: torch.Tensor, border_width: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border_width) & (keypoints[:, 0] < (height - border_width))
    mask_w = (keypoints[:, 1] >= border_width) & (keypoints[:, 1] < (width - border_width))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def select_topK(keypoints: torch.Tensor, scores: torch.Tensor, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

def extract_points(semi:torch.Tensor, 
                   nms_radius: int = 4, 
                   conf_thresh: float = 5e-3,
                   border_width: int = 4,
                   topK: int = -1):
    # Softmax & remove dustbin
    scores = F.softmax(semi, 1)[:, :-1]
    b, _, h, w = scores.shape # _=64
    # b*64*h*w -> b*h*w*64 -> b*h*w*8*8
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
    # b*h*w*8*8 -> b*h*8*w*8 -> b*(h*8)*(w*8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
    heatmap = scores.clone()
    # NMS
    scores = simple_nms(scores, nms_radius)
    # For each batch, get location/index of keypoint
    keypoints = [torch.nonzero(s > conf_thresh) for s in scores] # b*n_kp*2
    # For each batch, get score of the point by location
    scores = [s[tuple(k.T)] for s, k in zip(scores, keypoints)] # b*n_kp

    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(k, s, border_width, h*8, w*8)
        for k, s in zip(keypoints, scores)]))
    
    # Keep the k keypoints with highest score
    if topK >= 0:
        keypoints, scores = list(zip(*[
            select_topK(k, s, topK)
            for k, s in zip(keypoints, scores)]))
        
    # Convert (h, w) to (x, y)
    keypoints = [torch.flip(k, [1]).float() for k in keypoints]
    return keypoints, scores, heatmap


def extract_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    D = descriptors.shape[1] # dim of a descriptor
    def sample(pnts, descs):
        b, c, h, w = descs.shape  # 1*256*hc*wc, b=1
        pnts = pnts - s / 2 + 0.5
        pnts /= torch.tensor([(w*s - s/2 - 0.5),
                             (h*s - s/2 - 0.5)],).to(pnts)[None]
        pnts = pnts*2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
        descs = F.grid_sample(descs, pnts.view(b, 1, -1, 2),
                              mode='bilinear', **args)
        descs = F.normalize(descs.reshape(b, c, -1), p=2, dim=1)
        return descs.reshape(-1, D)

    return [sample(k[None], d[None])
            for k, d in zip(keypoints, descriptors)]

if __name__ == '__main__':
    args = get_args()

    # This class helps load input images from different sources.
    vs = VideoStreamer(args.input, args.camid, args.H,
                        args.W, args.skip, args.img_glob)
    
    device = torch.device('cpu')

    # ONNX model
    ONNX_PATH = Path("./models/tmp.onnx")
    ie = Core()
    model_onnx = ie.read_model(model=f"{ONNX_PATH}")
    superpoint = ie.compile_model(model=model_onnx, device_name="CPU")
    # output_layer_onnx = superpoint.output(0)

    # Torch model
    superpoint = SuperPointNet_gauss2()
    checkpoint = torch.load(args.weights_path)
    superpoint.load_state_dict(checkpoint["model_state_dict"])
    superpoint.to(device)
    superpoint.eval()

    # Other config for testing
    tracker = PointTracker(args.max_length, nn_thresh=args.nn_thresh)
    # Create a window to display the demo.
    win = 'SuperPoint Tracker'
    cv2.namedWindow(win)
    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

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
        output = superpoint([inp])
        val = tuple(output.values())
        semi = val[0]
        desc = val[1]
        pts, scores, heatmap = extract_points(torch.tensor(semi))
        desc = extract_descriptors(pts, torch.tensor(desc))
        end1 = time.time()

        end = time.time()
        net_t = (1. / float(end1 - start))
        total_t = (1. / float(end - start))
        print(f"Processed image {vs.i} (net+post_process: {net_t:.2f} FPS, total: {total_t:.2f} FPS).")

        # Get first from batch
        pts = pts[0]
        scores = scores[0]
        desc = desc[0]
        heatmap = heatmap[0]
        # Reshape to display 
        pts = torch.cat((pts.T, scores[None]), dim=0).numpy()
        desc = desc.T.numpy()
        heatmap = heatmap.numpy()
        # Add points and descriptors to the tracker.
        tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = tracker.get_tracks(args.min_length)

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(args.nn_thresh)  # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        cv2.putText(out1, 'Point Tracks', font_pt, font,
                    font_sc, font_clr, lineType=16)

        # Extra output -- Show current point detections.
        out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, 'Raw Point Detections', font_pt,
                    font, font_sc, font_clr, lineType=16)

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min() + .00001)
            
            out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
            out3 = (out3*255).astype('uint8')
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(out3, 'Raw Point Confidences', font_pt,
                    font, font_sc, font_clr, lineType=16)

        # Resize final output.
        out = np.hstack((out1, out2, out3))
        out = cv2.resize(
            out, (3*args.display_scale*args.W, args.display_scale*args.H))

        # Display visualization image to screen.
        cv2.imshow(win, out)
        key = cv2.waitKey(args.waitkey) & 0xFF
        if key == ord('q'):
            print('Quitting, \'q\' pressed.')
            break
