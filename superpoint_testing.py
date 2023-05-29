import copy
from datetime import datetime
from typing import Dict

import pandas as pd
from openvino.runtime import Core
from models.superpoint.models.SuperPointNet_gauss2 import SuperPointNet_gauss2
from models.superpoint.VideoStreamer import VideoStreamer
from models.superpoint.PointTracker import PointTracker, myjet
import argparse
from pathlib import Path
import time
import cv2

import numpy as np
import torch
import torch.nn.functional as F


def get_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str,
                        help='Path to pretrained weights file.')
    parser.add_argument('--onnx_path', type=str,
                        help='Path to pretrained onnx file.')
    parser.add_argument('--ir_path', type=str,
                        help='Path to pretrained ir file.')
    parser.add_argument('--cv_kp', action='store_true',
                        help='Use OpenCV to detect keypoints')
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
    parser.add_argument('--colab', action='store_true',
                        help='Use google colab\' cv_imshow()')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write', action='store_true',
                        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)
    return opt


@torch.no_grad()
def simple_nms(scores: torch.Tensor, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

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


@torch.no_grad()
def remove_borders(keypoints: torch.Tensor, scores: torch.Tensor, border_width: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border_width) & (
        keypoints[:, 0] < (height - border_width))
    mask_w = (keypoints[:, 1] >= border_width) & (
        keypoints[:, 1] < (width - border_width))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


@torch.no_grad()
def select_topK(keypoints: torch.Tensor, scores: torch.Tensor, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


@torch.no_grad()
def extract_points(semi: torch.Tensor,
                   nms_radius: int = 4,
                   conf_thresh: float = 5e-3,
                   border_width: int = 4,
                   topK: int = -1):
    # Softmax & remove dustbin
    scores = F.softmax(semi, 1)[:, :-1]
    b, _, h, w = scores.shape  # _=64
    # b*64*h*w -> b*h*w*64 -> b*h*w*8*8
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
    # b*h*w*8*8 -> b*h*8*w*8 -> b*(h*8)*(w*8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
    heatmap = scores.clone()
    # NMS
    scores = simple_nms(scores, nms_radius)
    # For each batch, get location/index of keypoint
    keypoints = [torch.nonzero(s > conf_thresh) for s in scores]  # b*n_kp*2
    # For each batch, get score of the point by location
    scores = [s[tuple(k.T)] for s, k in zip(scores, keypoints)]  # b*n_kp

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
    return keypoints, scores, heatmap # N*2, N, H/8*W/8


@torch.no_grad()
def extract_descriptors(keypoints, descriptors, s: int = 8):
    def sample(pnts, descs):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descs.shape
        pnts = pnts - s / 2 + 0.5
        pnts /= torch.tensor([(w*s - s/2 - 0.5),
                              (h*s - s/2 - 0.5)],).to(pnts)[None]
        pnts = pnts*2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
        descs = F.grid_sample(descs, pnts.view(b, 1, -1, 2),
                              mode='bilinear', **args)
        descs = F.normalize(descs.reshape(b, c, -1), p=2, dim=1)
        return descs
    # N*D
    return [sample(k[None], d[None])[0].T
            for k, d in zip(keypoints, descriptors)]


def log_dict(d: Dict):
    for k, v in d.items():
        v = float(v)
        if "time" in k:
            sv = f"{v*1000:.2f}ms"
        elif "prec" in k:
            sv = f"{v*100:.1f}%"
        else:
            sv = f"{v:.2f}"
        sfps = f"({1./float(v):.2f} FPS)" if "total" in k else ""
        print(f"{k}: {sv}{sfps}", end=", ")
    print()

def convert2CVKeypoint(keypoints:np.ndarray, descriptors:np.ndarray):
    # keypoints: N*3
    # descriptors: N*D
    N = keypoints.shape[0]
    assert(N==descriptors.shape[0])
    kp = []
    desc = []
    for i in range(N):
        x, y, score = pts[i, :]
        d = descriptors[i, :]
        kp.append(cv2.KeyPoint(x, y, size=1, angle=-1, response=score, octave=0, class_id=-1))
        desc.append(d)
    # keypoints: N, descriptors: N*D
    return tuple(kp), np.array(desc)

def cv_match(query_desc:np.ndarray, train_desc:np.ndarray, ratio: float=0.7):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    K = 2
    if len(query_desc) < K or len(train_desc) < K:
        return np.array([])
    matches = flann.knnMatch(query_desc,train_desc,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    return np.array(good) # N_match


def draw_matches(img_query, img_train, kp_query, kp_train, matches, alpha: float=0.5, draw_params=None):
    if draw_params is None:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(0, 0, 255),
                           flags=cv2.DrawMatchesFlags_DEFAULT)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_GRAY2RGB)
    img_train = cv2.cvtColor(img_train, cv2.COLOR_GRAY2RGB)
    blank = np.zeros_like(img_query)
    # Draw matched lines
    img_match_lines = cv2.drawMatches(blank, kp_query, blank, kp_train, matches,
                                      None, **draw_params)
    img_2img = np.hstack((img_query, img_train))
    return cv2.addWeighted(img_match_lines, alpha, img_2img, 1, 0)

if __name__ == '__main__':
    with torch.no_grad():
        args = get_args()

        # For Colab
        if args.colab:
            from google.colab.patches import cv2_imshow

        CSV_FILE = Path(f"./output_{datetime.now():%Y%m%d_%H%M}.csv")

        # This class helps load input images from different sources.
        vs = VideoStreamer(args.input, args.camid, args.H,
                           args.W, args.skip, args.img_glob)

        device = torch.device('cpu')
        if args.cuda:
            device = torch.device('cuda')

        superpoint = None
        # Torch model
        if args.weights_path:
            superpoint = SuperPointNet_gauss2()
            checkpoint = torch.load(args.weights_path, map_location=device)
            superpoint.load_state_dict(checkpoint["model_state_dict"])
            superpoint.to(device)
            superpoint.eval()

        # ONNX model
        if args.onnx_path or args.ir_path:
            ie = Core()
            model_path = args.ir_path if args.onnx_path is None else args.onnx_path
            model_onnx = ie.read_model(model=model_path)
            superpoint = ie.compile_model(model=model_onnx, device_name="CPU")
            # output_layer_onnx = superpoint.output(0)

        # Log
        log = {
            "image": 0,
            "pre_time": 0,
            "net_time": 0,
            "post_time": 0,
            "total_time": 0,
            "num_kp": 0,
            "match_prec": 0,
        }
        pd_rows = []

        # Other config for testing
        tracker = PointTracker(args.max_length, nn_thresh=args.nn_thresh)
        # Create a window to display the demo.
        win = 'SuperPoint Tracker'
        if not args.colab:
            cv2.namedWindow(win)
        # Font parameters for visualizaton.
        font = cv2.FONT_HERSHEY_DUPLEX
        font_clr = (255, 255, 255)
        font_pt = (4, 12)
        font_sc = 0.4

        img_p = None
        cv_kp_c = None
        cv_des_c = None
        cv_kp_p = None
        cv_des_p = None
        while True:

            start = time.time()

            # Preprocess
            start_pre = time.time()
            # Get a new image.
            img, status = vs.next_frame()
            if status is False:
                break
            img_c = (img * 255.).astype('uint8')
            assert img.ndim == 2, 'Image must be grayscale.'
            assert img.dtype == np.float32, 'Image must be float32.'
            H, W = img.shape[0], img.shape[1]
            inp = img.copy()
            inp = inp.reshape(1, 1, H, W)
            if args.weights_path:
                inp = torch.from_numpy(inp)
                inp = inp.to(device)
            end_pre = time.time()

            # Inference
            start_net = time.time()
            if not args.cv_kp:
                if args.weights_path:
                    output = superpoint(inp)
                else:
                    output = superpoint([inp])
            else:
                sift = cv2.SIFT_create()
                # find the keypoints and descriptors with SIFT
                pts, desc = sift.detectAndCompute(img_c,None)
                if len(pts) == 0:
                    desc = []
            end_net = time.time()

            # Post processing, Get points and descriptors.
            start_post = time.time()
            if not args.cv_kp:
                val = tuple(output.values())
                semi = torch.tensor(val[0])
                desc = torch.tensor(val[1])
                pts, scores, heatmap = extract_points(semi)
                desc = extract_descriptors(pts, desc)
            end_post = time.time()

            end = time.time()

            log["image"] = vs.i
            log["pre_time"] = end_pre - start_pre
            log["net_time"] = end_net - start_net
            log["post_time"] = end_post - start_post
            log["total_time"] = end - start

            # Visualization
            cv_kp_c, cv_des_c = pts, desc
            if not args.cv_kp:
                # Get first from batch
                pts = pts[0].cpu()
                scores = scores[0].cpu()
                desc = desc[0].cpu()
                heatmap = heatmap[0].cpu()
                # Reshape to display
                pts = torch.cat((pts, scores[None].T), dim=1).numpy()
                desc = desc.numpy()
                heatmap = heatmap.numpy()
                # current keypoints
                cv_kp_c, cv_des_c = convert2CVKeypoint(pts, desc)

            # draw matches with previous frame
            out = img_c
            matches = np.zeros((1, 1))
            if cv_des_p is not None and cv_kp_p is not None and img_p is not None:
                matches = cv_match(cv_des_c, cv_des_p)
                if args.show_extra:
                    out = draw_matches(img_c, img_p, cv_kp_c, cv_kp_p, matches)

            # Update previous data
            cv_kp_p = cv_kp_c
            cv_des_p = cv_des_c
            img_p = img_c

            # Add points and descriptors to the tracker.
            # matches = tracker.update(pts, desc)
            
            # log["match_prec"] = float(matches.shape[1]) / float(pts.shape[1])
            log["match_prec"] = float(len(matches)) / float(len(cv_kp_c)) if len(cv_kp_c) > 0 else 0
            log["num_kp"] = len(cv_kp_c)
            log_dict(log)
            pd_rows.append(copy.deepcopy(log))

            # # Get tracks for points which were match successfully across all frames.
            # tracks = tracker.get_tracks(args.min_length)

            # # Primary output - Show point tracks overlayed on top of input image.
            # out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
            # # Normalize track scores to [0,1].
            # tracks[:, 1] /= float(args.nn_thresh)
            # tracker.draw_tracks(out1, tracks)
            # cv2.putText(out1, 'Point Tracks', font_pt, font,
            #             font_sc, font_clr, lineType=16)

            # # Extra output -- Show current point detections.
            # out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
            # for pt in pts.T:
            #     pt1 = (int(round(pt[0])), int(round(pt[1])))
            #     cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
            # cv2.putText(out2, 'Raw Point Detections', font_pt,
            #             font, font_sc, font_clr, lineType=16)

            # # Extra output -- Show the point confidence heatmap.
            # if heatmap is not None:
            #     min_conf = 0.001
            #     heatmap[heatmap < min_conf] = min_conf
            #     heatmap = -np.log(heatmap)
            #     heatmap = (heatmap - heatmap.min()) / \
            #         (heatmap.max() - heatmap.min() + .00001)

            #     out3 = myjet[np.round(
            #         np.clip(heatmap*10, 0, 9)).astype('int'), :]
            #     out3 = (out3*255).astype('uint8')
            # else:
            #     out3 = np.zeros_like(out2)
            # cv2.putText(out3, 'Raw Point Confidences', font_pt,
            #             font, font_sc, font_clr, lineType=16)

            # # Resize final output.
            # out = np.hstack((out1, out2, out3))
            if args.show_extra:
                out = cv2.resize(
                    out, (args.display_scale*out.shape[1], args.display_scale*out.shape[0]))

                # Display visualization image to screen.
                if not args.colab:
                    cv2.imshow(win, out)
                else:
                  # For Colab, DisabledFunctionError inherit from ValueError
                  cv2_imshow(out)
                
                key = cv2.waitKey(args.waitkey) & 0xFF
                if key == ord('q'):
                    print('Quitting, \'q\' pressed.')
                    break
            # === End while ===
        pd.DataFrame(pd_rows).to_csv(f"{CSV_FILE}")
