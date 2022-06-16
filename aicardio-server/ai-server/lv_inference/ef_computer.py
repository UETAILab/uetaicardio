import os
import time
import math
import uuid

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict
from pykalman import KalmanFilter

from lv_inference.pivot_analyzer import PivotAnalyzer
from lv_inference.multiangle_pivot_tracker import MultiAnglePivotTracker
from lv_inference.cv2_pivot_tracker import CV2PivotTracker


class EFComputer:
    def __init__(self):
        # 2C
        tracker_config = EasyDict(dict(
            kernel_size=(91, 91), #(31, 31),#(121, 121), # (61, 61),
            velocity=7.2,
            angles=[0]
        ))
        self.pivot_tracker = MultiAnglePivotTracker(tracker_config)
        self.pivot_analyzer = PivotAnalyzer()
        self.kalman_params = EasyDict(dict(
            transition_matrix=[[1, 1, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 0, 1]],
            observation_matrix=[[1, 0, 0, 0],
                                [0, 0, 1, 0]]
        ))
    
    def compute_ef(self, msks, metadata, dataset, output_dir=None):
        start = time.time()
        contours = [self.__get_contour(m) for m in msks]
        peaks = [self.__get_peak_by_middle_mfpunet(c) for c in contours]
        basepoints = [self.__adjusted_get_base_points_by_bbox_basepoints(c) for c in contours]
        contours, peaks, basepoints = self.__smooth_contours(contours, peaks, basepoints)
        pivot_sequence = self.__estimate_pivots(contours, peaks, basepoints, dataset, msks, metadata)

        areas = [self.__get_area(c, metadata['x_scale'], metadata['y_scale']) for c in contours]
        lengths = [self.__get_length(c, pivots[3], pivots[[0, -1]], metadata['x_scale'], metadata['y_scale']) for c, pivots in zip(contours, pivot_sequence)]
        volumes = np.array([ 8.0 / 3.0 / math.pi * area * area / l[0] for area, l in zip(areas, lengths)])
        print(f"Misc time: {time.time() - start:.4f}")

        # Compute EF and GLS
        EF, efs, idxEF = self.__get_EF(metadata['window'], volumes)
        GLS1 = self.__get_gls_by_basepoints(metadata['window'], basepoints, contours)
        GLS2 = self.__get_gls_by_segments(metadata['window'], pivot_sequence)
        
        
        
        
        
        # VISUALIZATION ---
        if (output_dir is not None) and (not os.path.exists(f"{output_dir}")):
            os.makedirs(f"{output_dir}")
        
        def extend_contour(frame, contour):
            def scale_contour(contour, scale = 1.0):
                cx, cy = get_center(contour)
                cnt = contour - [cx, cy]
                cnt_scaled = cnt * scale
                cnt_scaled += [cx, cy]
                cnt_scaled = cnt_scaled.astype(np.int32)
                return cnt_scaled

            def get_center(contour):
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                return (cX, cY)
            neighbor_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            in_cnts = scale_contour(contour, 0.85)
            out_cnts = scale_contour(contour, 1.15)
            cv2.fillPoly(neighbor_map, pts=[out_cnts], color=(255, 255, 255))
            cv2.fillPoly(neighbor_map, pts=[in_cnts], color=(0, 0, 0))
            neighbor_ys, neighbor_xs = np.where(neighbor_map > 0)
            extended_contour = np.concatenate([neighbor_xs[..., None], neighbor_ys[..., None]], axis=-1)
            return extended_contour

        vis_images = []
        for i, (data, msk, contour, pivots, bps, area, length, volume) in enumerate(zip(dataset, msks, contours, pivot_sequence, basepoints, areas, lengths, volumes)):
            vis1 = data["image"].copy()
            vis1 = cv2.addWeighted(vis1, 1, msk, 0.7, 0)
#             extended_contour = extend_contour(data["image"], contour)
#             vis1[extended_contour[:, 1], extended_contour[:, 0]] = (255, 0, 0)
#             vis_kernel_size = (33, 33)
#             for pivot_point in pivots:
#                 pivot_point = int(pivot_point[0]), int(pivot_point[1])
#                 x1, y1 = pivot_point[0] - (vis_kernel_size[0] - 1)//2, pivot_point[1] - (vis_kernel_size[1] - 1)//2
#                 x2, y2 = x1 + vis_kernel_size[0], y1 + vis_kernel_size[1]
#                 cv2.rectangle(vis1, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.circle(vis1, tuple(pivot_point), 2, (0, 255, 0), thickness=3)
            box = bps["box"].astype(int)
            cv2.line(vis1, tuple(box[0]), tuple(box[1]), (0, 0, 255), thickness=3)
            cv2.line(vis1, tuple(box[1]), tuple(box[2]), (0, 0, 255), thickness=3)
            cv2.line(vis1, tuple(box[2]), tuple(box[3]), (0, 0, 255), thickness=3)
            cv2.line(vis1, tuple(box[3]), tuple(box[0]), (0, 0, 255), thickness=3)
            cv2.circle(vis1, tuple(bps["left_basepoint"]), 4, (0, 255, 0), thickness=3)
            cv2.circle(vis1, tuple(bps["right_basepoint"]), 4, (0, 255, 0), thickness=3)
            
#             vis2 = data["image"].copy()
#             for pivot_point in pivots:
#                 pivot_point = int(pivot_point[0]), int(pivot_point[1])
#                 cv2.circle(vis2, tuple(pivot_point), 2, (0, 255, 0), thickness=3)

            vis2 = data["image"].copy()
            vis2 = cv2.addWeighted(vis2, 1, msk, 0.7, 0)
            rTriangle = cv2.minEnclosingTriangle(contour)
            points = rTriangle[1].squeeze(1)

            # Find nearest contour points to triangle points
            dcontour = contour.squeeze(1)
            dist = ((dcontour[:, None, :] - points[None, :, :])**2).sum(axis=-1)
            nearest_contour_points = dcontour[np.argmin(dist, axis=0)]
            nearest_contour_points = nearest_contour_points[np.argsort(nearest_contour_points[:, 1])]
            peak = nearest_contour_points[0].astype(int)
            cv2.line(vis2, tuple(points[0]), tuple(points[1]), (0, 0, 255), thickness=3)
            cv2.line(vis2, tuple(points[1]), tuple(points[2]), (0, 0, 255), thickness=3)
            cv2.line(vis2, tuple(points[2]), tuple(points[0]), (0, 0, 255), thickness=3)
            cv2.circle(vis2, tuple(peak), 4, (0, 255, 0), thickness=3)
            
            vis3 = data["image"].copy()
            for pivot_point in pivots:
                pivot_point = int(pivot_point[0]), int(pivot_point[1])
                x1, y1 = pivot_point[0] - (self.pivot_tracker.kernel_size[0] - 1)//2, pivot_point[1] - (self.pivot_tracker.kernel_size[1] - 1)//2
                x2, y2 = x1 + self.pivot_tracker.kernel_size[0], y1 + self.pivot_tracker.kernel_size[1]
                cv2.rectangle(vis3, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(vis3, tuple(pivot_point), 2, (0, 255, 0), thickness=3)
            
            vis4 = data["image"].copy()
            cv2.circle(vis4, length[1]["peak"], 2, (0, 255, 0), thickness=3)
            cv2.circle(vis4, length[1]["bottom"], 2, (0, 255, 0), thickness=3)
            cv2.line(vis4, length[1]["peak"], length[1]["bottom"], (0, 255, 0))
            
            for j in range(len(contour)):
                p1 = int(contour[j-1, 0, 0]), int(contour[j-1, 0, 1])
                p2 = int(contour[j, 0, 0]), int(contour[j, 0, 1])
                cv2.line(vis4, tuple(p1), tuple(p2), (255, 255, 0, 128), thickness=2)
            for pivot_point in pivots:
                pivot_point = int(pivot_point[0]), int(pivot_point[1])
                x1, y1 = pivot_point[0] - (self.pivot_tracker.kernel_size[0] - 1)//2, pivot_point[1] - (self.pivot_tracker.kernel_size[1] - 1)//2
                x2, y2 = x1 + self.pivot_tracker.kernel_size[0], y1 + self.pivot_tracker.kernel_size[1]
                cv2.circle(vis4, tuple(pivot_point), 2, (0, 255, 0), thickness=5)
            
            vis1 = np.concatenate([vis1, vis2], axis=1)
            vis2 = np.concatenate([vis3, vis4], axis=1)
            vis = np.concatenate([vis1, vis2], axis=0)
            vis = cv2.resize(vis, (int(vis.shape[1] * 0.75), (int(vis.shape[0] * 0.75))))
            
#             l = length[0]
#             cv2.putText(vis, f"case idx: {dataset.config.case_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis, f"area: {area:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis, f"len: {l:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis, f"vol: {volume:.4f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis, f"frame: {i}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #--- VIS FOR DEMO
#             vis1 = data["image"].copy()
#             vis2 = data["image"].copy()
#             for j in range(len(contour)):
#                 p1 = int(contour[j-1, 0, 0]), int(contour[j-1, 0, 1])
#                 p2 = int(contour[j, 0, 0]), int(contour[j, 0, 1])
#                 cv2.line(vis2, tuple(p1), tuple(p2), (255, 255, 0, 128), thickness=2)
#             cv2.putText(vis2, f"case idx: {dataset.config.case_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis2, f"area: {area:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis2, f"len: {l:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis2, f"vol: {volume:.4f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis2, f"frame: {i}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(vis2, f"EF: {EF:.4f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0, 128), 2)
#             cv2.putText(vis2, f"GLS: {GLS2:.4f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0, 128), 2)
#             vis = np.concatenate([vis1, vis2], axis=1)
            #---
            
            vis = Image.fromarray(vis)
            vis.save(f"{output_dir}/vis-{i}.jpg")
            vis_images.append(vis)
        vis_images[0].save(f"{output_dir}/vis.gif", save_all=True, append_images=vis_images, optimize=False, duration=30, loop=0)
        
#         efs.append( (i, ef, idxMin,minV,idxMax, maxV) )
        
        fig, ax = plt.subplots(3, 1, figsize=(15, 15))
        
        volumes = np.array(volumes)
#         volumes = (volumes - volumes.min()) / (volumes.max() - volumes.min())
        esv_vol, edv_vol = volumes[efs[idxEF][2]], volumes[efs[idxEF][4]]

        areas = np.array(areas)
#         areas = (areas - areas.min()) / (areas.max() - areas.min())
        esv_area, edv_area = areas[efs[idxEF][2]], areas[efs[idxEF][4]]
    
        lengths = np.array([l[0] for l in lengths])
#         lengths = (lengths - lengths.min()) / (lengths.max() - lengths.min())
        esv_len, edv_len = lengths[efs[idxEF][2]], lengths[efs[idxEF][4]]
    
        ax[0].plot(volumes, "--", label=f"vol")
        ax[0].plot([efs[idxEF][2], efs[idxEF][2]], [volumes.min(), volumes.max()], "--", label=f"esv - vol {esv_vol:.4f} - len {esv_len:.4f} - area {esv_area:.4f}")
        ax[0].plot([efs[idxEF][4], efs[idxEF][4]], [volumes.min(), volumes.max()], "--", label=f"edv - vol {edv_vol:.4f} - len {edv_len:.4f} - area {edv_area:.4f}")
        ax[0].set_title(f"volume - case idx: {dataset.config.case_idx}")
        ax[0].legend()
        
        ax[1].plot(areas, "--", label=f"area")
        ax[1].plot([efs[idxEF][2], efs[idxEF][2]], [areas.min(), areas.max()], "--", label=f"esv - vol {esv_vol:.4f} - len {esv_len:.4f} - area {esv_area:.4f}")
        ax[1].plot([efs[idxEF][4], efs[idxEF][4]], [areas.min(), areas.max()], "--", label=f"edv - vol {edv_vol:.4f} - len {edv_len:.4f} - area {edv_area:.4f}")
        ax[1].set_title(f"area - case idx: {dataset.config.case_idx}")
        ax[1].legend()
        
        ax[2].plot(lengths, "--", label=f"len")
        ax[2].plot([efs[idxEF][2], efs[idxEF][2]], [lengths.min(), lengths.max()], "--", label=f"esv - vol {esv_vol:.4f} - len {esv_len:.4f} - area {esv_area:.4f}")
        ax[2].plot([efs[idxEF][4], efs[idxEF][4]], [lengths.min(), lengths.max()], "--", label=f"edv - vol {edv_vol:.4f} - len {edv_len:.4f} - area {edv_area:.4f}")
        ax[2].set_title(f"length - case idx: {dataset.config.case_idx}")
        ax[2].legend()
        
        fig.suptitle(f"EF {EF:.4f} - bpGLS {GLS1:.4f} - sGLS {GLS2:.4f}")
        plt.savefig(f"{output_dir}/plot.png")
        
        # ---
        
        results = {
            "contours": [],
            "pivot_sequence": [],
            "areas": [],
            "volumes": []
        }
        for msk, contour, pivots, area, volume in zip(msks, contours, pivot_sequence, areas, volumes):
            results["contours"].append([
                {
                    "x": point[0, 0] / msk.shape[1],
                    "y": point[0, 1] / msk.shape[0]
                }
            for point in contour])
            results["pivot_sequence"].append([
                {
                    "x": point[0] / msk.shape[1],
                    "y": point[1] / msk.shape[0]
                }
            for point in pivots])
            results["areas"].append(area)
            results["volumes"].append(volume)
        results["ef"] = EF
        results["basepoint_gls"] = GLS1
        results["segmental_gls"] = GLS2
        
        return results
    
    def __get_contour(self, mask):
        imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 230, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area, max_contour = max([(cv2.contourArea(c), c) for c in contours], key= lambda x:x[0])
        return self.__get_smooth_contour(max_contour)

    def __get_smooth_contour(self, c):
        n, _, _ = c.shape
        if n < 21:
            return c
        ret = np.zeros_like(c)
        for i in range(n):
            s0, s1 = 0, 0
            width = 15
            for j in range(-(width//2), (width//2)+1):
                s0 += c[(i+n+j) % n, 0, 0]
                s1 += c[(i+n+j) % n, 0, 1]
            ret[i, 0, 0] = s0 // width
            ret[i, 0, 1] = s1 // width
        return ret

    def __get_area(self, contour, x_scale, y_scale):
        area = cv2.contourArea(self.__scale(contour.astype(np.float32), x_scale, y_scale))
        return area
    
    def __scale(self, xc, xs, ys):
        c = xc.astype(np.float32).copy()
        c[:, :, 0] = c[:, :, 0]*xs
        c[:, :, 1] = c[:, :, 1]*ys
        return c
    
    def __get_peak_by_middle_mfpunet(self, contour):
        rTriangle = cv2.minEnclosingTriangle(contour)
        points = rTriangle[1].squeeze(1)
        
        # Find nearest contour points to triangle points
        contour = contour.squeeze(1)
        dist = ((contour[:, None, :] - points[None, :, :])**2).sum(axis=-1)
        nearest_contour_points = contour[np.argmin(dist, axis=0)]
        nearest_contour_points = nearest_contour_points[np.argsort(nearest_contour_points[:, 1])]
        peak = nearest_contour_points[0].astype(int)
        return (peak[0], peak[1])
    
    def __get_length(self, contour, peak, basepoints, x_scale, y_scale):
        # Find projection from peak to base
        middle = basepoints.mean(axis=0)
        ip = self.__get_intersection_hull(peak, middle, np.reshape(contour, (-1,2)).astype(np.float))
        assert(len(ip) >= 1)
        bottom = max(ip, key=lambda x: x[1]).astype(int)
        
        return self.__length(peak, bottom, x_scale, y_scale), {
            'peak': (peak[0], peak[1]), 
            'bottom': (bottom[0], bottom[1])
        }
    
    def __get_intersection_hull(self, p1, p2, hull):
        n = len(hull)
        ret = []
        for i in range(n):
            q1, q2 = hull[i], hull[(i+1)%n]
            ip = self.__get_intersection(p1,p2,q1,q2)
            if ip is not None: ret.append(ip)
        return np.array(ret)
    
    def __get_intersection(self, p1,p2,q1,q2):
        A1,B1,C1 = self.__get_line_eq(p1,p2)
        A2,B2,C2 = self.__get_line_eq(q1,q2)
        det = A1*B2-A2*B1
        if abs(det) < 1e-6: return None
        ip = np.array([(B2*C1-B1*C2)/det, (A1*C2-A2*C1)/det], dtype=np.float)
        return ip if (ip[0]-q1[0])*(ip[0]-q2[0]) <= 0 and (ip[1]-q1[1])*(ip[1]-q2[1]) <= 0 else None
    
    def __get_line_eq(self, p1, p2):
        # Ax + By = C
        A = p2[1]-p1[1]
        B = p1[0]-p2[0]
        C = B*p1[1]+A*p1[0]
        return A,B,C
    
    def __get_dist_to_line(self, p, line_coeff):
        A, B, C = line_coeff
        return np.abs(A*p[0] + B*p[1] - C) / np.sqrt(A**2 + B**2)

    def __length(self, p, q, xs, ys):
        dx = (p[0]-q[0])*xs
        dy = (p[1]-q[1])*ys
        return math.hypot(dx, dy)
    
    def __get_EF(self, window, volumes):
        volumes = [(i,v) for i, v in enumerate(volumes) if v is not None]
        if len(volumes) == 0:
            return None, None, None
        efs = []
        n = len(volumes)
        min_quantile = 0.05
        max_quantile = 0.95
        win_size = int(window*0.9)
#         print("n", n, win_size)
        for i in range(0, max(1,n-win_size)):
            vs = volumes[i:i+win_size]
            (idxMin, minV) = self.__get_quantile(vs[win_size//3:], min_quantile)
            (idxMax, maxV) = self.__get_quantile(vs[:win_size*2//3], max_quantile)
            ef = (maxV-minV)/maxV*100
            efs.append( (i, ef, idxMin,minV,idxMax, maxV) )
        idxEF, EF = self.__get_quantile([(e[0],e[1]) for e in efs], 0.5)
#         print("ef", [f"{e[1]:.2f}" for e in efs])
        return EF, efs, idxEF
    
    def __get_base_points_by_bbox_basepoints(self, contour):
        rRect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rRect) # (4, 2)
        sorted_box = np.array(sorted(box, key=lambda x: -x[1]))

        # Find nearest contour points to box points
        contour = contour.squeeze(1)
        dist = ((contour[:, None, :] - sorted_box[None, :, :])**2).sum(axis=-1) # (cnt_len, 1, 2) - (1, 4, 2) ~> (cnt_len, 4)
        nearest_contour_points = contour[np.argmin(dist, axis=0)]
        nearest_contour_points = nearest_contour_points[np.argsort(nearest_contour_points[:, 1])]
        basepoints = nearest_contour_points[2:].astype(int)
        basepoints = sorted(basepoints, key=lambda x: x[0])
        
        # adjust basepoints
        A_base, B_base, C_base = self.__get_line_eq(sorted_box[2], sorted_box[3])
        original_left_pt_idx = np.where((contour == basepoints[0]).all(axis=-1))[0][0]
        original_right_pt_idx = np.where((contour == basepoints[1]).all(axis=-1))[0][0]
 
        vertical_threshold = 0.01 * min(self.__length(sorted_box[0], sorted_box[-1], 1, 1), self.__length(sorted_box[0], sorted_box[-2], 1, 1))
        horizontal_threshold = 0.7 * self.__length(sorted_box[0], sorted_box[1], 1, 1)
        max_step = 20
        left_pt_idx = original_left_pt_idx
        for i in range(original_left_pt_idx, len(contour)):
            if self.__get_dist_to_line(contour[i], (A_base, B_base, C_base)) >= vertical_threshold:
                left_pt_idx = i
            else:
                break
            if self.__length(contour[i], contour[original_right_pt_idx], 1, 1) >= horizontal_threshold:
                left_pt_idx = i
            else:
                break
            if i >= original_left_pt_idx + max_step:
                break
        right_pt_idx = original_right_pt_idx
        for i in range(original_right_pt_idx, -1, -1):
            if self.__get_dist_to_line(contour[i], (A_base, B_base, C_base)) >= vertical_threshold:
                right_pt_idx = i
            else:
                break
            if self.__length(contour[left_pt_idx], contour[i], 1, 1) >= horizontal_threshold:
                right_pt_idx = i
            else:
                break
            if i <= original_right_pt_idx - max_step:
                 break
         
        return {
            "left_basepoint": (contour[left_pt_idx][0], contour[left_pt_idx][1]),
            "right_basepoint": (contour[right_pt_idx][0], contour[right_pt_idx][1]),
            "box": box
        }

    def __adjusted_get_base_points_by_bbox_basepoints(self, contour, vertical_threshold=0.01, horizontal_threshold=0.7, max_step=20):
        rRect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rRect) # (4, 2)
        sorted_box = np.array(sorted(box, key=lambda x: -x[1]))

        # Find nearest contour points to box points
        contour = contour.squeeze(1)
        dist = ((contour[:, None, :] - sorted_box[None, :, :])**2).sum(axis=-1) # (cnt_len, 1, 2) - (1, 4, 2) ~> (cnt_len, 4)
        nearest_contour_points = contour[np.argmin(dist, axis=0)]
        nearest_contour_points = nearest_contour_points[np.argsort(nearest_contour_points[:, 1])]
        basepoints = nearest_contour_points[2:].astype(int)
        basepoints = sorted(basepoints, key=lambda x: x[0])
        
        # adjust basepoints
        A_base, B_base, C_base = self.__get_line_eq(sorted_box[2], sorted_box[3])
        original_left_pt_idx = np.where((contour == basepoints[0]).all(axis=-1))[0][0]
        original_right_pt_idx = np.where((contour == basepoints[1]).all(axis=-1))[0][0]

        vertical_threshold = vertical_threshold * min(self.__length(sorted_box[0], sorted_box[-1], 1, 1), self.__length(sorted_box[0], sorted_box[-2], 1, 1))
        horizontal_threshold = horizontal_threshold * self.__length(sorted_box[0], sorted_box[1], 1, 1) / self.__compute_cos(sorted_box[0], sorted_box[1], contour[original_left_pt_idx], contour[original_right_pt_idx])

        left_pt_idx, right_pt_idx = original_left_pt_idx, original_right_pt_idx
        while True:
            is_updated = False
            if left_pt_idx+1 <= right_pt_idx and left_pt_idx+1 <= original_left_pt_idx + max_step and\
                self.__get_dist_to_line(contour[left_pt_idx+1], (A_base, B_base, C_base)) <= vertical_threshold and\
                self.__length(contour[left_pt_idx+1], contour[right_pt_idx], 1, 1) <= horizontal_threshold:
                left_pt_idx += 1
                is_updated = True
            if right_pt_idx-1 >= left_pt_idx and right_pt_idx-1 >= original_right_pt_idx - max_step and\
                self.__get_dist_to_line(contour[right_pt_idx-1], (A_base, B_base, C_base)) <= vertical_threshold and\
                self.__length(contour[left_pt_idx], contour[right_pt_idx-1], 1, 1) <= horizontal_threshold:
                right_pt_idx -= 1
                is_updated = True
            if not is_updated:
                break
        
        return {
            "left_basepoint": (contour[left_pt_idx][0], contour[left_pt_idx][1]),
            "right_basepoint": (contour[right_pt_idx][0], contour[right_pt_idx][1]),
            "box": box
        }
    
    def __compute_cos(self, p1, p2, p3, p4):
        r"""Compute cos between vector (p2 - p1) and vector (p4 - p3)"""
        cos = np.abs(np.sum((p2 - p1) * (p4 - p3)) / (self.__length(p1, p2, 1, 1) * self.__length(p3, p4, 1, 1)))
        return cos
    
    def __get_gls_by_basepoints(self, window, basepoints, contours):
        lv_diameters = []
        for i in range(len(basepoints)):
            contour = contours[i].squeeze(1)
            left_pt_idx = np.where((contour == basepoints[i]["left_basepoint"]).all(axis=-1))[0][0]
            right_pt_idx = np.where((contour == basepoints[i]["right_basepoint"]).all(axis=-1))[0][0]
            dist = np.sqrt(np.sum((contour[1:] - contour[:-1])**2, axis=-1))
            lv_diameter = dist[:left_pt_idx].sum() + dist[right_pt_idx:].sum()
            lv_diameters.append(lv_diameter)
        GLS, glss, idxGLS = self.__get_EF(window, lv_diameters)
        return GLS
    
    def __get_gls_by_segments(self, window, pivot_sequence):
        dist = np.sqrt(np.sum((pivot_sequence[:, 1:, :] - pivot_sequence[:, :-1, :])**2, axis=-1))
        GLS_components = [self.__get_EF(window, dist[:, i])[0] for i in range(dist.shape[1])]
        print("GLS components", GLS_components)
        GLS = np.mean(GLS_components)
        return GLS
    
    def __get_quantile(self, x, quantile):
        '''x (list): each element is of the form (idx, object)'''
        return sorted(x, key=lambda x:x[1])[int(quantile*len(x))]

    def __smooth_pivots(self, pivot_sequence, covariance_scale=30, n_iter=10):
        for i in range(pivot_sequence.shape[1]):
            initial_state_mean = [pivot_sequence[0, i, 0], 0, 
                                  pivot_sequence[0, i, 1], 0]
            
            # first Kalman filter
            kf = KalmanFilter(transition_matrices=self.kalman_params.transition_matrix, 
                              observation_matrices=self.kalman_params.observation_matrix, 
                              initial_state_mean=initial_state_mean)
            kf = kf.em(pivot_sequence[:, i], n_iter=n_iter)
            smoothed_state_means, smoothed_state_covariances = kf.smooth(pivot_sequence[:, i])
            
            # second Kalman filter
            kf2 = KalmanFilter(transition_matrices=self.kalman_params.transition_matrix,
                               observation_matrices=self.kalman_params.observation_matrix,
                               initial_state_mean=initial_state_mean,
                               observation_covariance=covariance_scale*kf.observation_covariance,
                               em_vars=['transition_covariance', 'initial_state_covariance'])
            kf2 = kf2.em(pivot_sequence[:, i], n_iter=n_iter)
            smoothed_state_means, smoothed_state_covariances = kf2.smooth(pivot_sequence[:, i])
            
            # final prediction
            pivot_sequence[:, i, 0] = smoothed_state_means[:, 0]
            pivot_sequence[:, i, 1] = smoothed_state_means[:, 2]
        return pivot_sequence

    def __estimate_pivots(self, contours, peaks, basepoints, dataset, msks, metadata):
        frames = [data["image"] for data in dataset]
        detected_pivots_sequence = [self.pivot_analyzer.detect_pivots(c, p, b, n_segments_per_side=3) for c, p, b in zip(contours, peaks, basepoints)]
        initial_pivots = self.__initialize_pivots_for_tracking(
            detected_pivots_sequence[0],
            frames, [1, 2, 4, 5]
        )
        tracked_pivot_sequence = self.pivot_tracker.track_pivots(
            initial_pivots,
            frames, msks,
            contours, metadata
        )
        pivot_sequence = np.array([[tracked_pivots[0],
                                    tracked_pivots[1],
                                    tracked_pivots[2],
                                    tracked_pivots[3],
                                    tracked_pivots[4],
                                    tracked_pivots[5],
                                    tracked_pivots[6]] for detected_pivots, tracked_pivots in zip(detected_pivots_sequence, tracked_pivot_sequence)]).astype(int)
#         projected_pivot_sequence = np.array([self.__project_pivots_on_contour(pivots, contour) for pivots, contour in zip(pivot_sequence, contours)])
#         pivot_sequence[:, [0, 3, 6]] = projected_pivot_sequence[:, [0, 3, 6]]
#         pivot_sequence = projected_pivot_sequence
        pivot_sequence = self.__smooth_pivots(pivot_sequence, covariance_scale=5)
        return pivot_sequence

    def __initialize_pivots_for_tracking(self, coarse_initial_pivots, frames, ignored_pivots, max_distance=35, score_threshold=0.7):
        r"""Initialize pivots for tracking"""
        # remove misc pixels in the image
        frames = np.array(frames)
        frame_mean = frames.mean(axis=0).astype(np.uint8)
        initial_frame = frames[0].copy()
        initial_frame[(initial_frame - frame_mean) == 0] = 0
        initial_frame = cv2.medianBlur(initial_frame, 3)
        
        initial_frame = cv2.medianBlur(initial_frame, 5)
        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        
        # fine-tune coarse pivots
        finetuned_pivots = []
        for i, pivot in enumerate(coarse_initial_pivots):
            roi_mask = np.zeros((frame_mean.shape[0], frame_mean.shape[1]), dtype=np.uint8)
            cv2.circle(roi_mask, tuple(pivot), max_distance, (255, 255, 255), -1)

            candidates = cv2.goodFeaturesToTrack(
                gray, mask=roi_mask, maxCorners=1, qualityLevel=score_threshold,
                minDistance=1, blockSize=61
            )
            if candidates is None or i in ignored_pivots:
                finetuned_pivots.append(pivot)
            else:
                candidates = np.array(candidates).squeeze(1)
                dist = np.sum((candidates - pivot)**2, axis=-1)
                nearest_candidate = candidates[np.argmin(dist)].astype(int)
                finetuned_pivots.append(nearest_candidate)
        finetuned_pivots = np.array(finetuned_pivots)
        return finetuned_pivots
    
    def __project_pivots_on_contour(self, pivots, contour):
        contour = contour[:, 0, :] # (n_cnts, 2)
        dist = np.sqrt(np.sum((pivots[:, None, :] - contour[None, :, :])**2, axis=-1)) # (n_pivots, n_cnts)
        closest_cnt_points = np.argmin(dist, axis=-1) # (n_pivots, )
        return contour[closest_cnt_points]
    
    def __smooth_contours(self, contours, peaks, basepoints):
        start = time.time()
        detected_pivots_sequence = [self.pivot_analyzer.detect_pivots(c, p, b, n_segments_per_side=20) for c, p, b in zip(contours, peaks, basepoints)]
        pivot_sequence = np.concatenate([pivots[None, ...] for pivots in detected_pivots_sequence])
        pivot_sequence = self.__smooth_pivots(pivot_sequence, n_iter=3)

        contours = [self.__convert_pivots_to_contours(pivots) for pivots in pivot_sequence]
        peaks = [tuple(contour[0, 0]) for contour in contours]
        basepoints = [{"left_basepoint": tuple(contour[len(contour)//2, 0]), "right_basepoint": tuple(contour[len(contour)//2+1, 0]), "box": basepoint["box"]} for contour, basepoint in zip(contours, basepoints)]
        print(f"Contour smooth time: {time.time() - start}")
        return contours, peaks, basepoints
    
    def __convert_pivots_to_contours(self, pivots):
        n_pivots = len(pivots)
        left_side = pivots[n_pivots//2::-1]
        right_side = pivots[:n_pivots//2:-1]
        contours = np.concatenate([left_side, right_side], axis=0)
        return contours[:, None, :]