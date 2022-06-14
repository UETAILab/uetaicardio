import os
import cv2
import math
import time
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from PIL import Image
from easydict import EasyDict
from pykalman import KalmanFilter
from echols.log import logger
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info(f'Enlapsed time {method.__name__}: {te-ts}')
        return result
    return timed

def extract_contour(mask, thresh_hold=230):
    def smooth_contour(c):
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

    imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, thresh_hold, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    if len(contours) == 0: return []
    max_area, max_contour = max([(cv2.contourArea(c), c) for c in contours], key= lambda x:x[0])
    return smooth_contour(max_contour)

def get_peak_by_middle_mfpunet(contour):
    rTriangle = cv2.minEnclosingTriangle(contour)
    points = rTriangle[1].squeeze(1)

    # Find nearest contour points to triangle points
    contour = contour.squeeze(1)
    dist = ((contour[:, None, :] - points[None, :, :])**2).sum(axis=-1)
    nearest_contour_points = contour[np.argmin(dist, axis=0)]
    nearest_contour_points = nearest_contour_points[np.argsort(nearest_contour_points[:, 1])]
    peak = nearest_contour_points[0].astype(int)
    return (peak[0], peak[1])

def line_equation_from_points(p1, p2):
    # Ax + By = C
    A = p2[1]-p1[1]
    B = p1[0]-p2[0]
    C = B*p1[1]+A*p1[0]
    return A,B,C

def hypot_length(p, q, xs, ys):
    # length of hypotenuse
    dx = (p[0]-q[0])*xs
    dy = (p[1]-q[1])*ys
    return math.hypot(dx, dy)

def compute_cos(p1, p2, p3, p4):
    r"""Compute cos between vector (p2 - p1) and vector (p4 - p3)"""
    cos = np.abs(np.sum((p2 - p1) * (p4 - p3)) / (hypot_length(p1, p2, 1, 1) * hypot_length(p3, p4, 1, 1)))
    return cos

def pdist_to_line(p, line_coeff):
    A, B, C = line_coeff
    return np.abs(A*p[0] + B*p[1] - C) / np.sqrt(A**2 + B**2)

def adjusted_get_base_points_by_bbox_basepoints(contour, vertical_threshold=0.01, horizontal_threshold=0.7, max_step=20):
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
    A_base, B_base, C_base = line_equation_from_points(sorted_box[2], sorted_box[3])
    original_left_pt_idx = np.where((contour == basepoints[0]).all(axis=-1))[0][0]
    original_right_pt_idx = np.where((contour == basepoints[1]).all(axis=-1))[0][0]

    vertical_threshold = vertical_threshold \
            * min(hypot_length(sorted_box[0], sorted_box[-1], 1, 1), hypot_length(sorted_box[0], sorted_box[-2], 1, 1))

    horizontal_threshold = horizontal_threshold \
            * hypot_length(sorted_box[0], sorted_box[1], 1, 1) \
            / compute_cos(sorted_box[0], sorted_box[1], contour[original_left_pt_idx], contour[original_right_pt_idx])

    left_pt_idx, right_pt_idx = original_left_pt_idx, original_right_pt_idx
    while True:
        is_updated = False
        if left_pt_idx+1 <= right_pt_idx and left_pt_idx+1 <= original_left_pt_idx + max_step and\
            pdist_to_line(contour[left_pt_idx+1], (A_base, B_base, C_base)) <= vertical_threshold and\
            hypot_length(contour[left_pt_idx+1], contour[right_pt_idx], 1, 1) <= horizontal_threshold:
            left_pt_idx += 1
            is_updated = True
        if right_pt_idx-1 >= left_pt_idx and right_pt_idx-1 >= original_right_pt_idx - max_step and\
            pdist_to_line(contour[right_pt_idx-1], (A_base, B_base, C_base)) <= vertical_threshold and\
            hypot_length(contour[left_pt_idx], contour[right_pt_idx-1], 1, 1) <= horizontal_threshold:
            right_pt_idx -= 1
            is_updated = True
        if not is_updated:
            break

    return {"lbase": (contour[left_pt_idx][0], contour[left_pt_idx][1]),
            "rbase": (contour[right_pt_idx][0], contour[right_pt_idx][1]),
            "box": box}

@timeit
def smooth_pivots(pivot_sequence, covariance_scale=30, n_iter=10):
    kalman_params = EasyDict(dict(
        transition_matrix=[[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]],
        observation_matrix=[[1, 0, 0, 0],
                            [0, 0, 1, 0]]
    ))
    # TODO improve this kalman filter process
    for i in range(pivot_sequence.shape[1]):
        initial_state_mean = [pivot_sequence[0, i, 0], 0,
                              pivot_sequence[0, i, 1], 0]

        # first Kalman filter
        kf = KalmanFilter(transition_matrices=kalman_params.transition_matrix,
                          observation_matrices=kalman_params.observation_matrix,
                          initial_state_mean=initial_state_mean)
        kf = kf.em(pivot_sequence[:, i], n_iter=n_iter)
        smoothed_state_means, smoothed_state_covariances = kf.smooth(pivot_sequence[:, i])

        # second Kalman filter
        kf2 = KalmanFilter(transition_matrices=kalman_params.transition_matrix,
                           observation_matrices=kalman_params.observation_matrix,
                           initial_state_mean=initial_state_mean,
                           observation_covariance=covariance_scale*kf.observation_covariance,
                           em_vars=['transition_covariance', 'initial_state_covariance'])
        kf2 = kf2.em(pivot_sequence[:, i], n_iter=n_iter)
        smoothed_state_means, smoothed_state_covariances = kf2.smooth(pivot_sequence[:, i])

        # final prediction
        pivot_sequence[:, i, 0] = smoothed_state_means[:, 0]
        pivot_sequence[:, i, 1] = smoothed_state_means[:, 2]
    return pivot_sequence

def convert_pivots_to_contour(pivots):
    n_pivots = len(pivots)
    left_side = pivots[n_pivots//2::-1]
    right_side = pivots[:n_pivots//2:-1]
    contour = np.concatenate([left_side, right_side], axis=0)
    return contour[:, None, :]

def init_pivots_for_tracking(coarse_initial_pivots, frames, ignored_pivots, max_distance=35, score_threshold=0.7):
    # remove misc pixels in the image
    frames = np.array(frames)
    frame_mean = frames.mean(axis=0).astype(np.uint8)
    initial_frame = frames[0].copy()
    initial_frame[(initial_frame - frame_mean) == 0] = 0
    #initial_frame = cv2.medianBlur(initial_frame, 3) # TODO uncmt this
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

def project_pivots_on_contour(pivots, contour):
    contour = contour[:, 0, :] # (n_cnts, 2)
    dist = np.sqrt(np.sum((pivots[:, None, :] - contour[None, :, :])**2, axis=-1)) # (n_pivots, n_cnts)
    closest_cnt_points = np.argmin(dist, axis=-1) # (n_pivots, )
    return contour[closest_cnt_points]

def compute_area(contour, x_scale, y_scale):
    def scale(xc, xs, ys):
        c = xc.astype(np.float32).copy()
        c[:, :, 0] = c[:, :, 0]*xs
        c[:, :, 1] = c[:, :, 1]*ys
        return c
    area = cv2.contourArea(scale(contour.astype(np.float32), x_scale, y_scale))
    return area

def compute_lv_length(contour, peak, basepoints, x_scale, y_scale):
    def get_intersection(p1, p2, q1, q2):
        A1,B1,C1 = line_equation_from_points(p1,p2)
        A2,B2,C2 = line_equation_from_points(q1,q2)
        det = A1*B2-A2*B1
        if abs(det) < 1e-6: return None
        ip = np.array([(B2*C1-B1*C2)/det, (A1*C2-A2*C1)/det], dtype=np.float)
        return ip if (ip[0]-q1[0])*(ip[0]-q2[0]) <= 0 and (ip[1]-q1[1])*(ip[1]-q2[1]) <= 0 else None

    def get_intersection_hull(p1, p2, hull):
        n = len(hull)
        ret = []
        for i in range(n):
            q1, q2 = hull[i], hull[(i+1)%n]
            ip = get_intersection(p1, p2, q1, q2)
            if ip is not None: ret.append(ip)
        return np.array(ret)

    # Find projection from peak to base
    middle = basepoints.mean(axis=0)
    ip = get_intersection_hull(peak, middle, np.reshape(contour, (-1,2)).astype(np.float))
    assert(len(ip) >= 1), 'something wrong'
    bottom = max(ip, key=lambda x: x[1]).astype(int)

    return hypot_length(peak, bottom, x_scale, y_scale), {
        'peak': (peak[0], peak[1]),
        'bottom': (bottom[0], bottom[1])
    }

def compute_EF(window, volumes):
    def get_quantile(x, quantile):
        '''x (list): each element is of the form (idx, object)'''
        return sorted(x, key=lambda x:x[1])[int(quantile*len(x))]

    volumes = [(i, v) for i, v in enumerate(volumes) if v is not None]
    if len(volumes) == 0: return None, None, None
    efs = []
    n = len(volumes)
    min_quantile = 0.05
    max_quantile = 0.95
    win_size = int(window*0.9)
    for i in range(0, max(1, n-win_size)):
        vs = volumes[i:i+win_size]
        (idxMin, minV) = get_quantile(vs[win_size//3:], min_quantile)
        (idxMax, maxV) = get_quantile(vs[:win_size*2//3], max_quantile)
        ef = (maxV-minV)/maxV*100
        efs.append( (i, ef, idxMin,minV,idxMax, maxV) )
    idxEF, EF = get_quantile([(e[0],e[1]) for e in efs], 0.5)
    return EF, efs, idxEF

def get_quantile(x, quantile):
    '''x (list): each element is of the form (idx, object)'''
    return sorted(x, key=lambda x:x[1])[int(quantile*len(x))]

def echonet_method(volumes, win_size, min_quan=0.05, max_quan=0.95):
    n = len(volumes)
    out = []
    volumes = [(i, v) for i, v in enumerate(volumes) if v is not None]
    for i in range(0, n-win_size):
        dat = volumes[i:i+win_size]
        (idxMin, minV) = get_quantile(dat, min_quan)
        (idxMax, maxV) = get_quantile(dat, max_quan)
        ef = (minV-maxV)/maxV*100
        org_min_id = idxMin
        org_max_id = idxMax
        out.append([i, ef, org_min_id, org_max_id])
    ef_idx, EF = get_quantile([(e[0], e[1]) for e in out], 0.5)
    return EF, out[ef_idx]

def compute_gls(time_window, pivots_seq, min_idx, max_idx):
    dist = np.sqrt(np.sum((pivots_seq[:, 1:] - pivots_seq[:, :-1])**2, axis=-1))
    # distance between points in a contour
    sums = np.sum(dist, axis=-1)
    # sum of distance is equals to length of that contour

    es = sums[min_idx]
    ed = sums[max_idx]
    GLS = (es - ed) / ed * 100.0
    print('GLS:', GLS)
    print(dist.shape, min_idx, max_idx)

    SLS6 = []
    num_points = dist.shape[1]
    split = num_points // 5
    for i in range(0, num_points, split):
        data = dist[:, i:i+split]
        data = np.sum(data, axis=-1)
        sed = data[max_idx]
        ses = data[min_idx]
        ssl = (ses - sed) / sed * 100.0

        #SLS = (data - sed) / sed * 100.0
        #SLS = gaussian_filter1d(SLS, sigma=1)
        SLS6.append(ssl)
        #plt.plot(SLS)

    if os.getenv('PLOT') is not None:
        plt.savefig('results/fig.png')
    return GLS, SLS6

def get_gls_by_basepoints(window, basepoints, contours):
    lv_diameters = []
    for i in range(len(basepoints)):
        contour = contours[i].squeeze(1)
        left_pt_idx = np.where((contour == basepoints[i]["lbase"]).all(axis=-1))[0][0]
        right_pt_idx = np.where((contour == basepoints[i]["rbase"]).all(axis=-1))[0][0]
        dist = np.sqrt(np.sum((contour[1:] - contour[:-1])**2, axis=-1))
        lv_diameter = dist[:left_pt_idx].sum() + dist[right_pt_idx:].sum()
        lv_diameters.append(lv_diameter)
    GLS, glss, idxGLS = compute_EF(window, lv_diameters)
    return GLS

def get_gls_by_segments(window, pivot_sequence):
    dist = np.sqrt(np.sum((pivot_sequence[:, 1:, :] - pivot_sequence[:, :-1, :])**2, axis=-1))
    GLS_components = [compute_EF(window, dist[:, i])[0] for i in range(dist.shape[1])]
    #logger.debug(f"GLS components {GLS_components}")
    GLS = np.mean(GLS_components)
    return GLS

def get_base_points_by_bbox_basepoints(self, contour):
    r"""redundant function"""
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
    A_base, B_base, C_base = line_equation_from_points(sorted_box[2], sorted_box[3])
    original_left_pt_idx = np.where((contour == basepoints[0]).all(axis=-1))[0][0]
    original_right_pt_idx = np.where((contour == basepoints[1]).all(axis=-1))[0][0]

    vertical_threshold = 0.01 * min(hypot_length(sorted_box[0], sorted_box[-1], 1, 1), hypot_length(sorted_box[0], sorted_box[-2], 1, 1))
    horizontal_threshold = 0.7 * hypot_length(sorted_box[0], sorted_box[1], 1, 1)
    max_step = 20
    left_pt_idx = original_left_pt_idx
    for i in range(original_left_pt_idx, len(contour)):
        if pdist_to_line(contour[i], (A_base, B_base, C_base)) >= vertical_threshold:
            left_pt_idx = i
        else:
            break
        if hypot_length(contour[i], contour[original_right_pt_idx], 1, 1) >= horizontal_threshold:
            left_pt_idx = i
        else:
            break
        if i >= original_left_pt_idx + max_step:
            break
    right_pt_idx = original_right_pt_idx
    for i in range(original_right_pt_idx, -1, -1):
        if pdist_to_line(contour[i], (A_base, B_base, C_base)) >= vertical_threshold:
            right_pt_idx = i
        else:
            break
        if hypot_length(contour[left_pt_idx], contour[i], 1, 1) >= horizontal_threshold:
            right_pt_idx = i
        else:
            break
        if i <= original_right_pt_idx - max_step:
             break

    return {
        "lbase": (contour[left_pt_idx][0], contour[left_pt_idx][1]),
        "rbase": (contour[right_pt_idx][0], contour[right_pt_idx][1]),
        "box": box
    }

def visualize(output_dir, dataset, msks, contours, pivot_sequence, basepoints, areas, lengths, volumes, pivot_tracker):
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
    def overlay_mask(frame, mask):
        red = np.zeros_like(frame)
        red[:, :] = [255, 0, 0]
        red_mask = cv2.bitwise_and(red, red, mask=mask[:,:,0])
        out = cv2.addWeighted(frame, 1, red_mask, 0.7, 0)
        return out
    def draw_base_points(frame, mask, bps):
        '''Draw base points using min enclosing rectangle method'''
        vis1 = frame.copy()
        vis1 = overlay_mask(vis1, mask)
        box = bps["box"].astype(int)
        cv2.line(vis1, tuple(box[0]), tuple(box[1]), (0, 0, 255), thickness=3)
        cv2.line(vis1, tuple(box[1]), tuple(box[2]), (0, 0, 255), thickness=3)
        cv2.line(vis1, tuple(box[2]), tuple(box[3]), (0, 0, 255), thickness=3)
        cv2.line(vis1, tuple(box[3]), tuple(box[0]), (0, 0, 255), thickness=3)
        cv2.circle(vis1, tuple(bps["lbase"]), 4, (0, 255, 0), thickness=3)
        cv2.circle(vis1, tuple(bps["rbase"]), 4, (0, 255, 0), thickness=3)
        return vis1
    def draw_peak_point(frame, mask, contour):
        '''draw peak point using enclosing triangle method'''
        vis2 = frame.copy()
        vis2 = overlay_mask(vis2, mask)
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
        return vis2
    def draw_speckles(frame, pivots, pivot_tracker):
        vis3 = frame.copy()
        for pivot_point in pivots:
            pivot_point = int(pivot_point[0]), int(pivot_point[1])
            x1, y1 = pivot_point[0] - (pivot_tracker.kernel_size[0] - 1)//2, pivot_point[1] - (pivot_tracker.kernel_size[1] - 1)//2
            x2, y2 = x1 + pivot_tracker.kernel_size[0], y1 + pivot_tracker.kernel_size[1]
            cv2.rectangle(vis3, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis3, tuple(pivot_point), 2, (0, 255, 0), thickness=3)
        return vis3
    def draw_contour(frame, length, pivots, pivot_tracker):
        vis4 = frame.copy()
        cv2.circle(vis4, length[1]["peak"], 2, (0, 255, 0), thickness=3)
        cv2.circle(vis4, length[1]["bottom"], 2, (0, 255, 0), thickness=3)
        cv2.line(vis4, length[1]["peak"], length[1]["bottom"], (0, 255, 0))

        for j in range(len(contour)):
            p1 = int(contour[j-1, 0, 0]), int(contour[j-1, 0, 1])
            p2 = int(contour[j, 0, 0]), int(contour[j, 0, 1])
            cv2.line(vis4, tuple(p1), tuple(p2), (255, 255, 0, 128), thickness=2)
        for pivot_point in pivots:
            pivot_point = int(pivot_point[0]), int(pivot_point[1])
            x1, y1 = pivot_point[0] - (pivot_tracker.kernel_size[0] - 1)//2, pivot_point[1] - (pivot_tracker.kernel_size[1] - 1)//2
            x2, y2 = x1 + pivot_tracker.kernel_size[0], y1 + pivot_tracker.kernel_size[1]
            cv2.circle(vis4, tuple(pivot_point), 2, (0, 255, 0), thickness=5)
        return vis4
    def make_grid(*viss):
        vis1, vis2, vis3, vis4 = viss
        top = np.concatenate([vis1, vis2], axis=1)
        bot = np.concatenate([vis3, vis4], axis=1)
        vis = np.concatenate([top, bot], axis=0)
        vis = cv2.resize(vis, (int(vis.shape[1] * 0.75), (int(vis.shape[0] * 0.75))))
        return vis
    #----------
    if output_dir is None: return
    if not os.path.exists(f"{output_dir}"): os.makedirs(f"{output_dir}")

    vis_images = []
    for i, (data, msk, contour, pivots, bps, area, length, volume) in \
        enumerate(zip(dataset, msks, contours, pivot_sequence, basepoints, areas, lengths, volumes)):
        frame = data['image'].copy()
        vis1 = draw_base_points(frame, msk, bps)
        vis2 = draw_peak_point(frame, msk, contour)
        vis3 = draw_speckles(frame, pivots, pivot_tracker)
        vis4 = draw_contour(frame, length, pivots, pivot_tracker)
        visg = make_grid(vis1, vis2, vis3, vis4)

        vis = Image.fromarray(visg)
        vis.save(f"{output_dir}/vis-{i}.jpg")
        vis_images.append(vis)
    vis_images[0].save(f"{output_dir}/vis.gif", save_all=True, append_images=vis_images, optimize=False, duration=30, loop=0)

def plot(output_dir, areas, lengths, volumes, EF, efs, idxEF, GLS1, GLS2):
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))

    volumes = np.array(volumes)
    #volumes = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    esv_vol, edv_vol = volumes[efs[idxEF][2]], volumes[efs[idxEF][4]]

    areas = np.array(areas)
    #areas = (areas - areas.min()) / (areas.max() - areas.min())
    esv_area, edv_area = areas[efs[idxEF][2]], areas[efs[idxEF][4]]

    lengths = np.array([l[0] for l in lengths])
    #lengths = (lengths - lengths.min()) / (lengths.max() - lengths.min())
    esv_len, edv_len = lengths[efs[idxEF][2]], lengths[efs[idxEF][4]]

    ax[0].plot(volumes, "--", label=f"vol")
    ax[0].plot([efs[idxEF][2], efs[idxEF][2]], [volumes.min(), volumes.max()], \
            "--", label=f"esv - vol {esv_vol:.4f} - len {esv_len:.4f} - area {esv_area:.4f}")
    ax[0].plot([efs[idxEF][4], efs[idxEF][4]], [volumes.min(), volumes.max()], \
            "--", label=f"edv - vol {edv_vol:.4f} - len {edv_len:.4f} - area {edv_area:.4f}")
    ax[0].set_title(f"volume")
    ax[0].legend()

    ax[1].plot(areas, "--", label=f"area")
    ax[1].plot([efs[idxEF][2], efs[idxEF][2]], [areas.min(), areas.max()], \
            "--", label=f"esv - vol {esv_vol:.4f} - len {esv_len:.4f} - area {esv_area:.4f}")
    ax[1].plot([efs[idxEF][4], efs[idxEF][4]], [areas.min(), areas.max()], \
            "--", label=f"edv - vol {edv_vol:.4f} - len {edv_len:.4f} - area {edv_area:.4f}")
    ax[1].set_title(f"area")
    ax[1].legend()

    ax[2].plot(lengths, "--", label=f"len")
    ax[2].plot([efs[idxEF][2], efs[idxEF][2]], [lengths.min(), lengths.max()], \
            "--", label=f"esv - vol {esv_vol:.4f} - len {esv_len:.4f} - area {esv_area:.4f}")
    ax[2].plot([efs[idxEF][4], efs[idxEF][4]], [lengths.min(), lengths.max()], \
            "--", label=f"edv - vol {edv_vol:.4f} - len {edv_len:.4f} - area {edv_area:.4f}")
    ax[2].set_title(f"length")
    ax[2].legend()

    fig.suptitle(f"EF {EF:.4f} - bpGLS {GLS1:.4f} - sGLS {GLS2:.4f}")
    plt.savefig(f"{output_dir}/plot.png")

# https://stackoverflow.com/questions/66676502/how-can-i-move-points-along-the-normal-vector-to-a-curve-in-python
def extend_contour(contour, d_length=5, eps=1e-6):
    r"""Extend contour using Bspline method"""
    # pivot sequence has shape n*2
    x, y = contour[:, 0], contour[:, 1]

    # get the cumulative distance along the contour
    dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))

    # build a spline representation of the contour
    spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=0)

    ul = np.array(u) - eps
    ur = np.array(u) + eps

    xul, yul = scipy.interpolate.splev(ul, spline)
    xu , yu  = scipy.interpolate.splev(u , spline)
    xur, yur = scipy.interpolate.splev(ur, spline)

    xtans = (xur - xul) / (2*eps)
    ytans = (yur - yul) / (2*eps)

    norm_vecs = np.vstack([-ytans, xtans]).transpose(1, 0)
    unorm_vecs = norm_vecs / np.linalg.norm(norm_vecs, axis=-1)[..., None]

    old_ps = np.vstack([xu, yu]).transpose(1, 0)
    new_ps = old_ps + d_length * unorm_vecs

    new_ps = np.array(new_ps[:, :], dtype=np.int)
    return new_ps
