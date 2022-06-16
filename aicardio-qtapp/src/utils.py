import cv2
import time
import numpy as np
import shapely.geometry as geom


def draw_contour(frame, contour):
    out = frame.copy()
    n = len(contour)
    contour = np.array(contour)[None,...]
    out = cv2.drawContours(out, contour, -1, (0, 255, 0), thickness=2)
    #for p in contour:
        #out = cv2.circle(out, (p[0], p[1]), 1, (0, 255, 0), 3)
    #for p in contour[::6]:
        #out = cv2.circle(out, (p[0], p[1]), 1, (0, 0, 255), 4)
    return out


def draw_line(img, p1, p2):
    img = cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), thickness=1)
    return img


def get_base_line(peak, mid_point, cd_length=2000):
    midLine = geom.LineString([peak, mid_point])
    left = midLine.parallel_offset(cd_length, 'left', join_style=2)
    right = midLine.parallel_offset(cd_length, 'right', join_style=2)
    c = left.coords[0]
    d = right.coords[-1]
    # note the different orientation for right offset
    baseLine = geom.LineString([c, d])
    return baseLine


def biplane(contour, n_part=21):
    contour = np.array(contour)
    n = len(contour)
    m = n // 2
    peak = contour[m]

    lbase_idx, rbase_idx = 0, -1
    lbase = contour[lbase_idx]
    rbase = contour[rbase_idx]

    mid_point = (lbase + rbase) / 2
    hypot_length = np.sqrt((peak[0] - mid_point[0])**2 + (peak[1] - mid_point[1])**2)

    base_line = get_base_line(peak, mid_point)

    left_wall = [mid_point] + list(contour[:m])
    right_wall = list(contour[m:]) + [mid_point]
    left_wall = geom.LineString(left_wall)
    right_wall = geom.LineString(right_wall)

    avg_height = hypot_length / (n_part-1)
    lines = []
    radius = []
    for length in np.linspace(0, hypot_length, n_part)[1:-1]:
        above_line = base_line.parallel_offset(length, 'left', join_style=2)
        if not (above_line.intersects(left_wall) and above_line.intersects(right_wall)):
            continue

        lp = above_line.intersection(left_wall).coords[0]
        rp = above_line.intersection(right_wall).coords[0]
        lines.append([lp, rp])
        radius.append(np.sqrt((lp[0] - rp[0])**2 + (lp[1] - rp[1])**2))
    return avg_height, radius, lines, peak, mid_point


class MyTimer:
    def __init__(self):
        self.time = 0
    def tik(self):
        self.time = time.time()
    def tok(self, fps=30):
        duration = time.time() - self.time
        if duration <= 1/float(fps):
            time.sleep(1/float(fps) - duration)
