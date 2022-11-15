import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
import cv2
import os


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


pcd_path = "./"
xyz_all = np.array([0, 0, 0])
for filename in os.listdir(pcd_path):
    whole_path = os.path.join(pcd_path, filename)
    # print(whole_path)
    if whole_path.find(".pcd") != 0:
        pcd = o3d.io.read_point_cloud(whole_path)
        xyz = np.asarray(pcd.points)
        xyz_all = np.vstack((xyz_all, xyz))

yz = xyz_all[:, 1:3]
yz = yz[1:, :]
cnt = np.array(yz)
cnt = np.array(cnt, dtype=np.float32)
rect = cv2.minAreaRect(cnt)
box_corners = cv2.boxPoints(rect)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(121)
ax.scatter(cnt[:, 0], cnt[:, 1])

ax.scatter([rect[0][0]], [rect[0][1]])
ax.plot(box_corners[:, 0], box_corners[:, 1], 'k-')

plt.title('vehicle length: ' + str(rect[1][1])[:6] + 'm, vehicle width: ' + str(rect[1][0])[:6] + "m")
plt.axis('equal')

ax = fig.add_subplot(122)
cnt_norm = cnt - rect[0]
rad = -math.radians(rect[2])
trans_pt = []
for idx, pt in enumerate(cnt_norm):
    trans_pt_x = math.cos(rad) * pt[0] - math.sin(rad) * pt[1]
    trans_pt_y = math.sin(rad) * pt[0] + math.cos(rad) * pt[1]
    trans_pt.append(trans_pt_x)
    trans_pt.append(trans_pt_y)
cnt_norm = np.array(trans_pt)
cnt_norm = np.reshape(cnt_norm, (-1, 2))
ax.scatter(cnt_norm[:, 0], cnt_norm[:, 1], color="red")

cnt_norm = np.array(cnt_norm, dtype=np.float32)
rect_norm = cv2.minAreaRect(cnt_norm)
box_corners_norm = cv2.boxPoints(rect_norm)

ax.scatter([rect_norm[0][0]], [rect_norm[0][1]])
ax.plot(box_corners_norm[:, 0], box_corners_norm[:, 1], 'k-')
plt.axis('equal')
plt.show()
