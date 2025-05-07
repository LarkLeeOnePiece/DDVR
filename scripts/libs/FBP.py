import os
import numpy as np
from PIL import Image
import re
import glob
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import map_coordinates
from skimage.transform import iradon
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import tomopy
import time



def dataLoader(folder, resize_to=None):
    """
    Load cryo-ET projection data and convert it to log space using Beer-Lambert law.
    
    Parameters:
        folder (str): Path to folder containing .tif projection images.
        resize_to (tuple or None): (new_W, new_H), if not None, resize images.
    
    Returns:
        proj_log (np.ndarray): Log-transformed projections, shape (N, H, W).
    """
    file_list = sorted(glob.glob(f"{folder}/*.tif"))
    print(f"file_list: {file_list}")

    projections = []
    for f in file_list:
        img = Image.open(f).convert("F")  # "F" mode for float32
        
        if resize_to is not None:
            img = img.resize(resize_to, Image.BILINEAR)
        
        projections.append(np.array(img, dtype=np.float32))

    proj = np.stack(projections, axis=-1)  # (H, W, N)
    I0 = np.max(proj)
    proj_log = -np.log(proj / I0 + 1e-6)

    return proj_log

def load_tilt_angles(file_path):
    """
    Load tilt angles from a .rawtlt file.

    Parameters:
        file_path (str): Path to the tilt.rawtlt file.
    
    Returns:
        np.ndarray: Array of tilt angles.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove empty lines and convert to float
    angles = [float(line.strip()) for line in lines if line.strip()]
    return np.array(angles)



# Step 1: Ram-Lak filter
def apply_ramlak_filter(sino):
    n, n_angles = sino.shape
    f = fftfreq(n).reshape(-1, 1)
    filter_kernel = 2 * np.abs(f)
    sino_fft = fft(sino, axis=0)
    filtered_fft = sino_fft * filter_kernel
    return np.real(ifft(filtered_fft, axis=0))  # shape: (n, n_angles)
# Step 2: Back Projection (accelerated with numba)
@njit(parallel=True)
def backproject_2d_numba(filtered_sino, angles, output_size):
    recon = np.zeros((output_size, output_size), dtype=np.float32)
    n = filtered_sino.shape[0]
    x = np.arange(output_size) - output_size // 2
    y = np.arange(output_size) - output_size // 2
    for i in prange(output_size):
        for j in prange(output_size):
            xx = x[j]
            yy = y[i]
            sum_val = 0.0
            for k in range(len(angles)):
                angle = angles[k]
                t = xx * np.cos(angle) + yy * np.sin(angle)
                t = t * (n / output_size) + n // 2
                t = min(max(t, 0), n - 2)
                t0 = int(t)
                t1 = t0 + 1
                alpha = t - t0
                val = (1 - alpha) * filtered_sino[t0, k] + alpha * filtered_sino[t1, k]
                sum_val += val
            recon[i, j] = sum_val * (np.pi / len(angles))
    return recon

# Step 2: single layer Back Projection
def backproject_2d(filtered_sino, angles, output_size):
    recon = np.zeros((output_size, output_size), dtype=np.float32)
    n = filtered_sino.shape[0]
    x = np.arange(output_size) - output_size // 2
    y = np.arange(output_size) - output_size // 2
    xx, yy = np.meshgrid(x, y)
    
    for i, angle in enumerate(angles):
        t = xx * np.cos(angle) + yy * np.sin(angle)
        t = t * (n / output_size) + n // 2
        t = np.clip(t, 0, n - 1)
        projection = filtered_sino[:, i]
        recon += np.interp(t, np.arange(n), projection)
    
    recon *= np.pi / len(angles)
    return recon

# Step 3: main function：for each slice apply FBP
def fbp_reconstruct_manual(projs, angles):
    num_angles, num_slices, num_detectors = projs.shape
    recon_slices = []
    run_time=0.0
    for i in range(num_slices):
        sino = projs[:, i, :].T  # shape: (num_detectors, num_angles)
        filtered = apply_ramlak_filter(sino)
        start = time.perf_counter()
        recon = backproject_2d_numba(filtered, angles, output_size=num_detectors)
        end = time.perf_counter()
        time_iterval=end-start
        run_time+=time_iterval
        recon_slices.append(recon)
    return np.stack(recon_slices, axis=0)  # shape: (num_slices, H, W)




@njit
def query_fbp_point_2d(filtered_sino, cosines, sines, n_det, out_sz, x, y):
    """
    在单层 (2D) 上查询浮点 (x,y) 处的密度值。
    filtered_sino: (n_det, n_angles)
    cosines, sines: (n_angles,)
    n_det: 探测器数量
    out_sz: 输出网格在 x,y 方向的像素数（通常 = n_det）
    x,y: 以像素为单位、中心在 (0,0) 的浮点坐标
    """
    n_angles = cosines.shape[0]
    acc = 0.0
    for k in range(n_angles):
        # 1) 计算投影距离 t
        t = x * cosines[k] + y * sines[k]
        # 2) 映射到探测器编号
        t = t * (n_det / out_sz) + n_det * 0.5
        # 3) 边界裁剪
        if t < 0.0:
            t = 0.0
        elif t > n_det - 2:
            t = n_det - 2.0
        # 4) 线性插值
        i0 = int(t)
        alpha = t - i0
        val = (1.0 - alpha) * filtered_sino[i0, k] + alpha * filtered_sino[i0+1, k]
        acc += val
    # 5) 乘上 π/|θ|
    return acc * (np.pi / n_angles)

@njit
def query_fbp_point_3d(filtered_vol, cosines, sines, n_det, out_sz, x, y, z):
    """
    支持任意浮点 z 的 3D 查询。
    filtered_vol: (Nz, n_det, n_angles)
    cosines, sines: (n_angles,)
    n_det: 探测器数量
    out_sz: 输出网格在 x,y 方向的像素数
    x,y: 以像素为单位、中心在 (0,0) 的浮点坐标
    z: 切片索引的浮点坐标，范围 [0, Nz-1]
    """
    if filtered_vol is not None:
        Nz = filtered_vol.shape[0]
    else:
        Nz = 0
    # 限制 z 在合法范围内
    if z < 0.0:
        z = 0.0
    elif z > Nz - 1:
        z = Nz - 1.0

    # 取出相邻整数层
    z0 = int(z)
    z1 = z0 + 1 if z0 < Nz - 1 else z0
    tz = z - z0

    # 分别查询两层
    val0 = query_fbp_point_2d(filtered_vol[z0], cosines, sines, n_det, out_sz, x, y)
    val1 = query_fbp_point_2d(filtered_vol[z1], cosines, sines, n_det, out_sz, x, y)
    # 在 z 方向上线性插值
    return (1.0 - tz) * val0 + tz * val1
