import numpy as np
import pygame
from numba import njit, prange
from libs.FBP import *
import matplotlib.pyplot as plt
from libs.utils import *
import os
import time
import cv2
from loguru import logger
# ==== View matrix from yaw/pitch ====
def get_view_matrix(yaw, pitch):
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    R_y = np.array([[cos_y, 0, sin_y],
                    [0,     1, 0],
                    [-sin_y, 0, cos_y]])
    R_x = np.array([[1, 0, 0],
                    [0, cos_p, -sin_p],
                    [0, sin_p, cos_p]])
    return R_x @ R_y

@njit
def estimate_local_density(volume, x, y, z, radius=1,querry=False,filtered_vol=None,cosines=None, sines=None,
                            n_det=None, out_sz=None,):
    D, H, W = volume.shape
    sum_density = 0.0
    count = 0
    for dz in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                xi = int(np.floor(x)) + dx
                yi = int(np.floor(y)) + dy
                zi = int(np.floor(z)) + dz
                if 0 <= xi < W and 0 <= yi < D and 0 <= zi < H:
                    if querry and filtered_vol is not None:
                        # pass
                        sum_density +=query_fbp_point_3d(
                            filtered_vol,cosines, sines,
                            n_det, out_sz,
                            x=xi, y=zi, z=yi  # need to make sure about this
                        )# in volume (z,y,z)
                    else:
                        sum_density += volume[yi, zi, xi]
                    count += 1
    if count > 0:
        return sum_density / count
    else:
        return 0.0

# ==== Ray marching rendering (Numba accelerated) ====
@njit(parallel=True)
def render_volume_non_normalized(volume, view_matrix, zoom_factor, image_size=256, step_size=0.5,querry=False,filtered_vol=None,cosines=None,sines=None,n_det=None,out_sz=None,with_AO=False):
    if volume is not None:
        X, Z, Y = volume.shape[1],volume.shape[2],volume.shape[0]#num_detectors, num_detectors,num_slices, 
    else:
        X, Z, Y = 2000,2000,1000
    img = np.zeros((image_size, image_size))
    vec = view_matrix[:, 2]
    distance=np.sqrt(X*X+Z*Z+Y*Y)/2
    end_dis=3*distance
    occulusion_intensity=1
    # print("view_matrix[:, 2]:", vec[0], vec[1], vec[2])#view_matrix[:, 2]: 0.0 0.0 1.0
    for i in prange(image_size):
        for j in prange(image_size):
            offset_x = (j - image_size/2) # - image_size/2 to image_size/2
            offset_y = (i - image_size/2)
            offset_z = distance
            v = np.empty(3)
            v[0] = offset_x
            v[1] = offset_y
            v[2] = offset_z
            cam_origin= np.dot(view_matrix, v)
            dir = np.empty(3)
            dir[0] = 0.0
            dir[1] = 0.0
            dir[2] = -1.0
            ray_dir = np.dot(view_matrix, dir)
            ray_dir /= np.linalg.norm(ray_dir)

            color = 0.0
            opacity = 0.0
            t = 0.0
            while t < end_dis and opacity < 0.95:
                pos = cam_origin + t * ray_dir
                x = pos[0]+(X-1)/2 # X
                y = pos[1]+(Y-1)/2  # 
                z = pos[2]+(Z-1)/2  #
                # x = pos_norm[0] * (X - 1) # X
                # y = pos_norm[1] * (Y - 1) # 
                # z = pos_norm[2] * (Z - 1) #
                if 0 <= x < X-1 and 0 <= y < Y-1 and 0 <= z < Z-1:
                    
                    if querry and filtered_vol is not None:
                        # xi, yi, zi = int(x), int(y), int(z)
                        xi, yi, zi = x,y,z
                        # density=0# ignore it here now
                        xi = xi - n_det//2
                        zi = zi - n_det//2
                        
                        if with_AO:
                            local_density=estimate_local_density(volume=volume, x=x, y=y, z=z,querry=querry,filtered_vol=filtered_vol,cosines=cosines, sines=sines,
                            n_det=n_det, out_sz=out_sz)
                            occlusion = (1.0 - occulusion_intensity * local_density)*3
                            occlusion = max(0.0, min(occlusion, 1.0))
                            density=query_fbp_point_3d(
                                filtered_vol,cosines, sines,
                                n_det, out_sz,
                                x=xi, y=zi, z=yi  # need to make sure about this
                            )# in volume (z,y,z)
                            density *= occlusion
                        else:
                            
                            density=query_fbp_point_3d(
                                filtered_vol,cosines, sines,
                                n_det, out_sz,
                                x=xi, y=zi, z=yi  # need to make sure about this
                            )# in volume (z,y,z)
                    else:
                        # print(f"render in DVR mode")
                        if with_AO:
                            local_density=estimate_local_density(volume, x, y, z)
                            # print(f"local_density:{local_density}")
                            occlusion = (1.0 - occulusion_intensity * local_density)*3
                            occlusion = max(0.0, min(occlusion, 1.0))
                            density=trilinear_interp(volume, x, y, z)
                            density *= occlusion
                        else:
                            density=trilinear_interp(volume, x, y, z)
                    alpha = 1.0 - np.exp(-density * step_size)
                    color += (1.0 - opacity) * alpha * density
                    opacity += (1.0 - opacity) * alpha
                t += step_size
            img[i, j] = color
    return img

@njit(parallel=True)
def render_volume(volume, view_matrix, zoom_factor, image_size=256, step_size=0.01,querry=False,filtered_vol=None,cosines=None,sines=None,n_det=None,out_sz=None,with_AO=False):
    if volume is not None:
        X, Z, Y = volume.shape[1],volume.shape[2],volume.shape[0]#num_detectors, num_detectors,num_slices, 
    else:
        X, Z, Y = 2000,2000,1000
    img = np.zeros((image_size, image_size))

    for i in prange(image_size):
        for j in prange(image_size):
            ray_dir = np.dot(view_matrix, np.array([
                2 * (j / (image_size - 1)) - 1.0,
                2 * (i / (image_size - 1)) - 1.0,
                1.0
            ]))
            ray_dir /= np.linalg.norm(ray_dir)
            cam_origin = -zoom_factor * view_matrix[:, 2]

            color = 0.0
            opacity = 0.0
            t = 0.0
            while t < 4.0 and opacity < 0.95:
                pos = cam_origin + t * ray_dir
                pos_norm = (pos + 1.0) / 2.0
                x = pos_norm[0] * (X - 1) # X
                y = pos_norm[1] * (Y - 1) # 
                z = pos_norm[2] * (Z - 1) #
                if 0 <= x < X-1 and 0 <= y < Y-1 and 0 <= z < Z-1:
                    
                    if querry and filtered_vol is not None:
                        # xi, yi, zi = int(x), int(y), int(z)
                        xi, yi, zi = x,y,z
                        # density=0# ignore it here now
                        xi = xi - n_det//2
                        zi = zi - n_det//2
                        
                        if with_AO:
                            local_density=estimate_local_density(volume=volume, x=x, y=y, z=z,querry=querry,filtered_vol=filtered_vol,cosines=cosines, sines=sines,
                            n_det=n_det, out_sz=out_sz)
                            occlusion = 1.0 - 10 * local_density
                            occlusion = max(0.0, min(occlusion, 1.0))
                            density=query_fbp_point_3d(
                                filtered_vol,cosines, sines,
                                n_det, out_sz,
                                x=xi, y=zi, z=yi  # need to make sure about this
                            )# in volume (z,y,z)
                            density *= occlusion
                        else:
                            
                            density=query_fbp_point_3d(
                                filtered_vol,cosines, sines,
                                n_det, out_sz,
                                x=xi, y=zi, z=yi  # need to make sure about this
                            )*0.1# in volume (z,y,z)
                    else:
                        # print(f"render in DVR mode")
                        if with_AO:
                            local_density=estimate_local_density(volume, x, y, z)
                            # print(f"local_density:{local_density}")
                            occlusion = 1.0 - 10 * local_density
                            occlusion = max(0.0, min(occlusion, 1.0))
                            density=trilinear_interp(volume, x, y, z)
                            density *= occlusion
                        else:
                            density=trilinear_interp(volume, x, y, z)*0.1
                    alpha = 1.0 - np.exp(-density * step_size)
                    color += (1.0 - opacity) * alpha * density
                    opacity += (1.0 - opacity) * alpha
                t += step_size
            img[i, j] = color
    return img

def capture_rotation_images(volume, zoom_factor, size, cap_num=10,querry=False,exp="test",filtered_vol=None,cosines=None,sines=None,n_det=None,out_sz=None,with_AO=False,parent_folder=None):
    import cv2

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if with_AO:
        save_dir = os.path.join(parent_folder, f"{exp}_w_AO")
    else:
        save_dir = os.path.join(parent_folder, f"{exp}_wo_AO")
    os.makedirs(save_dir, exist_ok=True)

    pygame.init()
    temp_screen = pygame.display.set_mode((size, size))

    frames = []    
    for idx, yaw_angle in enumerate(np.linspace(0, np.pi, cap_num, endpoint=False)):
        pitch_angle = 0.0
        view_matrix = get_view_matrix(yaw_angle, pitch_angle)
        img = render_volume_non_normalized(volume, view_matrix, zoom_factor, image_size=size,querry=querry,filtered_vol=filtered_vol,cosines=cosines,sines=sines,n_det=n_det,out_sz=out_sz,with_AO=with_AO)

        # 标准化 img
        norm = np.clip(img / (np.max(img)+0.0000001), 0, 1)
        gray_image = (norm * 255).astype(np.uint8)  # (H, W)

        # 做成 RGB 三通道
        rgb_frame = np.stack([gray_image]*3, axis=-1)  # (H, W, 3)

        # 保存图片（保存的时候需要转置！）
        surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))  # (W, H, 3)
        filename = os.path.join(save_dir, f"frame_{idx:02d}.png")
        pygame.image.save(surface, filename)
        print(f"Saved {filename}")

        # 保存标准方向的frame，用于后续视频合成
        frames.append(rgb_frame)

    print(f"All images saved to {save_dir}")

    # ==== 用 cv2 保存成MP4 ====
    video_path = os.path.join(save_dir, "rotation_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (size, size))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV需要BGR
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Video saved to {video_path}")

    return save_dir

def FBP_Recon(angles,proj_log):
    #get rad angles
    rad_angles=np.deg2rad(angles)
    recon = fbp_reconstruct_manual(proj_log, rad_angles)
    print(f"FBP grid recon:{recon.shape}")#
    return recon

# ==== Pygame 交互窗口 ====
def run_renderer(screen_size=10,volume_size=128,exp=None,folder_path=None,angle_file=None,with_AO=False,parent_folder=None):
    
    #some hyper-params
    size = screen_size
    querry=False
    volume=None
    yaw, pitch = 0.0, 0.0
    zoom_factor = 4.0
    dragging = False
    last_mouse = None
    cap_num=10
    
    # preprocess for projections
    # data path
    # folder_path = r"D:\datasets\cryo\Dataset\Synthetic\Micrographs\Clean"
    #
    folder_path=folder_path
    angle_file=angle_file
    #preprocess
    
    
    proj_log=dataLoader(folder_path, resize_to=(screen_size, screen_size))#proj_log；(1000, 1000, 94)
    proj_log = np.transpose(proj_log, (2, 0, 1))  # shape 变成 (N, H, W)
    # print(f"proj_log:{proj_log.shape}")
    # get degrees angles
    angles=load_tilt_angles(os.path.join(folder_path,angle_file))
    # print(f"angles:{angles}")
    
    pygame.init()
    screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()
    
        #projs.shape = (num_angles, num_slices, num_detectors)
    num_angles, num_slices, num_detectors = proj_log.shape
    # 1) pre_cal cos/sin
    rad_angles=np.deg2rad(angles)
    cosines = np.cos(rad_angles)
    sines   = np.sin(rad_angles)

    # 2) pre filter
    filtered_vol = np.zeros((num_slices, num_detectors, num_angles), dtype=np.float32)
    for z in range(num_slices):
        sino = proj_log[:, z, :].T                  # (n_det, n_angles)
        filtered_vol[z] = apply_ramlak_filter(sino)

    num_angles, num_slices, num_detectors = proj_log.shape
    Nz, Ny, Nx = num_slices, num_detectors, num_detectors
    DDVR_data={
        "filtered_vol":filtered_vol,
        "cosines":cosines,
        "sines":sines,
        "n_det":num_detectors,
        "out_sz":num_detectors,
    }
    volume = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    for z in prange(Nz):
        sino = filtered_vol[z]  # (n_det, n_angles)
        for i in range(Ny):
            # 将像素索引映射到以 0 为中心的坐标
            y = i - Ny//2
            for j in range(Nx):
                x = j - Nx//2
                volume[z, i, j] = query_fbp_point_3d(
        filtered_vol, cosines, sines,
        n_det=num_detectors, out_sz=num_detectors,
        x=x, y=y, z=z
    )# be careful about the x,y,z
    
    # 获取最大值和最小值
    
    vmin = np.min(volume)
    vmax = np.max(volume)
    logger.info(f"{exp}-test with size:{screen_size}, Volume min-max: [{vmin}, {vmax}]")          
    """
    test with size:256
    Volume min: -0.03602917
    Volume max: 0.069636874
    
    test with size:512
    Volume min: -0.030534502
    Volume max: 0.049964953
    
    test with size:1000
    Volume min: -0.17711608
    Volume max: 0.40967748
    """     
    save_dir = os.path.join(parent_folder, f"{exp}_COMPARISON")
    os.makedirs(save_dir, exist_ok=True)
    
    volume.tofile(os.path.join(save_dir, f"{exp}_vol_{Nz}x{Ny}x{Nx}.bin"))
    
    querry=False
    # if not querry :
    #     logger.info(f"Process {exp}_DVR recon")
    #     DVR_dir=capture_rotation_images(volume, zoom_factor, size,cap_num,querry=querry,exp=f"{exp}_DVR",with_AO= with_AO,parent_folder=save_dir)  
    # querry=True
    # if querry:
    #     logger.info(f"Process {exp}_DDVR recon")
    #     DDVR_dir=capture_rotation_images(volume, zoom_factor, size,cap_num,querry=querry,exp=f"{exp}_DDVR",filtered_vol=filtered_vol,cosines=cosines,sines=sines,n_det=num_detectors,out_sz=num_detectors,with_AO=with_AO,parent_folder=save_dir)   
    # # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # create_comparison_video_and_images(DVR_dir, DDVR_dir, f"{exp}_DVR", f"{exp}_DDVR", save_dir)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                x, y = pygame.mouse.get_pos()
                dx = (x - last_mouse[0]) * 0.01
                dy = (y - last_mouse[1]) * 0.01
                yaw += dx
                pitch += dy
                last_mouse = (x, y)
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor *= 0.9 if event.y > 0 else 1.1
                zoom_factor = np.clip(zoom_factor, 0.1, 10.0)

        view_matrix = get_view_matrix(yaw, pitch)
        img = render_volume(volume, view_matrix, zoom_factor, image_size=size,querry=querry)

        # 转换为 RGB 显示
        norm = np.clip(img / (np.max(img)+0.0000001), 0, 1)
        surface = pygame.surfarray.make_surface((norm.T * 255).astype(np.uint8).repeat(3).reshape(size, size, 3))
        screen.blit(surface, (0, 0))

        # ==== 添加一个小坐标系指示器 ====
        axis_origin = np.array([40, size - 40])  # 屏幕左下角
        axis_len = 30  # 像素长度

        for vec, color in zip([np.array([1, 0, 0]),  # X Red
                               np.array([0, 1, 0]),  # Y Green
                               np.array([0, 0, 1])], # Z Blue
                              [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            v_rot = view_matrix @ vec
            end = axis_origin + axis_len * v_rot[:2][::-1]  # 注意 y 向下
            pygame.draw.line(screen, color, axis_origin, end.astype(int), 2)

        pygame.display.flip()
        # clock.tick(30)

if __name__ == '__main__':
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(os.getcwd(), f"Full_EXP_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    logger.add(os.path.join(save_dir, "log_output.txt"), format="{time} {level} {message}", level="INFO")
    #basic setting
    screen_size=256
    volume_size=screen_size
    
    #exp setting
    # exp1
    folder_path=r"D:\datasets\cryo\Dataset\Synthetic\Micrographs\Noisy"
    angle_file="tilt.rawtlt"
    with_AO=False
    exp="Syn_Noisy"
    if with_AO:
        exp=f"{exp}_wAO"
    else:
        exp=f"{exp}_woAO"
    run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)
    
    
    # # exp2
    # folder_path=r"D:\datasets\cryo\Dataset\Synthetic\Micrographs\Noisy"
    # angle_file="tilt.rawtlt"
    # with_AO=True
    # exp="Syn_Noisy"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)
    
    # # exp3
    # folder_path=r"D:\datasets\cryo\Dataset\Synthetic\Micrographs\Clean"
    # angle_file="tilt.rawtlt"
    # with_AO=False
    # exp="Syn_Clean"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)
    
    
    # # exp4
    # folder_path=r"D:\datasets\cryo\Dataset\Synthetic\Micrographs\Clean"
    # angle_file="tilt.rawtlt"
    # with_AO=True
    # exp="Syn_Clean"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)


    # # exp5
    # folder_path=r"D:\datasets\cryo\Dataset\Real_CovidInfectedCell\Micrographs\Denoised_BM3D"
    # angle_file='1_1_TomoSer2.rawtlt'
    # with_AO=False
    # exp="Real_Denoised_BM3D"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)
    
    # # exp6
    # folder_path=r"D:\datasets\cryo\Dataset\Real_CovidInfectedCell\Micrographs\Denoised_BM3D"
    # angle_file='1_1_TomoSer2.rawtlt'
    # with_AO=True
    # exp="Real_Denoised_BM3D"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)
    
    # # exp7
    # folder_path=r"D:\datasets\cryo\Dataset\Real_Nanoparticles\Micrographs\longexposure"
    # angle_file='1_1_TomoSer2.rawtlt' 
    # with_AO=False
    # exp="Real_NanoParticles"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)

    # folder_path=r"D:\datasets\cryo\Dataset\Real_Nanoparticles\Micrographs\longexposure"
    # angle_file='1_1_TomoSer2.rawtlt' 
    # with_AO=True
    # exp="Real_NanoParticles"
    # if with_AO:
    #     exp=f"{exp}_wAO"
    # else:
    #     exp=f"{exp}_woAO"
    # run_renderer(screen_size=screen_size,volume_size=volume_size,exp=exp,folder_path=folder_path,angle_file=angle_file,with_AO=with_AO,parent_folder=save_dir)