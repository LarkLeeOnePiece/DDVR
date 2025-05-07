import numpy as np
from numba import njit, prange
import cv2
import os
# ==== Volume generation ====
def generate_volume(size=64):
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    z = np.linspace(-10, 10, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.exp(-((3*X**2 + 5*Y**2 + Z**2)))

@njit
def trilinear_interp(volume, x, y, z):
    D, H, W = volume.shape  # D=y, H=z, W=x
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    zi = int(np.floor(z))

    if xi < 0 or yi < 0 or zi < 0 or xi+1 >= W or yi+1 >= D or zi+1 >= H:
        return 0.0

    dx = x - xi
    dy = y - yi
    dz = z - zi

    c000 = volume[yi,     zi,     xi    ]
    c100 = volume[yi,     zi,     xi + 1]
    c010 = volume[yi,     zi + 1, xi    ]
    c110 = volume[yi,     zi + 1, xi + 1]
    c001 = volume[yi + 1, zi,     xi    ]
    c101 = volume[yi + 1, zi,     xi + 1]
    c011 = volume[yi + 1, zi + 1, xi    ]
    c111 = volume[yi + 1, zi + 1, xi + 1]

    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dz) + c10 * dz
    c1 = c01 * (1 - dz) + c11 * dz

    c = c0 * (1 - dy) + c1 * dy
    return c

def create_comparison_grid(folder1, folder2, title1, title2, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    # 排序读取所有图片
    images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png', '.jpg'))])
    images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png', '.jpg'))])

    assert len(images1) == len(images2), "两个文件夹里的图片数量必须一致！"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # 白色字体
    background_color = (0, 0, 0)  # 黑色背景条

    for idx, (img_path1, img_path2) in enumerate(zip(images1, images2)):
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        assert h1 == h2, "两张图的高度必须一致！"

        # 在上方加黑色背景条
        title_height = 50
        img1_with_title = np.zeros((h1 + title_height, w1, 3), dtype=np.uint8)
        img2_with_title = np.zeros((h2 + title_height, w2, 3), dtype=np.uint8)

        img1_with_title[title_height:, :, :] = img1
        img2_with_title[title_height:, :, :] = img2

        # 写标题
        cv2.putText(img1_with_title, title1, (10, int(title_height*0.7)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(img2_with_title, title2, (10, int(title_height*0.7)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # 横向拼接
        combined = np.hstack((img1_with_title, img2_with_title))

        save_path = os.path.join(save_folder, f"frame_{idx:03d}.png")
        cv2.imwrite(save_path, combined)
        print(f"Saved {save_path}")

    print(f"Save comparison: {save_folder}")

    
def create_comparison_video_and_images(folder1, folder2, title1, title2, save_folder, fps=1):
    os.makedirs(save_folder, exist_ok=True)
    images_folder = os.path.join(save_folder, "frames")
    os.makedirs(images_folder, exist_ok=True)

    images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png', '.jpg'))])
    images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png', '.jpg'))])

    assert len(images1) == len(images2), "The number of images should be the same！"

    img1 = cv2.imread(images1[0])
    img2 = cv2.imread(images2[0])

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    assert h1 == h2 and w1 == w2, "The resolution should be the same！"

    title_height = 50  # 顶部标题高度
    frame_height = title_height + h1
    frame_width = w1 * 3  # MethodA + MethodB + Error Map

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(save_folder, "comparison_with_error.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = frame_height / 1024  # 或者 W / 512，根据你默认设计的参考尺寸来缩放
    font_scale = font_scale
    font_thickness = 1
    text_color = (255, 255, 255)  # 白色字

    for idx, (img_path1, img_path2) in enumerate(zip(images1, images2)):
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        # ==== 生成 Error Map ====
        error = cv2.absdiff(img1, img2)
        error_gray = cv2.cvtColor(error, cv2.COLOR_BGR2GRAY)
        error_color = cv2.applyColorMap(error_gray, cv2.COLORMAP_JET)

        # ==== 创建大画布 ====
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # ==== 写标题 ====
        cv2.putText(frame, title1, (2, int(title_height*0.7)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, title2, (w1 + 2, int(title_height*0.7)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, "Error Map", (2*w1 + 2, int(title_height*0.7)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # ==== 放置三张图 ====
        frame[title_height:title_height+h1, 0:w1, :] = img1
        frame[title_height:title_height+h1, w1:w1*2, :] = img2
        frame[title_height:title_height+h1, w1*2:w1*3, :] = error_color

        # ==== 保存单帧图片 ====
        frame_path = os.path.join(images_folder, f"frame_{idx:03d}.png")
        cv2.imwrite(frame_path, frame)

        # ==== 写入视频 ====
        video_writer.write(frame)

        # print(f"保存和写入第 {idx} 帧")

    video_writer.release()
    print(f"video save to : {video_path}, image save to: {images_folder}")