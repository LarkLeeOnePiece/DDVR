import numpy as np
import pygame
from numba import njit, prange

# ==== Volume generation ====
def generate_volume(size=64):
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    z = np.linspace(-10, 10, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.exp(-((3*X**2 + 5*Y**2 + Z**2)))

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

# ==== Ray marching rendering (Numba accelerated) ====
@njit(parallel=True)
def render_volume(volume, view_matrix, zoom_factor, image_size=256, step_size=0.01):
    D, H, W = volume.shape
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
                x = pos_norm[0] * (D - 1)
                y = pos_norm[1] * (H - 1)
                z = pos_norm[2] * (W - 1)
                if 0 <= x < D-1 and 0 <= y < H-1 and 0 <= z < W-1:
                    xi, yi, zi = int(x), int(y), int(z)
                    density = volume[xi, yi, zi]
                    alpha = 1.0 - np.exp(-density * step_size)
                    color += (1.0 - opacity) * alpha * density
                    opacity += (1.0 - opacity) * alpha
                t += step_size
            img[i, j] = color
    return img

# ==== Pygame 交互窗口 ====
def run_renderer():
    pygame.init()
    size = 256
    screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()

    volume = generate_volume(128)
    yaw, pitch = 0.0, 0.0
    zoom_factor = 3.0
    dragging = False
    last_mouse = None

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
                zoom_factor = np.clip(zoom_factor, 0.5, 10.0)

        view_matrix = get_view_matrix(yaw, pitch)
        img = render_volume(volume, view_matrix, zoom_factor, image_size=size)

        # 转换为 RGB 显示
        norm = np.clip(img / np.max(img), 0, 1)
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
        clock.tick(30)

run_renderer()
