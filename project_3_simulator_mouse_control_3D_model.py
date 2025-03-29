import cv2
import numpy as np
import trimesh
from math import cos, sin, pi

# Initial camera setup
camera_pos = np.array([0, 2, -5], dtype=np.float32)
camera_rotation = np.array([0, 0, 0], dtype=np.float32)
prev_mouse_pos = None
mouse_pressed = False
right_mouse_pressed = False

# Projection settings
focal_length = 500
screen_width, screen_height = 1200, 800
center = (screen_width // 2, screen_height // 2)

def rotation_matrix(angles):
    """Returns combined 3D rotation matrix from pitch, yaw, roll."""
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
    Ry = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
    Rz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def project_point(point):
    """Projects a 3D point into 2D space."""
    rot_matrix_val = rotation_matrix(camera_rotation)
    point = rot_matrix_val @ (point - camera_pos)
    if point[2] <= 0:
        return None
    x = (point[0] * focal_length) / point[2] + center[0]
    y = (point[1] * focal_length) / point[2] + center[1]
    return int(x), int(y), point[2]

def calculate_lighting(face_normal, light_dir):
    """Calculates diffuse lighting intensity."""
    face_normal = face_normal / np.linalg.norm(face_normal)
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = abs(np.dot(face_normal, light_dir))
    return max(0.2, min(1.0, intensity))

def draw_model(img, vertices, faces):
    """Draws the 3D model on the given image."""
    light_dir = np.array([1, 1, 1])
    face_depths = []

    for i, face in enumerate(faces):
        points_3d = [project_point(vertices[j]) for j in face]
        if None in points_3d:
            face_depths.append((float('inf'), i))
            continue
        avg_z = sum(p[2] for p in points_3d) / len(points_3d)
        face_depths.append((avg_z, i))

    face_depths.sort(reverse=True)

    for _, face_idx in face_depths:
        face = faces[face_idx]
        points_3d = [project_point(vertices[j]) for j in face]
        if None in points_3d:
            continue

        v1 = vertices[face[1]] - vertices[face[0]]
        v2 = vertices[face[2]] - vertices[face[0]]
        normal = np.cross(v1, v2)

        light_intensity = calculate_lighting(normal, light_dir)
        base_color = np.array([200, 200, 200])
        color = tuple(map(int, base_color * light_intensity))

        points = np.array([(p[0], p[1]) for p in points_3d])
        cv2.fillPoly(img, [points], color)
        cv2.polylines(img, [points], True, (255, 255, 255), 1)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse input for camera control."""
    global prev_mouse_pos, camera_pos, camera_rotation, mouse_pressed, right_mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        prev_mouse_pos = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        right_mouse_pressed = True
        prev_mouse_pos = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if prev_mouse_pos is None:
            prev_mouse_pos = (x, y)
            return

        dx = x - prev_mouse_pos[0]
        dy = y - prev_mouse_pos[1]

        if mouse_pressed:
            camera_pos[0] -= dx * 0.01
            camera_pos[2] -= dy * 0.01
        elif right_mouse_pressed:
            camera_rotation[0] += dy * 0.005
            camera_rotation[1] += dx * 0.005

        prev_mouse_pos = (x, y)
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = (flags >> 16) & 0xffff
        if delta > 32767:
            delta -= 65536
        camera_pos[2] += delta * 0.005
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        prev_mouse_pos = None
    elif event == cv2.EVENT_RBUTTONUP:
        right_mouse_pressed = False
        prev_mouse_pos = None

def main():
    window_name_sim = "3D Model Viewer (Mouse Controlled)"
    cv2.namedWindow(window_name_sim, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name_sim, mouse_callback)

    # Load and transform the model
    model = trimesh.load('ship.obj', force='mesh')

    # Fix orientation: rotate X (-90°) to upright, then Z (-90°) to face camera
    rotation_x = trimesh.transformations.rotation_matrix(-pi / 2, [1, 0, 0])
    rotation_z = trimesh.transformations.rotation_matrix(-pi / 2, [0, 0, 1])
    model.apply_transform(rotation_x)
    model.apply_transform(rotation_z)

    # Scale down
    scale_factor = 0.10
    model_vertices = np.array(model.vertices, dtype=np.float32) * scale_factor
    model_faces = np.array(model.faces, dtype=np.int32)

    # Create gradient background
    gradient_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    for y in range(screen_height):
        color = int(150 - (y * 0.2))
        cv2.line(gradient_bg, (0, y), (screen_width, y), (color, color, color+50), 1)

    while True:
        img = gradient_bg.copy()
        draw_model(img, model_vertices, model_faces)

        cv2.putText(img, "Mouse Controls:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Left Click: Pan", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Right Click: Rotate", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Mouse Wheel: Zoom", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name_sim, img)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
