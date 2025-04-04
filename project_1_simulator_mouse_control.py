import cv2
import numpy as np
from math import cos, sin, pi

# Camera settings and user interaction state
camera_pos = np.array([0, 2, -12], dtype=np.float32)
camera_rotation = np.array([0, 0, 0], dtype=np.float32)
prev_mouse_pos = None
mouse_pressed = False
right_mouse_pressed = False

# Ship dimensions
ship_length = 14   # Front to back
ship_width = 5     # Side to side
ship_height = 2.5  # Base height of main deck

# Ship structure definition (vertices)
ship_vertices = np.array([
    # Main deck corners (bottom)
    [-ship_length/2, 0, -ship_width/2],      # 0: Front left
    [-ship_length/2, 0,  ship_width/2],      # 1: Front right
    [ ship_length/2, 0, -ship_width/2],      # 2: Back left
    [ ship_length/2, 0,  ship_width/2],      # 3: Back right

    # Deck wall corners (top)
    [-ship_length/2, ship_height, -ship_width/2],  # 4
    [-ship_length/2, ship_height,  ship_width/2],  # 5
    [ ship_length/2, ship_height, -ship_width/2],  # 6
    [ ship_length/2, ship_height,  ship_width/2],  # 7

    # Bridge (captain's cabin)
    [-1.5, ship_height, -ship_width/4],            # 8
    [-1.5, ship_height,  ship_width/4],            # 9
    [ 1.5, ship_height, -ship_width/4],            # 10
    [ 1.5, ship_height,  ship_width/4],            # 11
    [-1.5, ship_height*1.6, -ship_width/4],        # 12
    [-1.5, ship_height*1.6,  ship_width/4],        # 13
    [ 1.5, ship_height*1.6, -ship_width/4],        # 14
    [ 1.5, ship_height*1.6,  ship_width/4],        # 15

    # Masts
    [-ship_length/3, ship_height, 0],              # 16: Front mast base
    [-ship_length/3, ship_height*3.2, 0],          # 17: Front mast top
    [ ship_length/3, ship_height, 0],              # 18: Rear mast base
    [ ship_length/3, ship_height*3.2, 0]           # 19: Rear mast top
], dtype=np.float32)

# Ship surfaces defined by vertex indices
faces = [
    [0, 1, 3, 2],  # Main deck base
    [0, 1, 5, 4],  # Front wall
    [2, 3, 7, 6],  # Rear wall
    [0, 2, 6, 4],  # Left wall
    [1, 3, 7, 5],  # Right wall

    # Bridge (captain's cabin)
    [8, 9, 11, 10],    # Base
    [8, 9, 13, 12],    # Front
    [10, 11, 15, 14],  # Back
    [8, 10, 14, 12],   # Left
    [9, 11, 15, 13],   # Right
    [12, 13, 15, 14],  # Roof
]

# Line elements for masts
lines = [
    (16, 17),
    (18, 19)
]

# Face base colors
face_colors = [
    (139, 69, 19),    # Deck - wood
    (169, 169, 169),  # Front wall - grey
    (169, 169, 169),  # Rear wall - grey
    (169, 169, 169),  # Left wall
    (169, 169, 169),  # Right wall
    (139, 69, 19),    # Bridge base
    (112, 128, 144),  # Bridge front
    (112, 128, 144),  # Bridge back
    (112, 128, 144),  # Bridge left
    (112, 128, 144),  # Bridge right
    (47, 79, 79),     # Roof
]

# Projection setup
focal_length = 500
width, height = 1200, 800
center = (width // 2, height // 2)

def rotation_matrix(angles):
    """Returns a 3D rotation matrix for the given pitch, yaw, and roll."""
    rx, ry, rz = angles

    Rx = np.array([
        [1, 0, 0],
        [0, cos(rx), -sin(rx)],
        [0, sin(rx), cos(rx)]
    ])

    Ry = np.array([
        [cos(ry), 0, sin(ry)],
        [0, 1, 0],
        [-sin(ry), 0, cos(ry)]
    ])

    Rz = np.array([
        [cos(rz), -sin(rz), 0],
        [sin(rz), cos(rz), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def project_point(point):
    """Projects a 3D point onto 2D space with perspective projection."""
    rot_matrix = rotation_matrix(camera_rotation)
    point = rot_matrix @ (point - camera_pos)

    if point[2] <= 0:
        return None

    x = (point[0] * focal_length) / point[2] + center[0]
    y = (point[1] * focal_length) / point[2] + center[1]
    return int(x), int(y), point[2]

def calculate_lighting(face_normal, light_dir):
    """Returns shading intensity based on face normal and light direction."""
    face_normal = face_normal / np.linalg.norm(face_normal)
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = abs(np.dot(face_normal, light_dir))
    return max(0.2, min(1.0, intensity))

def draw_ship(img):
    """Renders the 3D ship model with shading and masts."""
    light_dir = np.array([1, 1, 1])
    face_depths = []

    for i, face in enumerate(faces):
        points_3d = [project_point(ship_vertices[j]) for j in face]
        if None in points_3d:
            face_depths.append((float('inf'), i))
            continue
        avg_z = sum(p[2] for p in points_3d) / len(points_3d)
        face_depths.append((avg_z, i))

    face_depths.sort(reverse=True)

    for _, face_idx in face_depths:
        face = faces[face_idx]
        points_3d = [project_point(ship_vertices[j]) for j in face]

        if None in points_3d:
            continue

        v1 = ship_vertices[face[1]] - ship_vertices[face[0]]
        v2 = ship_vertices[face[2]] - ship_vertices[face[0]]
        normal = np.cross(v1, v2)

        light_intensity = calculate_lighting(normal, light_dir)
        base_color = np.array(face_colors[face_idx])
        color = tuple(map(int, base_color * light_intensity))

        points = np.array([(p[0], p[1]) for p in points_3d])
        cv2.fillPoly(img, [points], color)
        cv2.polylines(img, [points], True, (255, 255, 255), 1)

    for line in lines:
        pt1 = project_point(ship_vertices[line[0]])
        pt2 = project_point(ship_vertices[line[1]])
        if pt1 is not None and pt2 is not None:
            cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (101, 67, 33), 2)

def mouse_callback(event, x, y, flags, param):
    """Handles camera movement via mouse interaction."""
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
            camera_pos[2] += dy * 0.01
            camera_pos[0] += dx * 0.01
        elif right_mouse_pressed:
            camera_pos[1] += dy * 0.01

        prev_mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
    elif event == cv2.EVENT_RBUTTONUP:
        right_mouse_pressed = False

def main():
    window_name = "Ship Deck Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient (sky-like)
        for y in range(height):
            color = int(150 - (y * 0.2))
            cv2.line(img, (0, y), (width, y), (color, color, color+50), 1)

        draw_ship(img)

        cv2.putText(img, "Controls:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Left click + up/down: Move forward/backward", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Left click + left/right: Move sideways", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Right click + up/down: Move up/down", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
