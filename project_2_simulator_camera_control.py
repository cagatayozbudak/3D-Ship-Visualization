import cv2
import numpy as np
from math import cos, sin, pi

# Camera and interaction state
camera_pos = np.array([0, 2, -10], dtype=np.float32)
camera_rotation = np.array([0, 0, 0], dtype=np.float32)
prev_mouse_pos = None
mouse_pressed = False
right_mouse_pressed = False

# Face tracking setup
prev_face_center = None
prev_face_size = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ship dimensions
length = 10
width = 4
height = 2

# Define 3D ship geometry
ship_vertices = np.array([
    # Main deck
    [-length/2, 0, -width/2], [ -length/2, 0,  width/2],
    [ length/2, 0, -width/2], [  length/2, 0,  width/2],
    
    # Deck walls
    [-length/2, height, -width/2], [-length/2, height, width/2],
    [ length/2, height, -width/2], [ length/2, height,  width/2],
    
    # Bridge (captain’s cabin)
    [-1, height, -width/4], [-1, height,  width/4],
    [ 1, height, -width/4], [ 1, height,  width/4],
    [-1, height*1.5, -width/4], [-1, height*1.5, width/4],
    [ 1, height*1.5, -width/4], [ 1, height*1.5,  width/4],
    
    # Masts
    [-length/4, height, 0], [-length/4, height*2.5, 0],
    [ length/4, height, 0], [ length/4, height*2.5, 0]
], dtype=np.float32)

# Faces of the ship
faces = [
    [0, 1, 3, 2],  # Deck
    [0, 1, 5, 4], [2, 3, 7, 6],  # Front / Back walls
    [0, 2, 6, 4], [1, 3, 7, 5],  # Side walls

    # Bridge
    [8, 9, 11, 10],
    [8, 9, 13, 12],
    [10, 11, 15, 14],
    [8, 10, 14, 12],
    [9, 11, 15, 13],
    [12, 13, 15, 14]
]

# Lines for mast visualization
lines = [(16, 17), (18, 19)]

# Colors for each face
face_colors = [
    (139, 69, 19),    # Deck
    (169, 169, 169),  # Walls
    (169, 169, 169),
    (169, 169, 169),
    (169, 169, 169),
    (139, 69, 19),    # Bridge base
    (112, 128, 144),  # Bridge sides
    (112, 128, 144),
    (112, 128, 144),
    (112, 128, 144),
    (47, 79, 79)      # Bridge roof
]

# Projection settings
focal_length = 500
width, height = 1200, 800
center = (width // 2, height // 2)

def rotation_matrix(angles):
    """Returns a 3D rotation matrix from pitch, yaw, roll."""
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
    """Projects a 3D point into 2D space with perspective."""
    rot_matrix = rotation_matrix(camera_rotation)
    point = rot_matrix @ (point - camera_pos)

    if point[2] <= 0:
        return None

    x = (point[0] * focal_length) / point[2] + center[0]
    y = (point[1] * focal_length) / point[2] + center[1]
    return int(x), int(y), point[2]

def calculate_lighting(face_normal, light_dir):
    """Calculates light intensity based on surface normal and light direction."""
    face_normal = face_normal / np.linalg.norm(face_normal)
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = abs(np.dot(face_normal, light_dir))
    return max(0.2, min(1.0, intensity))

def draw_ship(img):
    """Renders the 3D ship onto the image with lighting and masts."""
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
    """Mouse interaction for camera movement."""
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

def process_face_movement(frame):
    """Detects face and adjusts camera position based on its movement."""
    global prev_face_center, prev_face_size, camera_pos

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        face_center = np.array([x + w/2, y + h/2])
        face_size = w * h

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        screen_center = np.array([frame.shape[1]/2, frame.shape[0]/2])

        if prev_face_center is not None and prev_face_size is not None:
            delta_center = face_center - prev_face_center
            relative_pos = face_center - screen_center
            size_ratio = face_size / prev_face_size

            camera_pos[0] -= (delta_center[0] * 0.005 + relative_pos[0] * 0.001)
            camera_pos[1] -= (delta_center[1] * 0.005 + relative_pos[1] * 0.001)

            if abs(size_ratio - 1) > 0.05:
                camera_pos[2] += (size_ratio - 1) * 3.5

        prev_face_center = face_center
        prev_face_size = face_size

        # Draw face and screen centers
        cv2.circle(frame, tuple(screen_center.astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(face_center.astype(int)), 5, (0, 0, 255), -1)

    return frame

def main():
    window_name_sim = "Ship Deck Simulation"
    window_name_cam = "Webcam View"

    cv2.namedWindow(window_name_sim, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_cam, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_face_movement(frame)

        img = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            color = int(150 - (y * 0.2))
            cv2.line(img, (0, y), (width, y), (color, color, color+50), 1)

        draw_ship(img)

        # Display control info
        cv2.putText(img, "Controls:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Move your face left/right: Move camera left/right", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Move your face up/down: Move camera up/down", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Lean in/out: Move camera forward/backward", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name_sim, img)
        cv2.imshow(window_name_cam, frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
