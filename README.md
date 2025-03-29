🛠 Requirements
You need Python 3 installed.
All three projects use these Python libraries:

numpy → for working with 3D points

opencv-python → for drawing and webcam

trimesh → only for the 3rd project (to load a 3D model)

To install them:
pip install numpy opencv-python trimesh

Also, the third project needs a 3D model file: ship.obj
Make sure it's in the same folder as the code.

▶️ How to Run Each Project

1️⃣ Basic 3D Ship (Mouse Control)

This project creates a 3D ship using only code (NumPy arrays). You can move the camera using the mouse. It runs in a window with a simple 3D scene.

Controls:
🖱 Left click + drag mouse → move forward/backward, left/right

🖱 Right click + drag mouse → move up/down

2️⃣ 3D Ship with Face Tracking

This project adds face tracking using your webcam. When your face moves, the camera also moves. You don't need to click anything — just use your face!

Controls (Face):
Move face left/right → camera moves left/right

👈👉 Move face up/down → camera moves up/down

👆👇 Move closer/farther → camera zooms in/out

💡 Webcam must be working. Good light helps better detection.

3️⃣ 3D Ship Model Viewer (.obj file)

This project loads a real 3D model from a .obj file. It shows the ship with lighting and perspective. You can move and rotate the camera using your mouse.

Controls:
🖱 Left click + drag → move camera (pan)

🖱 Right click + drag → rotate view (orbit)

🖱 Mouse wheel → zoom in/out

✅ Make sure the ship.obj file is in the same folder.

