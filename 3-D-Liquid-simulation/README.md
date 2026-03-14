# 3D Liquid Simulation with Hand Gesture Control

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-MediaPipe-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-RealTime-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An interactive **real-time 3D liquid simulation** controlled using **hand gestures through your webcam**.

This project combines **computer vision**, **physics simulation**, and **real-time rendering** to create a cube filled with dynamic liquid particles that react to your hand movements.

The cube rotates, tilts, and resizes based on detected gestures while the liquid inside behaves according to simulated physics such as **gravity, particle interaction, and collision with cube walls**.

---

# Demo

### Simulation Preview

*(Add a GIF or screen recording here)*

Example:

```
docs/demo.gif
```

Then include it like this:

```markdown
![Demo](docs/demo.gif)
```

You can record a GIF using tools like **OBS Studio**, **ScreenToGif**, or **ShareX**.

---

# Features

• Real-time **hand gesture recognition**
• Interactive **3D cube rotation**
• **Particle-based liquid simulation**
• Smooth **fluid rendering using Gaussian blur**
• **Depth-based color gradients** for realistic liquid appearance
• Gesture-controlled **cube resizing**
• Dynamic **rotation inertia and flick interaction**
• Runs entirely on **CPU with webcam**

---

# Controls

| Gesture / Key         | Action                  |
| --------------------- | ----------------------- |
| Open Hand             | Rotate cube             |
| Fast Hand Movement    | Increase spin speed     |
| Closed Fist           | Freeze cube rotation    |
| Pinch (Thumb + Index) | Resize cube             |
| **Q key**             | **Quit the simulation** |

### How to Quit

Press **`Q`** on your keyboard while the simulation window is active.

The program will safely close:

```
cap.release()
cv2.destroyAllWindows()
```

You can also stop the program from the terminal with:

```
Ctrl + C
```

---

# Technologies Used

• Python
• OpenCV – camera input and rendering
• MediaPipe – real-time hand tracking
• NumPy – numerical computations
• Particle-based physics simulation

---

# Installation

## 1. Clone the Repository

```
git clone https://github.com/yourusername/3D-LiquidSimulation.git
cd 3D-LiquidSimulation
```

---

## 2. Create Virtual Environment (Recommended)

```
python -m venv venv
```

Activate it:

**Windows**

```
venv\Scripts\activate
```

**Mac / Linux**

```
source venv/bin/activate
```

---

## 3. Install Dependencies

```
pip install mediapipe opencv-python numpy
```

---

# Download the Hand Tracking Model

Download the **MediaPipe hand tracking model**:

```
hand_landmarker.task
```

Download link:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Place it in the project folder:

```
3-D-Liquid-simulation/
```

---

# Project Structure

```
3D-LiquidSimulation
│
├── 3-D-Liquid-simulation
│   ├── 3-D liquid simulation.py
│   ├── hand_landmarker.task
│   ├── README.md
│   └── LICENSE
│
└── venv
```

---

# Running the Project

Navigate into the simulation folder:

```
cd 3-D-Liquid-simulation
```

Run the program:

```
python "3-D liquid simulation.py"
```

A window titled **Blue Liquid Cube** will appear and your webcam will start tracking your hand.

---

# How It Works

## Hand Tracking

The program uses MediaPipe to detect **21 hand landmarks** in real time.

These landmarks include:

• Wrist
• Finger joints
• Fingertips

These points allow the system to determine **hand gestures and motion direction**.

---

## Cube Interaction

The fingertip centroid is used to calculate movement velocity.

Rotation sensitivity dynamically scales with motion speed:

• slow movement → slow cube rotation
• fast flick → rapid spinning cube

---

## Liquid Simulation

The cube contains **350 simulated particles**.

Each particle follows physics rules including:

• gravity
• particle repulsion
• wall collisions
• velocity damping

This creates the effect of **sloshing liquid inside the cube**.

---

## Rendering

Particles are rendered as **soft Gaussian blobs**.

Overlapping blobs blend together to create a **smooth fluid surface effect**.

Colors vary based on:

• depth
• velocity
• lighting approximation

---

# Requirements

• Python 3.9+
• Webcam
• Windows / Mac / Linux

No GPU required.

---

# Future Improvements

• Glass cube rendering
• Surface tension simulation
• GPU acceleration for thousands of particles
• WebGL browser version
• AR interaction

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# Author

**Subham**

Undergraduate student exploring **Machine Learning, Computer Vision, and Interactive Simulations**.

---

# Acknowledgements

• MediaPipe team for the hand tracking model
• OpenCV community
• Real-time particle physics simulation techniques used in graphics engines
