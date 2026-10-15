# ⚡ Running 3D Perception & Grasping Pipeline with Kaggle GPU

This guide explains how to stream your physical **Orbbec Gemini E** camera feed from your local machine to **Kaggle's free GPU (Tesla T4 / P100)** to run **Grounding DINO + SAM + 6-DoF Grasp Estimation** at high speed with low latency.

---

## 🏗️ How the Architecture Works

1. **Local Machine (Your PC):**
   * Physical **Orbbec Gemini E** camera is plugged in via USB.
   * `client_streamer.py` reads Color (RGB) + Depth (Z) frames, compresses them into compact binary packets, and streams them to Kaggle over a secure WebSocket tunnel.
2. **Kaggle Notebook (Cloud GPU):**
   * Runs `kaggle_server.py` on CUDA.
   * Receives RGB-D frames, runs zero-shot detection with Grounding DINO, segments with SAM, computes 3D coordinates & 6-DoF grasp poses (XYZ + RPY + Quaternions).
   * Serves an interactive **Web Dashboard** so you can view the live output in any browser!

---

## 📋 Step-by-Step Instructions

### Step 1: Push Your Code to GitHub
Ensure your latest changes (including `kaggle_server.py`, `client_streamer.py`, and `perception_pipeline/`) are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "Add Kaggle GPU streaming support"
git push origin main
```

---

### Step 2: Open Kaggle & Start a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and click **Create** $\rightarrow$ **New Notebook**.
2. In the right-hand panel under **Notebook settings**:
   * **Accelerator**: Select **GPU T4 x2** (or **GPU P100**).
   * **Internet**: Turn **ON** (Required to download models and create tunnel).

---

### Step 3: Run the Cells in Kaggle

#### Cell 1: Install Dependencies
```python
!pip install -q fastapi uvicorn websockets pyngrok opencv-python transformers timm supervision safetensors open3d
```

#### Cell 2: Clone Your Repository
```python
# Replace with your actual GitHub repo URL:
!git clone https://github.com/<your-username>/Segmentation_based_vision.git
%cd Segmentation_based_vision
```

#### Cell 3: Set Up Free Ngrok Tunnel
Get a free token from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) (takes 10 seconds):
```python
from pyngrok import ngrok

# Paste your free authtoken:
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN_HERE")

# Open public tunnel to port 8000
tunnel = ngrok.connect(8000)
print("=" * 60)
print(f"👉 PUBLIC TUNNEL URL: {tunnel.public_url}")
print("=" * 60)
```
*(Copy the generated URL, e.g., `https://xxxx-xx-xx.ngrok-free.app`)*

#### Cell 4: Start the GPU Perception Server
```python
!python kaggle_server.py
```
*You will see:*
```
[*] Initializing Grounded-SAM on device: cuda (Kaggle GPU)...
[*] Models loaded and warm!
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

### Step 4: Stream from Your Local Machine

Open a terminal in your project directory on your local machine and run:

```powershell
.venv\Scripts\python.exe client_streamer.py --server-url https://xxxx-xx-xx.ngrok-free.app --prompt "cup"
```
*(Replace `https://xxxx-xx-xx.ngrok-free.app` with the URL printed in Step 3)*

### What Happens:
1. `client_streamer.py` connects to Kaggle via WebSocket.
2. It captures the live feed from your **Orbbec Gemini E** camera.
3. Kaggle processes each frame on its GPU in ~**50–150ms** and streams back:
   * 3D Coordinates $(X, Y, Z)$ w.r.t camera.
   * Grasp Orientation (Roll, Pitch, Yaw, Quaternions).
   * Gripper Width & Pre-grasp standoff vector.
4. **Live Web Browser Dashboard**: Open `https://xxxx-xx-xx.ngrok-free.app` in any web browser to see the live video stream, masks, 3D grasp axes, and JSON telemetry in real time!

---

### Interactive Controls (on Local Machine)
* Press **`p`** in the local OpenCV preview window to change target prompt dynamically (e.g., `bottle`, `mouse`, `charger`).
* Press **`q`** or **`ESC`** to quit.
