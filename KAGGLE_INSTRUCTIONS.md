# ⚡ Running 3D Perception & Grasping Pipeline with Kaggle GPU (Cloudflare Tunnel)

This guide explains how to stream your physical **Orbbec Gemini E** camera feed from your local machine to **Kaggle's free GPU (Tesla T4 / P100)** using **Cloudflare Tunnels (`cloudflared`)** — **No accounts, tokens, or signups required!**

---

## 🏗️ Architecture

```
 ┌────────────────────────────────────────┐                ┌────────────────────────────────────────┐
 │            LOCAL MACHINE               │                │          KAGGLE NOTEBOOK (GPU)         │
 │                                        │                │                                        │
 │   Orbbec Gemini E Camera (USB)         │                │   FastAPI Server + WebSocket           │
 │               │                        │   WebSockets   │               │                        │
 │               ▼                        │ ─────────────► │               ▼                        │
 │   Local Client (client_streamer.py)    │   Cloudflare   │   Grounded-SAM + 3D Grasp Engine       │
 │   - Reads Color (RGB) + Depth (Z)      │   Tunnel       │   (Runs on Kaggle CUDA GPU)            │
 │   - Sends frames to Kaggle             │   (*.trycloud- │               │                        │
 │               ▲                        │    flare.com)  │               ▼                        │
 │               │                        │ ◄───────────── │   Web GUI / Stream Output              │
 │   Local Display or Browser Dashboard   │                │   - Live 3D Grasp Axes & Telemetry     │
 └────────────────────────────────────────┘                └────────────────────────────────────────┘
```

---

## 📋 Step-by-Step Instructions

### Step 1: Push Your Code to GitHub
Ensure your latest changes are pushed to your GitHub repository:
```bash
git add .
git commit -m "Update to Cloudflare Tunnel for Kaggle"
git push origin main
```

---

### Step 2: Open Kaggle & Start a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and click **Create** $\rightarrow$ **New Notebook**.
2. In the right-hand panel under **Notebook settings**:
   * **Accelerator**: Select **GPU T4 x2** (or **GPU P100**).
   * **Internet**: Turn **ON** (Required to download models and open the tunnel).

---

### Step 3: Run the Cells in Kaggle

#### Cell 1: Check GPU
```python
!nvidia-smi
```

#### Cell 2: Install Python Packages
```python
!pip install -q fastapi uvicorn websockets opencv-python transformers timm supervision safetensors open3d
```

#### Cell 3: Install Official Cloudflare Tunnel Client (`cloudflared`)
*No account or API key needed:*
```python
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
!cloudflared --version
```

#### Cell 4: Clone Your GitHub Repository
```python
# Replace with your actual GitHub repo URL:
!git clone https://github.com/<your-username>/Segmentation_based_vision.git
%cd Segmentation_based_vision
```

#### Cell 5: Start Cloudflare Tunnel
```python
import subprocess, time, re, os

# Start cloudflared pointing to port 8000
!nohup cloudflared tunnel --url http://127.0.0.1:8000 > cloudflared.log 2>&1 &

print("[*] Starting Cloudflare Tunnel...")
time.sleep(4)

# Find and print your public URL
for _ in range(15):
    if os.path.exists("cloudflared.log"):
        with open("cloudflared.log", "r") as f:
            matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
            if matches:
                print("=" * 65)
                print(f"🎉 PUBLIC TUNNEL URL: {matches[0]}")
                print("=" * 65)
                break
    time.sleep(1)
```
*(Copy the generated URL, e.g. `https://random-words.trycloudflare.com`)*

#### Cell 6: Start Kaggle GPU Perception Server
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

Open PowerShell on your local machine (with your Orbbec Gemini E plugged in) and run:

```powershell
.venv\Scripts\python.exe client_streamer.py --server-url https://random-words.trycloudflare.com --prompt "cup"
```
*(Replace `https://random-words.trycloudflare.com` with the exact URL printed by Cloudflare in Step 3)*

---

### 🖥️ Viewing the Output
1. **On your Local PC:** A window opens displaying the live camera feed with real-time round-trip latency (RTT) and terminal logs printing 3D coordinates and Grasp RPY angles.
2. **In any Web Browser:** Open `https://random-words.trycloudflare.com` in Chrome/Edge on your laptop or phone! You will see:
   * Real-time annotated video with SAM masks and 3D Grasp axes (Red/Green/Blue).
   * Live updating JSON telemetry panel.
   * Target prompt input field to change objects on the fly.

### Controls:
* Press **`p`** in the local OpenCV preview window to switch prompts (e.g. `bottle`, `mouse`, `charger`).
* Press **`q`** or **`ESC`** to exit cleanly.
