# LoRa VAE Image Transmission
**BSc Computer Science Dissertation — York St John University 2026**
 
Sending compressed images over LoRa using a Variational Autoencoder (VAE). The VAE compresses a 28x28 image down to just 3 LoRa packets, transmits wirelessly between two Raspberry Pis, and reconstructs the image at the other end.
 
---
 
## How It Works
 
```
Pi A (Sender)                        Pi B (Receiver)
─────────────                        ────────────────
Load image                           Wait for packets
Encode → 32 values      ──LoRa──→   Receive 3 packets
Split into 3 packets                 Decode → image
Send over LoRa                       Calculate SSIM
```
 
## Results
 
| Metric | Value |
|--------|-------|
| SSIM (no loss) | 0.9434 |
| Packets per image | 3 |
| Payload size | 128 bytes |
| Compression ratio | 24:1 |
| Encode time | 14.1ms |
| Decode time | 16.8ms |
 

 
## Hardware
 
- 2x Raspberry Pi 4 Model B
- 2x RFM9x LoRa Module (433MHz)
- 2x INA219 Power Sensor
- Jumper wires
 
---
 
## Quick Start
 
**Install dependencies on both Pis:**
```bash
pip3 install torch==2.6.0 torchvision==0.21.0 --break-system-packages
pip3 install adafruit-circuitpython-rfm9x adafruit-circuitpython-ina219 --break-system-packages
pip3 install scikit-image pillow numpy --break-system-packages
```
  
---
