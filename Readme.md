# LoRa VAE Image Transmission

**BSc Computer Science Dissertation — York St John University 2026**

Sending compressed images over LoRa using an Improved Variational Autoencoder (VAE). The VAE encodes a 28x28 image into a 64-dimensional latent vector, splits it across 6 LoRa packets, transmits wirelessly between two Raspberry Pis, and reconstructs the image at the receiver — evaluating quality via SSIM under simulated packet loss.

---

## How It Works

```
Pi A (Sender)                              Pi B (Receiver)
-------------                              ----------------
Load image (28x28)                         Wait for packets
Encode ? latent (64 values)                Receive up to 6 packets
Split into 6 LoRa packets    --LoRa-->     Reconstruct missing data
Send over LoRa                             Decode ? output image
Log power via INA219                       Calculate SSIM + latency
```

---

## Files

| File | Description |
|------|-------------|
| `New_LoRa_Sender.py` | Main sender — encodes image, splits latent, transmits 6 packets |
| `New_LoRa_Receiver.py` | Main receiver — receives packets, reconstructs, decodes, evaluates |
| `LoRa_Sender_test.py` | Earlier prototype sender (basic VAE, 3 packets) |
| `LoRa_Receiver_Test.py` | Earlier prototype receiver |
| `Packet Loss/` | Packet loss simulation scripts and result logs |

---

## Results

**No packet loss (baseline):**

| Metric | Value |
|--------|-------|
| SSIM | 0.9434 |
| Packets per image | 6 |
| Latent size | 64 |
| Encode time | ~14 ms |
| Decode time | ~17 ms |

**SSIM under simulated packet loss:**

| Packets Dropped | Packets Received | SSIM |
|:--------------:|:----------------:|:----:|
| 0 | 6 / 6 | 0.9434 |
| 1 | 5 / 6 | ~0.84 |
| 2 | 4 / 6 | ~0.73 |
| 3 | 3 / 6 | ~0.58 |

---

## Hardware

- 2x Raspberry Pi 4 Model B
- 2x RFM9x LoRa Module (433 MHz)
- 2x INA219 Power Sensor
- 2x Breadboard + jumper wires

---

## Setup

Install dependencies on both Pis:

```bash
pip3 install torch==2.6.0 torchvision==0.21.0 --break-system-packages
pip3 install adafruit-circuitpython-rfm9x adafruit-circuitpython-ina219 --break-system-packages
pip3 install scikit-image pillow numpy --break-system-packages
```

Enable SPI on both Pis via `raspi-config` ? Interface Options ? SPI.

---

## Running

**On Pi A (Sender):**
```bash
python3 New_LoRa_Sender.py
```

**On Pi B (Receiver):**
```bash
python3 New_LoRa_Receiver.py
```

Start the receiver before the sender. Results are printed to the console and optionally saved to a log file.

---





---

*Dissertation submitted in partial fulfilment of the requirements for BSc Computer Science, York St John University, 2026.*
