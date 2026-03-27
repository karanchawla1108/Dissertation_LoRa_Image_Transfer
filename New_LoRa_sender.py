# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
import busio
import board
import adafruit_ina219
import adafruit_rfm9x
import digitalio
from PIL import Image

# -----------------------------
# MATCH RECEIVER MODEL SETTINGS
# -----------------------------
LATENT_DIM = 64

class ImprovedVAE(nn.Module):
    def __init__(self):
        super(ImprovedVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(128, LATENT_DIM)
        self.var_layer  = nn.Linear(128, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_layer(x), self.var_layer(x)

    def reparameterise(self, mean, var):
        return mean + var * torch.randn_like(var)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterise(mean, var)
        return self.decode(z), mean, var


print("Loading improved VAE model...")
model = ImprovedVAE()
model.load_state_dict(torch.load(
    '/home/karan/Desktop/New Improved MNIST VAE model/vae_model_improved.pth',
    map_location='cpu'
))
model.eval()
print("Model loaded OK")





# -----------------------------
# INA219 POWER SENSOR
# -----------------------------
i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)

print("INA219 power sensor ready")
print("--- IDLE READINGS ---")
print("Voltage: " + str(round(ina.bus_voltage, 2)) + "V")
print("Current: " + str(round(ina.current, 2)) + "mA")
print("Power:   " + str(round(ina.power, 2)) + "mW")

# -----------------------------
# LORA SETUP
# -----------------------------
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
rst = digitalio.DigitalInOut(board.D25)
rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)

rfm.tx_power = 23
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True

print("LoRa ready")

# -----------------------------
# FUNCTIONS
# -----------------------------
def prepare_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor

def encode_image(tensor):
    with torch.no_grad():
        t_start = time.time()
        p_before = ina.power
        v_before = ina.bus_voltage
        i_before = ina.current

        mean, var = model.encode(tensor)
        z = model.reparameterise(mean, var)

        p_after = ina.power
        t_end = time.time()

    encode_time = (t_end - t_start) * 1000
    encode_power = (p_before + p_after) / 2

    print("--- ENCODING ---")
    print("Voltage: " + str(round(v_before, 2)) + "V")
    print("Current: " + str(round(i_before, 2)) + "mA")
    print("Power:   " + str(round(encode_power, 2)) + "mW")
    print("Time:    " + str(round(encode_time, 1)) + "ms")

    return z.numpy().flatten(), encode_time, encode_power

def split_packets(latent_vector):
    payload = latent_vector.astype(np.float32).tobytes()

    print("Payload bytes: " + str(len(payload)))  # should be 256 bytes

    packets = []
    for i in range(0, len(payload), 48):
        packets.append(payload[i:i+48])

    print("Split into " + str(len(packets)) + " packets")  # should be 6
    return packets

def send_packets(packets):
    print("--- TRANSMISSION ---")
    t_start = time.time()
    p_before = ina.power
    v_before = ina.bus_voltage
    i_before = ina.current

    for i, packet in enumerate(packets):
        header = bytes([i, len(packets)])
        rfm.send(header + packet)
        print("Sent packet " + str(i+1) + "/" + str(len(packets)))
        time.sleep(0.2)

    p_after = ina.power
    t_end = time.time()

    tx_time = (t_end - t_start) * 1000
    tx_power = (p_before + p_after) / 2

    print("Voltage: " + str(round(v_before, 2)) + "V")
    print("Current: " + str(round(i_before, 2)) + "mA")
    print("Power:   " + str(round(tx_power, 2)) + "mW")
    print("Time:    " + str(round(tx_time, 1)) + "ms")

    return tx_time, tx_power
    
    
    
    
    

# -----------------------------
# START
# -----------------------------
print("Waiting 20s for receiver to be ready...")
time.sleep(20)

print("Starting improved VAE transmission...")

image_path = '/home/karan/Desktop/Image_Disseratation/test_image.png'
tensor = prepare_image(image_path)

latent, enc_time, enc_power = encode_image(tensor)
packets = split_packets(latent)
tx_time, tx_power = send_packets(packets)

img_check = Image.open(image_path)
print("Image size: " + str(img_check.size))
print("Sending MNIST digit image...")

print("--- RESULTS SUMMARY ---")
print("Packets sent:  " + str(len(packets)))
print("Payload size:  " + str(len(latent.astype(np.float32).tobytes())) + " bytes")
print("Encode time:   " + str(round(enc_time, 1)) + "ms")
print("TX time:       " + str(round(tx_time, 1)) + "ms")
print("Total time:    " + str(round(enc_time + tx_time, 1)) + "ms")
print("Encode power:  " + str(round(enc_power, 2)) + "mW")
print("TX power:      " + str(round(tx_power, 2)) + "mW")
print("Done!")
