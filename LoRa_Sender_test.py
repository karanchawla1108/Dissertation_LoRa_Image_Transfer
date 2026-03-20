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

class SimpleVAE(nn.Module):
    def __init__(self):
        super(SimpleVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(64, 32)
        self.var_layer  = nn.Linear(64, 32)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
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

print("Loading VAE model...")
model = SimpleVAE()
model.load_state_dict(torch.load(
    '/home/karan/Desktop/AutoEncoders/MNIST/vae_model (3).pth', // Model Path
    map_location='cpu'
))
model.eval()
print("Model loaded OK")

i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)

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

def prepare_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor

def encode_image(tensor):
    with torch.no_grad():
        t_start = time.time()
        p_before = ina.power
        mean, var = model.encode(tensor)
        z = model.reparameterise(mean, var)
        p_after = ina.power
        t_end = time.time()
    encode_time = (t_end - t_start) * 1000
    encode_power = (p_before + p_after) / 2
    print("Encoded in " + str(round(encode_time, 1)) + "ms")
    print("Encode power: " + str(round(encode_power, 2)) + "mW")
    return z.numpy().flatten(), encode_time, encode_power

def split_packets(latent_vector):
    payload = latent_vector.astype(np.float32).tobytes()
    packets = []
    for i in range(0, len(payload), 48):
        packets.append(payload[i:i+48])
    print("Split into " + str(len(packets)) + " packets")
    return packets

def send_packets(packets):
    print("Sending " + str(len(packets)) + " packets...")
    t_start = time.time()
    p_before = ina.power
    for i, packet in enumerate(packets):
        header = bytes([i, len(packets)])
        rfm.send(header + packet)
        print("Sent packet " + str(i+1) + "/" + str(len(packets)))
        time.sleep(0.2)
    p_after = ina.power
    t_end = time.time()
    tx_time = (t_end - t_start) * 1000
    tx_power = (p_before + p_after) / 2
    print("Sent in " + str(round(tx_time, 1)) + "ms")
    print("TX power: " + str(round(tx_power, 2)) + "mW")
    return tx_time, tx_power

print("Starting VAE transmission...")
image_path = '/home/karan/Desktop/Image_Disseratation/test_image.png' / Image path
tensor = prepare_image(image_path)
latent, enc_time, enc_power = encode_image(tensor)
packets = split_packets(latent)
tx_time, tx_power = send_packets(packets)

# Show what digit we are sending
from PIL import Image as PILImage
img_check = PILImage.open(image_path)
print("Image size: " + str(img_check.size))
print("Sending MNIST digit image...")



print("--- RESULTS ---")
print("Packets sent:  " + str(len(packets)))
print("Payload size:  128 bytes")
print("Encode time:   " + str(round(enc_time, 1)) + "ms")
print("TX time:       " + str(round(tx_time, 1)) + "ms")
print("Total time:    " + str(round(enc_time + tx_time, 1)) + "ms")
print("Encode power:  " + str(round(enc_power, 2)) + "mW")
print("TX power:      " + str(round(tx_power, 2)) + "mW")
print("Done!")
