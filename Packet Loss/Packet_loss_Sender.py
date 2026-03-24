# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
import busio
import board
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
    '/home/karan/Desktop/AutoEncoders/MNIST/vae_model (3).pth',
    map_location='cpu'
))
model.eval()
print("Model loaded OK")
 
import adafruit_ina219
i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)
print("INA219 ready!")
print("Idle voltage: " + str(round(ina.bus_voltage, 2)) + "V")
print("Idle current: " + str(round(ina.current, 2)) + "mA")
print("Idle power:   " + str(round(ina.power, 2)) + "mW")
 
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
        mean, var = model.encode(tensor)
        z = model.reparameterise(mean, var)
        t_end = time.time()
    encode_time = (t_end - t_start) * 1000
    print("Encoded in " + str(round(encode_time, 1)) + "ms")
    return z.numpy().flatten(), encode_time
 
def split_packets(latent_vector):
    payload = latent_vector.astype(np.float32).tobytes()
    packets = []
    for i in range(0, len(payload), 48):
        packets.append(payload[i:i+48])
    return packets
 
def send_packets(packets, drop_list=[]):
    print("Sending " + str(len(packets)) + " packets - dropping: " + str(drop_list))
    t_start = time.time()
    sent_count = 0
    dropped_count = 0
    for i, packet in enumerate(packets):
        if (i+1) in drop_list:
            print("  Packet " + str(i+1) + "/" + str(len(packets)) + " DROPPED (simulated loss)")
            dropped_count += 1
        else:
            header = bytes([i, len(packets)])
            rfm.send(header + packet)
            print("  Sent packet " + str(i+1) + "/" + str(len(packets)))
            sent_count += 1
        time.sleep(0.2)
    t_end = time.time()
    tx_time = (t_end - t_start) * 1000
    print("Sent " + str(sent_count) + "/" + str(len(packets)) + " packets")
    print("Dropped " + str(dropped_count) + "/" + str(len(packets)) + " packets")
    print("TX time: " + str(round(tx_time, 1)) + "ms")
    return tx_time, sent_count, dropped_count
 
def run_test(test_num, drop_list, image_path, label=""):
    print("")
    print("==========================================")
    print("TEST " + str(test_num) + " - " + label)
    print("==========================================")
    tensor = prepare_image(image_path)
    latent, enc_time = encode_image(tensor)
    packets = split_packets(latent)
    tx_time, sent, dropped = send_packets(packets, drop_list=drop_list)
    print("Test " + str(test_num) + " done. Waiting 25 seconds for receiver...")
    time.sleep(25)
    return enc_time, tx_time, sent, dropped
 
image_path = '/home/karan/test_image.png'
print("")
print("Starting 3 packet loss tests...")
print("Make sure receiver is running on Pi B!")
print("Starting in 10 seconds...")
time.sleep(10)
 
results = []
tests = [
    (1, [],     "0%  loss - send all 3 packets"),
    (2, [2],    "33% loss - drop packet 2"),
    (3, [1, 2], "66% loss - drop packets 1 and 2"),
]


for test_num, drop_list, label in tests:
    enc_time, tx_time, sent, dropped = run_test(test_num, drop_list, image_path, label)
    results.append({
        'test': test_num,
        'label': label,
        'drop_list': drop_list,
        'sent': sent,
        'dropped': dropped,
        'enc_time': enc_time,
        'tx_time': tx_time
    })
 
print("")
print("==========================================")
print("ALL TESTS COMPLETE - SUMMARY")
print("==========================================")
for r in results:
    print("Test " + str(r['test']) + " | " + r['label'] + " | Sent: " + str(r['sent']) + "/3 | TX: " + str(round(r['tx_time'], 1)) + "ms")
print("Check Pi B for SSIM scores!")
