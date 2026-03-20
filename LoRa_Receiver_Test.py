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
from skimage.metrics import structural_similarity as ssim
import os
 
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
    '/home/ysj/Desktop/AutoEncoders/MNIST/vae_model (3).pth', // Autoencoder Model Path
    map_location='cpu'
))
model.eval()
print("Model loaded OK")


i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)
print("INA219 power sensor ready")
print("Voltage: " + str(round(ina.bus_voltage, 2)) + "V")
print("Current: " + str(round(ina.current, 2)) + "mA")
print("Power:   " + str(round(ina.power, 2)) + "mW")
 
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
rst = digitalio.DigitalInOut(board.D25)
rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True
print("LoRa ready - waiting for packets...")
 
BASE_DIR = '/home/ysj/Result for the image'
os.makedirs(BASE_DIR, exist_ok=True)
 
def get_next_folder():
    existing = [f for f in os.listdir(BASE_DIR) if f.startswith('run') and os.path.isdir(BASE_DIR + '/' + f)]
    if not existing:
        return BASE_DIR + '/run1'
    numbers = []
    for f in existing:
        try:
            num = int(f.replace('run', ''))
            numbers.append(num)
        except:
            pass
    next_num = max(numbers) + 1 if numbers else 1
    return BASE_DIR + '/run' + str(next_num)
 
def receive_packets(num_packets=3, timeout=30):
    received = {}
    t_start = time.time()
    p_before = ina.power
    while len(received) < num_packets:
        if time.time() - t_start > timeout:
            print("Timeout - moving on")
            break
        packet = rfm.receive(timeout=5.0)
        if packet is not None:
            pkt_num = packet[0]
            total   = packet[1]
            data    = bytes(packet[2:])
            received[pkt_num] = data
            print("Received packet " + str(pkt_num+1) + "/" + str(total))
    p_after = ina.power
    rx_time = (time.time() - t_start) * 1000
    rx_power = (p_before + p_after) / 2
    all_packets = []
    for i in range(num_packets):
        if i in received:
            all_packets.append(received[i])
        else:
            print("Packet " + str(i+1) + " LOST - filling with zeros")
            size = 48 if i < num_packets - 1 else 32
            all_packets.append(bytes(size))
    packets_received = len(received)
    print("Received " + str(packets_received) + "/" + str(num_packets) + " packets")
    print("RX time:  " + str(round(rx_time, 1)) + "ms")
    print("RX power: " + str(round(rx_power, 2)) + "mW")
    return all_packets, packets_received, rx_time, rx_power
 
def decode_image(packets):
    payload = b''.join(packets)
    latent = np.frombuffer(payload, dtype=np.float32).copy()
    z = torch.FloatTensor(latent).unsqueeze(0)
    t_start = time.time()
    p_before = ina.power
    with torch.no_grad():
        reconstructed = model.decode(z)
    p_after = ina.power
    t_end = time.time()
    decode_time = (t_end - t_start) * 1000
    decode_power = (p_before + p_after) / 2
    img_array = reconstructed.numpy().reshape(28, 28)
    print("Decoded in " + str(round(decode_time, 1)) + "ms")
    print("Decode power: " + str(round(decode_power, 2)) + "mW")
    return img_array, decode_time, decode_power
 
def save_and_score(img_array, run_folder):
    output = (img_array * 255).astype(np.uint8)
 
    normal_path = run_folder + '/result.png'
    Image.fromarray(output).save(normal_path)
    print("Saved: " + normal_path)
 
    big_path = run_folder + '/result_big.png'
    Image.fromarray(output).resize((280, 280), Image.NEAREST).save(big_path)
    print("Saved big: " + big_path)
    
    
    try:
        original = np.array(
            Image.open('/home/ysj/test_image.png').convert('L').resize((28, 28))
        ) / 255.0
        score = ssim(original, img_array, data_range=1.0)
        print("SSIM Score: " + str(round(score, 4)))
 
        orig_big = Image.fromarray((original * 255).astype(np.uint8)).resize((280, 280), Image.NEAREST)
        recon_big = Image.fromarray(output).resize((280, 280), Image.NEAREST)
        comparison = Image.new('L', (580, 300), color=128)
        comparison.paste(orig_big, (10, 10))
        comparison.paste(recon_big, (300, 10))
        comp_path = run_folder + '/comparison.png'
        comparison.save(comp_path)
        print("Saved comparison: " + comp_path)
        return score
    except Exception as e:
        print("Could not calculate SSIM: " + str(e))
        return None
 
def save_log(run_folder, packets_received, rx_time, rx_power, decode_time, decode_power, total_time, ssim_score):
    log_path = run_folder + '/results.txt'
    with open(log_path, 'w') as f:
        f.write("VAE LoRa Transmission Results\n")
        f.write("==============================\n")
        f.write("Packets received:  " + str(packets_received) + "/3\n")
        f.write("RX time:           " + str(round(rx_time, 1)) + "ms\n")
        f.write("Decode time:       " + str(round(decode_time, 1)) + "ms\n")
        f.write("Total latency:     " + str(round(total_time, 1)) + "ms\n")
        f.write("RX power:          " + str(round(rx_power, 2)) + "mW\n")
        f.write("Decode power:      " + str(round(decode_power, 2)) + "mW\n")
        if ssim_score:
            f.write("SSIM Score:        " + str(round(ssim_score, 4)) + "\n")
    print("Saved log: " + log_path)
 
print("Waiting for transmission...")
t_total_start = time.time()
 
run_folder = get_next_folder()
os.makedirs(run_folder)
print("Results folder: " + run_folder)
 
packets, packets_received, rx_time, rx_power = receive_packets()
img_array, decode_time, decode_power = decode_image(packets)
ssim_score = save_and_score(img_array, run_folder)
 
total_time = (time.time() - t_total_start) * 1000
save_log(run_folder, packets_received, rx_time, rx_power, decode_time, decode_power, total_time, ssim_score)
 
print("--- RESULTS ---")
print("Run folder:        " + run_folder)
print("Packets received:  " + str(packets_received) + "/3")
print("RX time:           " + str(round(rx_time, 1)) + "ms")
print("Decode time:       " + str(round(decode_time, 1)) + "ms")
print("Total latency:     " + str(round(total_time, 1)) + "ms")
print("RX power:          " + str(round(rx_power, 2)) + "mW")
print("Decode power:      " + str(round(decode_power, 2)) + "mW")
if ssim_score:
    print("SSIM Score:        " + str(round(ssim_score, 4)))
print("Done!")
