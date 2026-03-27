# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
import os
import busio
import board
import adafruit_rfm9x
import digitalio
import adafruit_ina219
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# MODEL
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


# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = '/home/ysj/Desktop/New Improved MNIST VAE model/vae_model_improved.pth'
ORIGINAL_IMAGE_PATH = '/home/ysj/Image_dissertation/test_image.png'
BASE_FOLDER = '/home/ysj/Packet loss Image Improved'

print("Checking files...")
print("Model exists:", os.path.exists(MODEL_PATH))
print("Original image exists:", os.path.exists(ORIGINAL_IMAGE_PATH))

print("Loading improved VAE model...")
model = ImprovedVAE()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("Model loaded OK")

# -----------------------------
# INA219
# -----------------------------
i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)
print("INA219 ready!")
print("Idle voltage: " + str(round(ina.bus_voltage, 2)) + "V")
print("Idle current: " + str(round(ina.current, 2)) + "mA")
print("Idle power:   " + str(round(ina.power, 2)) + "mW")

# -----------------------------
# LORA
# -----------------------------
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
rst = digitalio.DigitalInOut(board.D25)
rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True
print("LoRa ready - waiting for packets...")




# -----------------------------
# RUN FOLDER
# -----------------------------
os.makedirs(BASE_FOLDER, exist_ok=True)

run_num = 1
while os.path.exists(BASE_FOLDER + '/run' + str(run_num)):
    run_num += 1

run_folder = BASE_FOLDER + '/run' + str(run_num)
os.makedirs(run_folder)
print("Saving images to: " + run_folder)

# -----------------------------
# RECEIVE
# -----------------------------
def receive_packets(expected_test_num, num_packets=6, first_packet_timeout=60, inter_packet_timeout=3):
    received = {}
    first_packet_received = False
    t_start = time.time()
    p_before = ina.power
    last_packet_time = None

    print("Waiting for packets for TEST " + str(expected_test_num) + "...")

    while True:
        if not first_packet_received:
            if time.time() - t_start > first_packet_timeout:
                print("Timeout waiting for first packet!")
                break
            packet = rfm.receive(timeout=1.0)
        else:
            if time.time() - last_packet_time > inter_packet_timeout:
                print("No new packets - finishing this test")
                break
            packet = rfm.receive(timeout=0.5)

        if packet is None:
            continue

        if len(packet) < 3:
            print("  Ignored short packet")
            continue

        rx_test_num = packet[0]
        pkt_num = packet[1]
        total = packet[2]
        data = bytes(packet[3:])

        if rx_test_num != expected_test_num:
            print("  Ignored packet from test " + str(rx_test_num))
            continue

        if not first_packet_received:
            first_packet_received = True
            last_packet_time = time.time()
            print("First packet received for TEST " + str(expected_test_num))

        if pkt_num not in received:
            received[pkt_num] = data
            last_packet_time = time.time()
            print("  Received packet " + str(pkt_num + 1) + "/" + str(total))
        else:
            print("  Duplicate packet " + str(pkt_num + 1) + " ignored")

        if len(received) == num_packets:
            print("All packets received for this test")
            break

    p_after = ina.power
    rx_time = (time.time() - t_start) * 1000
    rx_power = (p_before + p_after) / 2

    all_packets = []
    lost = []

    for i in range(num_packets):
        if i in received:
            all_packets.append(received[i])
        else:
            lost.append(i + 1)
            print("  Packet " + str(i + 1) + " LOST - filling with zeros")
            size = 48 if i < num_packets - 1 else 16
            all_packets.append(bytes(size))

    packets_received = len(received)

    print("Received " + str(packets_received) + "/" + str(num_packets) + " packets")
    if lost:
        print("Lost packets: " + str(lost))
    print("RX time: " + str(round(rx_time, 1)) + "ms")
    print("RX power: " + str(round(rx_power, 2)) + "mW")

    return all_packets, packets_received, rx_time, rx_power, lost
    
    
    
# -----------------------------
# DECODE
# -----------------------------
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

# -----------------------------
# SAVE + SCORE
# -----------------------------
def save_and_score(img_array, test_num, loss_label, packets_received, lost):
    output = (img_array * 255).astype(np.uint8)

    recon_filename = run_folder + '/test' + str(test_num) + '_' + loss_label + '_reconstructed.png'
    Image.fromarray(output).save(recon_filename)
    print("Saved: " + recon_filename)

    try:
        original_pil = Image.open(ORIGINAL_IMAGE_PATH).convert('L').resize((28, 28))
        original_arr = np.array(original_pil, dtype=np.float32) / 255.0
        score = ssim(original_arr, img_array, data_range=1.0)
        print("SSIM Score: " + str(round(score, 4)))
    except Exception as e:
        print("SSIM error: " + str(e))
        score = None
        original_pil = Image.new('L', (28, 28), 0)

    scale = 10
    size = 28 * scale
    padding = 10
    label_height = 30
    total_width = (size * 2) + (padding * 3)
    total_height = size + (padding * 2) + (label_height * 2)

    comparison = Image.new('RGB', (total_width, total_height), color=(40, 40, 40))

    orig_big = original_pil.resize((size, size), Image.NEAREST).convert('RGB')
    comparison.paste(orig_big, (padding, label_height + padding))

    recon_big = Image.fromarray(output).resize((size, size), Image.NEAREST).convert('RGB')
    comparison.paste(recon_big, (size + padding * 2, label_height + padding))

    draw = ImageDraw.Draw(comparison)

    title = "Test " + str(test_num) + " | " + loss_label + " | Packets: " + str(packets_received) + "/6"
    if score is not None:
        title += " | SSIM: " + str(round(score, 4))
    draw.text((padding, 5), title, fill=(255, 255, 100))

    draw.text((padding, label_height + size + padding + 5), "ORIGINAL", fill=(100, 255, 100))
    draw.text((size + padding * 2, label_height + size + padding + 5), "RECONSTRUCTED", fill=(100, 200, 255))

    if lost:
        draw.text((size + padding * 2, 5), "Lost packets: " + str(lost), fill=(255, 100, 100))

    comp_filename = run_folder + '/test' + str(test_num) + '_' + loss_label + '_comparison.png'
    comparison.save(comp_filename)
    print("Comparison saved: " + comp_filename)

    return score
    
    
    
    
# -----------------------------
# SAVE SUMMARY
# -----------------------------
def save_summary(all_results):
    summary_file = run_folder + '/final_results.txt'
    with open(summary_file, 'w') as f:
        f.write("Improved VAE Packet Loss Test Results\n")
        f.write("=====================================\n\n")

        for r in all_results:
            f.write("Test " + str(r['test']) + "\n")
            f.write("Label:           " + r['label'] + "\n")
            f.write("Packets RX:      " + str(r['received']) + "/6\n")
            f.write("Lost packets:    " + (str(r['lost']) if r['lost'] else "none") + "\n")
            f.write("RX time:         " + str(round(r['rx_time'], 1)) + "ms\n")
            f.write("RX power:        " + str(round(r['rx_power'], 2)) + "mW\n")
            f.write("Decode time:     " + str(round(r['decode_time'], 1)) + "ms\n")
            f.write("Decode power:    " + str(round(r['decode_power'], 2)) + "mW\n")
            if r['ssim'] is not None:
                f.write("SSIM:            " + str(round(r['ssim'], 4)) + "\n")
            else:
                f.write("SSIM:            N/A\n")
            f.write("\n")

    print("Saved summary: " + summary_file)

# -----------------------------
# TEST LABELS
# -----------------------------
loss_labels = [
    '0pct_loss',
    '1packet_loss',
    '2packet_loss',
    '3packet_loss'
]

all_results = []

# -----------------------------
# MAIN LOOP
# -----------------------------
for test_num in range(1, 5):
    print("")
    print("==========================================")
    print("Waiting for TEST " + str(test_num) + " (" + loss_labels[test_num - 1] + ")...")
    print("==========================================")

    packets, packets_received, rx_time, rx_power, lost = receive_packets(
        expected_test_num=test_num,
        num_packets=6,
        first_packet_timeout=60,
        inter_packet_timeout=3
    )

    img_array, decode_time, decode_power = decode_image(packets)
    ssim_score = save_and_score(img_array, test_num, loss_labels[test_num - 1], packets_received, lost)

    all_results.append({
        'test': test_num,
        'label': loss_labels[test_num - 1],
        'received': packets_received,
        'lost': lost,
        'rx_time': rx_time,
        'rx_power': rx_power,
        'decode_time': decode_time,
        'decode_power': decode_power,
        'ssim': ssim_score
    })

    print("Test " + str(test_num) + " complete.")

# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("")
print("==========================================")
print("ALL TESTS COMPLETE - FINAL RESULTS")
print("==========================================")
print("")
print("Test | Label          | Packets RX | Lost       | Decode   | SSIM")
print("-----|----------------|------------|------------|----------|------")

for r in all_results:
    lost_str = str(r['lost']) if r['lost'] else "none"
    ssim_str = str(round(r['ssim'], 4)) if r['ssim'] is not None else "N/A"
    print(
        str(r['test']) + "    | " +
        r['label'].ljust(14) + " | " +
        str(r['received']) + "/6        | " +
        lost_str.ljust(10) + " | " +
        str(round(r['decode_time'], 1)).ljust(7) + "ms | " +
        ssim_str
    )

save_summary(all_results)

print("")
print("All images saved in: " + run_folder)
print("Done!")
