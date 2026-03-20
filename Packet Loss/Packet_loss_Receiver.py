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
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
 
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
    '/home/ysj/Desktop/AutoEncoders/MNIST/vae_model (3).pth', // Trained Model path
    map_location='cpu'
))
model.eval()
print("Model loaded OK")


class FakeINA:
    power = 0.0
ina = FakeINA()
 
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
rst = digitalio.DigitalInOut(board.D25)
rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True
print("LoRa ready - waiting for packets...")
 
# Create run folder automatically
base_folder = '/home/ysj/Packet loss Image' // Image save folder path
os.makedirs(base_folder, exist_ok=True)
 
run_num = 1
while os.path.exists(base_folder + '/run' + str(run_num)):
    run_num += 1
run_folder = base_folder + '/run' + str(run_num)
os.makedirs(run_folder)
print("Saving images to: " + run_folder)
 
def receive_packets(num_packets=3, timeout=30):
    received = {}
    t_start = time.time()
    while len(received) < num_packets:
        if time.time() - t_start > timeout:
            print("Timeout!")
            break
        packet = rfm.receive(timeout=5.0)
        if packet is not None:
            pkt_num = packet[0]
            total   = packet[1]
            data    = bytes(packet[2:])
            received[pkt_num] = data
            print("  Received packet " + str(pkt_num+1) + "/" + str(total))
    rx_time = (time.time() - t_start) * 1000
    all_packets = []
    lost = []
    for i in range(num_packets):
        if i in received:
            all_packets.append(received[i])
        else:
            lost.append(i+1)
            print("  Packet " + str(i+1) + " LOST - filling with zeros")
            size = 48 if i < num_packets - 1 else 32
            all_packets.append(bytes(size))
    packets_received = len(received)
    print("Received " + str(packets_received) + "/" + str(num_packets) + " packets")
    if lost:
        print("Lost packets: " + str(lost))
    print("RX time: " + str(round(rx_time, 1)) + "ms")
    return all_packets, packets_received, rx_time, lost
 
def decode_image(packets):
    payload = b''.join(packets)
    latent = np.frombuffer(payload, dtype=np.float32).copy()
    z = torch.FloatTensor(latent).unsqueeze(0)
    t_start = time.time()
    with torch.no_grad():
        reconstructed = model.decode(z)
    t_end = time.time()
    decode_time = (t_end - t_start) * 1000
    img_array = reconstructed.numpy().reshape(28, 28)
    print("Decoded in " + str(round(decode_time, 1)) + "ms")
    return img_array, decode_time
 
def save_and_score(img_array, test_num, loss_label, packets_received, lost, ssim_val=None):
    output = (img_array * 255).astype(np.uint8)
 
    # Save reconstructed image
    recon_filename = run_folder + '/test' + str(test_num) + '_' + loss_label + '_reconstructed.png'
    Image.fromarray(output).save(recon_filename)
    print("Saved: " + recon_filename)




    # Load original
    try:
        original_pil = Image.open('/home/ysj/test_image.png').convert('L').resize((28, 28))
        original_arr = np.array(original_pil) / 255.0
        score = ssim(original_arr, img_array, data_range=1.0)
        print("SSIM Score: " + str(round(score, 4)))
    except Exception as e:
        print("SSIM error: " + str(e))
        score = None
        original_pil = Image.new('L', (28, 28), 0)
 
    # Make comparison image
    scale = 10
    size = 28 * scale
    padding = 10
    label_height = 30
    total_width = (size * 2) + (padding * 3)
    total_height = size + (padding * 2) + (label_height * 2)
 
    comparison = Image.new('RGB', (total_width, total_height), color=(40, 40, 40))
 
    # Paste original
    orig_big = original_pil.resize((size, size), Image.NEAREST).convert('RGB')
    comparison.paste(orig_big, (padding, label_height + padding))
 
    # Paste reconstructed
    recon_big = Image.fromarray(output).resize((size, size), Image.NEAREST).convert('RGB')
    comparison.paste(recon_big, (size + padding * 2, label_height + padding))
 
    # Add labels
    draw = ImageDraw.Draw(comparison)
 
    # Title
    title = "Test " + str(test_num) + " | " + loss_label + " | Packets: " + str(packets_received) + "/3"
    if score is not None:
        title = title + " | SSIM: " + str(round(score, 4))
    draw.text((padding, 5), title, fill=(255, 255, 100))
 
    # Image labels
    draw.text((padding, label_height + size + padding + 5), "ORIGINAL", fill=(100, 255, 100))
    draw.text((size + padding * 2, label_height + size + padding + 5), "RECONSTRUCTED", fill=(100, 200, 255))
 
    # Lost packet info
    if lost:
        lost_text = "Lost packets: " + str(lost)
        draw.text((size + padding * 2, 5), lost_text, fill=(255, 100, 100))
 
    # Save comparison
    comp_filename = run_folder + '/test' + str(test_num) + '_' + loss_label + '_comparison.png'
    comparison.save(comp_filename)
    print("Comparison saved: " + comp_filename)
 
    return score
 
loss_labels = ['0pct_loss', '33pct_loss', '66pct_loss']
all_results = []
 
for test_num in range(1, 4):
    print("")
    print("==========================================")
    print("Waiting for TEST " + str(test_num) + " (" + loss_labels[test_num-1] + ")...")
    print("==========================================")
 
    packets, packets_received, rx_time, lost = receive_packets()
    img_array, decode_time = decode_image(packets)
    ssim_score = save_and_score(img_array, test_num, loss_labels[test_num-1], packets_received, lost)
 
    all_results.append({
        'test': test_num,
        'label': loss_labels[test_num-1],
        'received': packets_received,
        'lost': lost,
        'rx_time': rx_time,
        'decode_time': decode_time,
        'ssim': ssim_score
    })
 
    print("Test " + str(test_num) + " complete.")
    if test_num < 3:
        print("Waiting for next test...")
 
print("")
print("==========================================")
print("ALL TESTS COMPLETE - FINAL RESULTS")
print("==========================================")
print("")
print("Test | Label         | Packets RX | Lost  | Decode  | SSIM")
print("-----|---------------|------------|-------|---------|------")
for r in all_results:
    lost_str = str(r['lost']) if r['lost'] else "none"
    ssim_str = str(round(r['ssim'], 4)) if r['ssim'] else "N/A"
    print(str(r['test']) + "    | " + r['label'] + "  | " + str(r['received']) + "/3        | " + lost_str + " | " + str(round(r['decode_time'], 1)) + "ms   | " + ssim_str)
 
print("")
print("All images saved in: " + run_folder)
print("  test1_0pct_loss_reconstructed.png + comparison.png")
print("  test2_33pct_loss_reconstructed.png + comparison.png")
print("  test3_66pct_loss_reconstructed.png + comparison.png")
print("Done!")
