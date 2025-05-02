from flask import Flask, render_template, request, session
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import time
from scipy.fft import dct, idct  # Add scipy for DCT

app = Flask(__name__)
app.secret_key = 'mysecret123'
UPLOAD_FOLDER = '/Users/olaf/Desktop/ME_CS/CTF/Steganography/webapp/uploads'
OUTPUT_FOLDER = '/Users/olaf/Desktop/ME_CS/CTF/Steganography/webapp/static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# AI Encoder/Decoder (unchanged)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.msg_layer = nn.Linear(128, 64*64)
    def forward(self, image, message):
        msg = self.msg_layer(message).view(-1, 1, 64, 64)
        stego_image = image + (msg * 0.01)
        return torch.clamp(stego_image, 0, 1)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        self.msg_out = nn.Linear(64*64, 128)
    def forward(self, stego_image):
        x = self.conv(stego_image)
        x = x.view(-1, 64*64)
        return self.msg_out(x)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((64, 64))
    return transforms.ToTensor()(img).unsqueeze(0)

def preprocess_message(msg):
    binary = ''.join(format(ord(c), '08b') for c in msg)[:128]
    binary = (binary + '0' * 128)[:128]
    return torch.tensor([int(b) for b in binary], dtype=torch.float32).unsqueeze(0)

# LSB Encoding/Decoding (unchanged)
def lsb_encode(cover_path, message, output_path):
    img = Image.open(cover_path).convert('RGB')
    pixels = np.array(img, dtype=np.uint8)
    msg_bits = ''.join(format(ord(c), '08b') for c in message) + '00000000'
    msg_idx = 0
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            for k in range(3):
                if msg_idx < len(msg_bits):
                    pixels[i, j, k] = (pixels[i, j, k] & 0xFE) | int(msg_bits[msg_idx])
                    msg_idx += 1
                if msg_idx >= len(msg_bits):
                    Image.fromarray(pixels).save(output_path, format='PNG')
                    return
    Image.fromarray(pixels).save(output_path, format='PNG')

def lsb_decode(stego_path):
    img = Image.open(stego_path).convert('RGB')
    pixels = np.array(img)
    bits = ''
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            for k in range(3):
                bits += str(pixels[i, j, k] & 1)
                if len(bits) >= 8 and len(bits) % 8 == 0:
                    byte = bits[-8:]
                    if byte == '00000000':
                        msg = ''
                        for b in range(0, len(bits) - 8, 8):
                            msg += chr(int(bits[b:b+8], 2))
                        return msg
    msg = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            char = chr(int(byte, 2))
            if char == '\0':
                break
            msg += char
    return msg

# DCT Encoding
def dct_encode(cover_path, message, output_path):
    img = Image.open(cover_path).convert('L')  # Grayscale for simplicity
    pixels = np.array(img, dtype=np.float32)
    # Ensure 8x8 blocks fit (pad if needed)
    h, w = pixels.shape
    h = (h + 7) // 8 * 8
    w = (w + 7) // 8 * 8
    pixels = np.pad(pixels, ((0, h - pixels.shape[0]), (0, w - pixels.shape[1])), mode='constant')
    
    msg_bits = ''.join(format(ord(c), '08b') for c in message) + '00000000'
    msg_idx = 0
    
    # Process 8x8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = pixels[i:i+8, j:j+8]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            if msg_idx < len(msg_bits):
                # Embed in a mid-frequency coefficient (e.g., [1,1])
                dct_block[1, 1] = (int(dct_block[1, 1]) & ~1) | int(msg_bits[msg_idx])
                msg_idx += 1
            pixels[i:i+8, j:j+8] = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            if msg_idx >= len(msg_bits):
                break
        if msg_idx >= len(msg_bits):
            break
    
    Image.fromarray(pixels.astype(np.uint8)).save(output_path, format='PNG')

# DCT Decoding
def dct_decode(stego_path):
    img = Image.open(stego_path).convert('L')
    pixels = np.array(img, dtype=np.float32)
    h, w = pixels.shape
    bits = ''
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = pixels[i:i+8, j:j+8]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            bits += str(int(dct_block[1, 1]) & 1)
            if len(bits) >= 8 and len(bits) % 8 == 0:
                byte = bits[-8:]
                if byte == '00000000':
                    msg = ''
                    for b in range(0, len(bits) - 8, 8):
                        msg += chr(int(bits[b:b+8], 2))
                    return msg
    
    msg = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            char = chr(int(byte, 2))
            if char == '\0':
                break
            msg += char
    return msg

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encoder', methods=['GET', 'POST'])
def encoder():
    if request.method == 'POST':
        cover_file = request.files['cover_image']
        message = request.form['message']
        method = request.form['method']
        
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cover.png')
        output_filename = f"stego_{int(time.time())}.png"  # Back to timestamp
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        cover_file.save(cover_path)
        
        if method == 'ai':
            cover_img = preprocess_image(cover_path)
            msg_tensor = preprocess_message(message)
            encoder = Encoder()
            stego_img = encoder(cover_img, msg_tensor)
            transforms.ToPILImage()(stego_img.squeeze(0)).save(output_path)
        elif method == 'lsb':
            lsb_encode(cover_path, message, output_path)
        elif method == 'dct':
            dct_encode(cover_path, message, output_path)
        
        return render_template('encoder_output.html', output_image=f'output/{output_filename}')
    return render_template('encoder.html')

@app.route('/decoder', methods=['GET', 'POST'])
def decoder():
    if request.method == 'POST':
        method = request.form['method']
        stego_path = session.get('stego_path')
        
        stego_file = request.files.get('stego_image')
        if stego_file and stego_file.filename:
            stego_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego.png')
            stego_file.save(stego_path)
            session['stego_path'] = stego_path
            session['last_filename'] = stego_file.filename
        elif not stego_path or not os.path.exists(stego_path):
            return render_template('decoder.html', show_warning=True, 
                                  selected_file=session.get('last_filename', ''),
                                  selected_method=method,
                                  selected_manual_method=request.form.get('manual_method'))
        
        if method == 'auto':
            decoded_text = lsb_decode(stego_path)  # Default to LSB for auto
            manual_method = None
        elif method == 'manual':
            manual_method = request.form['manual_method']
            if manual_method == 'lsb':
                decoded_text = lsb_decode(stego_path)
            elif manual_method == 'ai':
                stego_img = preprocess_image(stego_path)
                decoder = Decoder()
                msg_tensor = decoder(stego_img)
                decoded_text = ''.join([chr(int(b)) for b in (msg_tensor > 0).int().squeeze().tolist()][:16])
            elif manual_method == 'dct':
                decoded_text = dct_decode(stego_path)
        
        return render_template('decoder.html', 
                              decoded_text=decoded_text, 
                              selected_file=session.get('last_filename', ''),
                              selected_method=method, 
                              selected_manual_method=manual_method,
                              show_warning=False)
    return render_template('decoder.html', show_warning=False, selected_file=session.get('last_filename', ''))

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)