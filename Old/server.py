# server.py
from flask import Flask, request, render_template, send_file
import os
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image, ImageEnhance
import torch
import traceback
import numpy as np
from skimage import color

# Import model
from model import build_res_unet

# Setup Flask app
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model ONCE at server startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_res_unet(n_input=1, n_output=2, size=256)
checkpoint = torch.load('final_model_weights.pt', map_location=device)
new_state_dict = {k.replace('net_G.', ''): v for k, v in checkpoint.items() if k.startswith('net_G.')}
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# Route: Home page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Upload and colorize
@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    file.save(input_path)

    try:
        # --- 1. Preprocess ---
        img = Image.open(input_path).convert('RGB')  # Load as RGB first

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img)  # (3, 256, 256)

        # Convert RGB -> LAB
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_lab = color.rgb2lab(img_np).astype("float32")

        L = img_lab[:, :, 0]
        L = (L / 50.) - 1.0  # Normalize L to [-1, 1]

        L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 256, 256)

        # --- 2. Model Inference ---
        with torch.no_grad():
            ab_pred = model(L)  # (1, 2, 256, 256)

        # --- 3. Postprocess ---
        L = (L + 1.) * 50.0  # Recover original L scale
        ab_pred = ab_pred * 110.0  # Recover ab scale

        lab = torch.cat([L.squeeze(0), ab_pred.squeeze(0)], dim=0)  # (3, 256, 256)
        lab = lab.permute(1, 2, 0).cpu().numpy()  # (256, 256, 3)

        rgb = color.lab2rgb(lab)
        rgb_image = (rgb * 255).astype(np.uint8)
        output_image = Image.fromarray(rgb_image)

        # --- ✨ Enhancement ✨ ---
        enhancer = ImageEnhance.Sharpness(output_image)
        output_image = enhancer.enhance(2.0)

        color_enhancer = ImageEnhance.Color(output_image)
        output_image = color_enhancer.enhance(1.8)

        contrast_enhancer = ImageEnhance.Contrast(output_image)
        output_image = contrast_enhancer.enhance(1.3)
        # -------------------------

        # Save the output
        output_image.save(output_path, format='JPEG')

        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error during colorization: {e}")
        traceback.print_exc()
        return 'Failed to colorize image!', 500

if __name__ == '__main__':
    app.run(debug=True)
