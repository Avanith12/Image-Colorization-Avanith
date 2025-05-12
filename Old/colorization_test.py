import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from transformers import ViTImageProcessor  # Use ViTImageProcessor instead of ViTFeatureExtractor
import numpy as np
from lightning.pytorch.core import LightningModule
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from transformers import ViTModel


# ====================== PATHS & CONFIG ====================== #
CACHE_DIR = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache/vgg19_bs_4_lr_0.0001_epochs_20_dropout_0.1"
CHECKPOINT_PATH = os.path.join(CACHE_DIR, "checkpoints/best_vit_colorization.ckpt")
IMAGE_SIZE = 224
DPI = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== VGG19 PERCEPTUAL LOSS ====================== #
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16]  # Use first 16 layers
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.l1_loss(pred_features, target_features)

# ====================== DEFINE MODEL CLASS ====================== #
from transformers import ViTModel
import torch.nn as nn

class ViTColorizationModel(LightningModule):
    def __init__(self, model_id="google/vit-base-patch16-224-in21k", dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Load ViT model
        self.vit = ViTModel.from_pretrained(model_id, cache_dir=CACHE_DIR)
        
        # CNN Decoder with dropout
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Add perceptual loss (VGG19)
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pixel_values):
        features = self.vit(pixel_values).last_hidden_state  # (B, 197, 768)
        
        # Remove CLS token (first token)
        features = features[:, 1:, :]  # Now shape is (B, 196, 768)

        # Ensure correct reshaping
        batch_size, seq_len, hidden_dim = features.shape
        num_patches = int(seq_len ** 0.5)

        # Reshape to (B, 768, 14, 14) before CNN decoder
        features = features.permute(0, 2, 1).reshape(batch_size, hidden_dim, num_patches, num_patches)

        # Pass through CNN decoder
        ab_output = self.decoder(features)  # Output: (B, 2, 224, 224)
        return ab_output

    def calculate_loss(self, pred_ab, img_ab, img_l, pred_rgb, target_rgb):
        mse_loss = F.mse_loss(pred_ab, img_ab)
        perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)
        
        total_loss = mse_loss + 0.1 * perceptual_loss  # Combine MSE and perceptual loss
        return total_loss, mse_loss, perceptual_loss

# ====================== LOAD CHECKPOINT ====================== #
def load_model():
    model = ViTColorizationModel(dropout_rate=0.1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # Allow mismatched keys
    model.to(DEVICE)
    model.eval()

    return model

# Load ViT feature extractor (Use ViTImageProcessor)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR, do_rescale=True)

# ====================== PREDICTION FUNCTION ====================== #
def prediction_vit_cnn_vgg19(image_path, save_path="colorized_output.png"):
    """Colorizes the input image using ViT + CNN decoder and VGG19 perceptual loss."""
    # Load the model
    model = load_model()

    # Open and process image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to LAB color space
    img_np = np.array(img_resized)
    img_lab = rgb2lab(img_np).astype("float32")

    # Extract L (grayscale) channel
    img_l = img_lab[:, :, 0] / 100.0
    img_l_tensor = torch.tensor(img_l).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Extract ViT features
    vit_inputs = feature_extractor(img_resized, return_tensors="pt")
    pixel_values = vit_inputs["pixel_values"].to(DEVICE)

    # Predict color channels
    with torch.no_grad():
        pred_ab = model(pixel_values).squeeze(0)

    # Convert LAB to RGB
    pred_ab = pred_ab.cpu() * 128
    img_l = img_l_tensor.cpu().squeeze(0) * 100
    lab_image = torch.cat([img_l, pred_ab], dim=0).permute(1, 2, 0).numpy()
    colorized_img = lab2rgb(lab_image)

    """
    # Display and save the colorized image
    plt.figure(figsize=(12, 4))

    # Grayscale Input
    plt.subplot(1, 3, 1)
    plt.imshow(img.convert("L"), cmap="gray")
    plt.title("Grayscale Input")
    plt.axis("off")

    # Colorized Output
    plt.subplot(1, 3, 2)
    plt.imshow(colorized_img)
    plt.title("Colorized Output")
    plt.axis("off")

    # Original Image
    plt.subplot(1, 3, 3)
    original_img = Image.open(image_path)  # Load the original image
    plt.imshow(original_img)  # Display it correctly
    plt.title("Original Image")
    plt.axis("off")

    plt.savefig(save_path)
    plt.show()

    """

# ====================== USAGE ====================== #
# Now you can call this function like so:
# prediction_vit_cnn_vgg19("/path/to/test/image.jpg", save_path="output_image.png")

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from transformers import ViTImageProcessor  # Use ViTImageProcessor instead of ViTFeatureExtractor
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
from transformers import ViTModel
import torch.nn as nn
import torch.nn.functional as F

# ====================== PATHS & CONFIG ====================== #
CACHE_DIR = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache/bs_1_lr_0.0001_epochs_20_dropout_0.1"
CHECKPOINT_PATH = os.path.join(CACHE_DIR, "checkpoints/best_vit_colorization.ckpt")
IMAGE_SIZE = 224
DPI = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== DEFINE MODEL CLASS ====================== #
class ViTColorizationModel(LightningModule):
    def __init__(self, model_id="google/vit-base-patch16-224-in21k", dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Load ViT model
        self.vit = ViTModel.from_pretrained(model_id, cache_dir=CACHE_DIR)
        
        # CNN Decoder with dropout
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout with rate
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout with rate
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout with rate
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout with rate
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize to [-1, 1]
        )

    def forward(self, pixel_values):
        features = self.vit(pixel_values).last_hidden_state  # (B, 197, 768)

        # Remove CLS token (first token)
        features = features[:, 1:, :]  # Now shape is (B, 196, 768)

        # Ensure correct reshaping
        batch_size, seq_len, hidden_dim = features.shape
        num_patches = int(seq_len ** 0.5)

        # Reshape to (B, 768, 14, 14) before CNN decoder
        features = features.permute(0, 2, 1).reshape(batch_size, hidden_dim, num_patches, num_patches)

        # Pass through CNN decoder
        ab_output = self.decoder(features)  # Output: (B, 2, 224, 224)
        return ab_output

# ====================== LOAD CHECKPOINT ====================== #
def load_model():
    model = ViTColorizationModel(dropout_rate=0.1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # Allow mismatched keys
    model.to(DEVICE)
    model.eval()

    return model

# Load ViT feature extractor (Use ViTImageProcessor)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR, do_rescale=True)

# ====================== PREDICTION FUNCTION ====================== #
def prediction_vit_cnn(image_path, save_path="colorized_output.png"):
    """Colorizes the input image using ViT + CNN decoder."""
    # Load the model
    model = load_model()

    # Open and process image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to LAB color space
    img_np = np.array(img_resized)
    img_lab = rgb2lab(img_np).astype("float32")

    # Extract L (grayscale) channel
    img_l = img_lab[:, :, 0] / 100.0
    img_l_tensor = torch.tensor(img_l).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Extract ViT features
    vit_inputs = feature_extractor(img_resized, return_tensors="pt")
    pixel_values = vit_inputs["pixel_values"].to(DEVICE)

    # Predict color channels
    with torch.no_grad():
        pred_ab = model(pixel_values).squeeze(0)

    # Convert LAB to RGB
    pred_ab = pred_ab.cpu() * 128
    img_l = img_l_tensor.cpu().squeeze(0) * 100
    lab_image = torch.cat([img_l, pred_ab], dim=0).permute(1, 2, 0).numpy()
    colorized_img = lab2rgb(lab_image)

    """
    
    # Display and save the colorized image
    plt.figure(figsize=(12, 4))

    # Grayscale Input
    plt.subplot(1, 3, 1)
    plt.imshow(img.convert("L"), cmap="gray")
    plt.title("Grayscale Input")
    plt.axis("off")

    # Colorized Output
    plt.subplot(1, 3, 2)
    plt.imshow(colorized_img)
    plt.title("Colorized Output")
    plt.axis("off")

    # Original Image
    plt.subplot(1, 3, 3)
    original_img = Image.open(image_path)  # Load the original image
    plt.imshow(original_img)  # Display it correctly
    plt.title("Original Image")
    plt.axis("off")

    plt.savefig(save_path)
    plt.show()
    
    """

# ====================== USAGE ====================== #
# You can now call prediction_vit_cnn() to colorize an image:
# prediction_vit_cnn("/path/to/test/image.jpg", save_path="colorized_image.png")
