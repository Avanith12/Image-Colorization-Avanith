import os
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models
import torchvision.models as models
from torchvision.models import VGG19_Weights


# ====================== ARGUMENT PARSER ====================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Run ViT Colorization Model Training")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--image_size', type=int, default=224, help="Image size for training")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--max_epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--validation_interval', type=int, default=2, help="Validation interval")
    
    # File names and paths
    parser.add_argument('--loss_log_filename', type=str, default="loss_log.csv", help="Loss log file name")
    parser.add_argument('--loss_plot_filename', type=str, default="loss_plot.png", help="Loss plot file name")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--checkpoint_filename', type=str, default="best_vit_colorization", help="Checkpoint file name")
    parser.add_argument('--plot_figsize', type=tuple, default=(10, 7), help="Plot figure size")
    parser.add_argument('--plot_dpi', type=int, default=300, help="Plot DPI")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for model layers")
    
    # File paths
    parser.add_argument('--cache_dir', type=str, default="/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache", help="Cache directory")
    parser.add_argument('--folder_path', type=str, default="/home/huuthanhvy.nguyen001/colorization/Image", help="Folder path with images")
    
    args = parser.parse_args()
    return args

# ====================== PERCEPTUAL LOSS ====================== #

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load VGG19 with pretrained weights from the new API
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16]  # Use first 16 layers
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        
        self.vgg.eval()  # Set to evaluation mode (no dropout/batch norm updates)
        self.l1_loss = nn.L1Loss()  # L1 loss for perceptual loss calculation

    def forward(self, pred, target):
        # Ensure the input images have a proper batch dimension
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.l1_loss(pred_features, target_features)


# ====================== CALLBACK: LOSS PLOTTING ====================== #
class LossPlotCallback(LearningRateMonitor):
    def __init__(self, cache_dir, loss_log_filename, loss_plot_filename, plot_figsize, plot_dpi):
        super().__init__()
        self.cache_dir = cache_dir
        self.loss_log_filename = loss_log_filename
        self.loss_plot_filename = loss_plot_filename
        self.plot_figsize = plot_figsize
        self.plot_dpi = plot_dpi

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", None)
        val_loss = metrics.get("val_loss", None)

        log_path = os.path.join(self.cache_dir, self.loss_log_filename)
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
        else:
            df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

        new_data = pd.DataFrame({
            "epoch": [trainer.current_epoch + 1], 
            "train_loss": [train_loss.cpu().numpy() if train_loss is not None else None], 
            "val_loss": [val_loss.cpu().numpy() if val_loss is not None else None]
        })

        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(log_path, index=False)

        plt.figure(figsize=self.plot_figsize)
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", linestyle="-")
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="o", linestyle="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.cache_dir, self.loss_plot_filename)
        plt.savefig(plot_path, dpi=self.plot_dpi)
        plt.close()

# ====================== CALLBACK: TEXT LOGGER ====================== #
class TextLogger:
    def __init__(self, log_dir):
        self.log_file = os.path.join(log_dir, "training_log.txt")
        with open(self.log_file, "w") as f:
            f.write("Epoch | Train Loss | Validation Loss | Test Loss | PSNR\n")
            f.write("-" * 60 + "\n")
    def log_epoch(self, epoch, train_loss, val_loss):
        train_loss = f"{train_loss:.6f}" if train_loss is not None else "N/A"
        val_loss = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        with open(self.log_file, "a") as f:
            f.write(f"{epoch:>5} | {train_loss:>10} | {val_loss:>15} | {'-'*8} | {'-'*5}\n")
    def log_test(self, test_loss, psnr):
        test_loss = f"{test_loss:.6f}" if test_loss is not None else "N/A"
        psnr = f"{psnr:.2f} dB" if psnr is not None else "N/A"
        with open(self.log_file, "a") as f:
            f.write(f"{'-'*5} | {'-'*10} | {'-'*15} | {test_loss:>8} | {psnr:>5}\n")
            f.write("=" * 60 + "\n")

# ====================== CALLBACK: LOSS LOGGING ====================== #
class LossLoggingCallback(Callback):
    def __init__(self, text_logger):
        super().__init__()
        self.text_logger = text_logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        train_loss = trainer.callback_metrics.get("train_loss", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        self.text_logger.log_epoch(epoch, train_loss, val_loss)

# ====================== DATASET ====================== #
class ColorizationDataset(Dataset):
    def __init__(self, image_paths, transform=None, cache_dir=None):
        self.image_paths = image_paths
        self.transform = transform
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=cache_dir,
            do_rescale=False
        )
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Ensure the image is in CHW format as a tensor; then convert to HWC for color conversion
        if torch.is_tensor(img):
            img_for_lab = np.array(img.permute(1, 2, 0))
        else:
            img_for_lab = np.array(img)
        img_lab = rgb2lab(img_for_lab).astype("float32")
        img_l = img_lab[:, :, 0] / 100.0  # Normalize L to [0,1]
        img_ab = img_lab[:, :, 1:] / 128.0  # Normalize AB to approx. [-1,1]
        img_l = torch.tensor(img_l).unsqueeze(0)  # (1, H, W)
        img_ab = torch.tensor(img_ab).permute(2, 0, 1)  # (2, H, W)
        vit_inputs = self.feature_extractor(img, return_tensors="pt")
        pixel_values = vit_inputs["pixel_values"].squeeze(0)  # (3, 224, 224)
        return img_l, img_ab, pixel_values

# ====================== MODEL ====================== #
class ViTColorizationModel(LightningModule):
    def __init__(self, model_id="google/vit-base-patch16-224-in21k", dropout_rate=0.5, cache_dir=None, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters(ignore=["cache_dir"])  # cache_dir is not saved
        self.learning_rate = learning_rate
        self.vit = ViTModel.from_pretrained(model_id, cache_dir=cache_dir)
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
            nn.Tanh()
        )
        self.perceptual_loss = PerceptualLoss()  # Initialize VGG19 Perceptual Loss

    def forward(self, pixel_values):
        features = self.vit(pixel_values).last_hidden_state  # (B, 197, 768)
        features = features[:, 1:, :]  # Remove CLS token -> (B, 196, 768)
        batch_size, seq_len, hidden_dim = features.shape
        num_patches = int(seq_len ** 0.5)
        features = features.permute(0, 2, 1).reshape(batch_size, hidden_dim, num_patches, num_patches)
        ab_output = self.decoder(features)
        return ab_output

    def training_step(self, batch, batch_idx):
        img_l, img_ab, pixel_values = batch
        img_l, img_ab, pixel_values = img_l.to(self.device), img_ab.to(self.device), pixel_values.to(self.device)
        pred_ab = self(pixel_values)

        # Compute MSE loss
        mse_loss = F.mse_loss(pred_ab, img_ab)

        # Convert AB channels back to a pseudo-RGB (using L channel and AB channels)
        pred_rgb = torch.cat([img_l, pred_ab], dim=1)  # (B, 3, H, W)
        target_rgb = torch.cat([img_l, img_ab], dim=1)

        # Compute Perceptual Loss
        perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)

        # Combine losses
        total_loss = mse_loss + 0.1 * perceptual_loss  # Adjust weight as needed

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        img_l, img_ab, pixel_values = batch
        img_l, img_ab, pixel_values = img_l.to(self.device), img_ab.to(self.device), pixel_values.to(self.device)
        pred_ab = self(pixel_values)

        mse_loss = F.mse_loss(pred_ab, img_ab)

        pred_rgb = torch.cat([img_l, pred_ab], dim=1)
        target_rgb = torch.cat([img_l, img_ab], dim=1)
        perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)

        total_loss = mse_loss + 0.1 * perceptual_loss

        self.log("val_loss", total_loss.cpu(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_mse_loss", mse_loss.cpu(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_perceptual_loss", perceptual_loss.cpu(), on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        img_l, img_ab, pixel_values = batch
        img_l, img_ab, pixel_values = img_l.to(self.device), img_ab.to(self.device), pixel_values.to(self.device)
        pred_ab = self(pixel_values)

        mse_loss = F.mse_loss(pred_ab, img_ab)

        pred_rgb = torch.cat([img_l, pred_ab], dim=1)
        target_rgb = torch.cat([img_l, img_ab], dim=1)
        perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)

        total_loss = mse_loss + 0.1 * perceptual_loss
        psnr = 10 * torch.log10(1.0 / total_loss)

        self.log("test_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# ====================== MAIN TRAINING SCRIPT ====================== #
def main():
    args = parse_args()
    
    # Build a unique cache subdirectory based on hyperparameters.
    folder_name = f"vgg19_bs_{args.batch_size}_lr_{args.learning_rate}_epochs_{args.max_epochs}_dropout_{args.dropout_rate}"
    CACHE_DIR = os.path.join(args.cache_dir, folder_name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    FOLDER_PATH = args.folder_path
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    LEARNING_RATE = args.learning_rate
    MAX_EPOCHS = args.max_epochs
    NUM_WORKERS = args.num_workers
    VALIDATION_INTERVAL = args.validation_interval
    LOSS_LOG_FILENAME = args.loss_log_filename
    LOSS_PLOT_FILENAME = args.loss_plot_filename
    CHECKPOINT_DIR = args.checkpoint_dir
    CHECKPOINT_FILENAME = args.checkpoint_filename
    PLOT_FIGSIZE = args.plot_figsize
    PLOT_DPI = args.plot_dpi

    # Gather image files.
    image_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.png'))]
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = ColorizationDataset(image_files, transform=transform, cache_dir=CACHE_DIR)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ViTColorizationModel(dropout_rate=args.dropout_rate, cache_dir=CACHE_DIR, learning_rate=LEARNING_RATE)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CACHE_DIR, CHECKPOINT_DIR),
        filename=CHECKPOINT_FILENAME,
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    
    text_logger = TextLogger(CACHE_DIR)
    loss_logging_callback = LossLoggingCallback(text_logger)
    
    loss_plot_callback = LossPlotCallback(
        cache_dir=CACHE_DIR,
        loss_log_filename=LOSS_LOG_FILENAME,
        loss_plot_filename=LOSS_PLOT_FILENAME,
        plot_figsize=PLOT_FIGSIZE,
        plot_dpi=PLOT_DPI
    )
    
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        precision=16,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        accumulate_grad_batches=4,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback, loss_plot_callback, loss_logging_callback],
        default_root_dir=CACHE_DIR,
        check_val_every_n_epoch=VALIDATION_INTERVAL
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
