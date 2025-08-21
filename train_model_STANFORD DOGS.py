import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import multiprocessing
import kornia.filters as kf
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.restoration import (wiener, richardson_lucy, denoise_tv_chambolle)
from PIL import Image
import glob
import urllib.request
import tarfile
import scipy.io as sio

# ======== STANFORD DOGS DATASET CLASS ========
class StanfordDogsDataset(Dataset):
    """Custom dataset class for Stanford Dogs dataset"""
    
    def __init__(self, root_dir, split='train', transform=None, download=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if download:
            self._download_dataset()
        
        self._load_dataset()
    
    def _download_dataset(self):
        """Download Stanford Dogs dataset if not already present"""
        images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
        lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
        
        os.makedirs(self.root_dir, exist_ok=True)
        
        images_path = os.path.join(self.root_dir, "images.tar")
        lists_path = os.path.join(self.root_dir, "lists.tar")
        
        # Download images if not exist
        if not os.path.exists(os.path.join(self.root_dir, "Images")):
            if not os.path.exists(images_path):
                print("Downloading Stanford Dogs images...")
                urllib.request.urlretrieve(images_url, images_path)
            
            print("Extracting images...")
            with tarfile.open(images_path, 'r') as tar:
                tar.extractall(self.root_dir)
        
        # Download lists if not exist
        if not os.path.exists(os.path.join(self.root_dir, "lists")):
            if not os.path.exists(lists_path):
                print("Downloading Stanford Dogs lists...")
                urllib.request.urlretrieve(lists_url, lists_path)
            
            print("Extracting lists...")
            with tarfile.open(lists_path, 'r') as tar:
                tar.extractall(self.root_dir)
    
    def _load_dataset(self):
        """Load the dataset file paths and labels"""
        images_dir = os.path.join(self.root_dir, "Images")
        
        if self.split == 'train':
            list_file = os.path.join(self.root_dir, "lists", "train_list.mat")
        else:
            list_file = os.path.join(self.root_dir, "lists", "test_list.mat")
        
        if os.path.exists(list_file):
            # Load from MATLAB file
            mat_data = sio.loadmat(list_file)
            if self.split == 'train':
                file_list = mat_data['file_list']
            else:
                file_list = mat_data['file_list']
            
            self.image_paths = []
            for file_path in file_list:
                full_path = os.path.join(images_dir, file_path[0][0])
                if os.path.exists(full_path):
                    self.image_paths.append(full_path)
        else:
            # Fallback: use all images in the directory
            print(f"List file not found, using all images in {images_dir}")
            self.image_paths = []
            for breed_dir in os.listdir(images_dir):
                breed_path = os.path.join(images_dir, breed_dir)
                if os.path.isdir(breed_path):
                    for img_file in os.listdir(breed_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(breed_path, img_file))
        
        # Split the data manually if needed
        if not os.path.exists(list_file):
            np.random.seed(42)
            np.random.shuffle(self.image_paths)
            split_idx = int(0.8 * len(self.image_paths))
            
            if self.split == 'train':
                self.image_paths = self.image_paths[:split_idx]
            else:
                self.image_paths = self.image_paths[split_idx:]
        
        print(f"Loaded {len(self.image_paths)} images for {self.split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Return image and dummy label (we don't need labels for deblurring)
            return image, 0
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random image if there's an error
            return torch.randn(3, 224, 224), 0

# ======== MODEL CLASSES (Same as before) ========
class SSMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        res = x  # Residual connection
        
        # Reshape for processing while preserving spatial information
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.layer_norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Reshape back to original spatial dimensions
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x + res  # Add residual
    
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        res = x  # Residual connection
        
        # Reshape for attention while preserving spatial information
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, -1, self.heads, C // self.heads).transpose(1, 2), qkv)
        
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)
        
        out = self.to_out(out)
        out = self.dropout(out)
        
        # Reshape back to original spatial dimensions
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + res  # Add residual

class ImprovedSharpnessModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Different sharpening kernels
        self.sharpen_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.sharpen_conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        
        # Initialize with different sharpening kernels
        with torch.no_grad():
            # Standard Laplacian
            kernel1 = torch.tensor([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Stronger sharpening kernel
            kernel2 = torch.tensor([
                [-1, -1, -1, -1, -1],
                [-1,  2,  2,  2, -1],
                [-1,  2,  8,  2, -1],
                [-1,  2,  2,  2, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
            
            self.sharpen_conv1.weight.data = kernel1.repeat(channels, 1, 1, 1)
            self.sharpen_conv2.weight.data = kernel2.repeat(channels, 1, 1, 1)
            self.sharpen_conv1.bias.data.zero_()
            self.sharpen_conv2.bias.data.zero_()
        
        self.alpha = nn.Parameter(torch.tensor(0.7))  # Stronger initial value
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.blend = nn.Conv2d(channels*2, channels, kernel_size=1)
        
    def forward(self, x):
        sharp1 = self.sharpen_conv1(x)
        sharp2 = self.sharpen_conv2(x)
        
        blended = self.blend(torch.cat([sharp1, sharp2], dim=1))
        return x + self.alpha * (blended - x)

class DetailPreservingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_detect = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        
        # Initialize with Sobel filters for edge detection
        with torch.no_grad():
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.edge_detect.weight.data = sobel_x.repeat(channels, 1, 1, 1)
        
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 1)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))  # Learnable intensity factor
        
    def forward(self, x):
        edges = self.edge_detect(x)
        enhanced_edges = self.edge_enhance(edges)
        return x + self.gamma * enhanced_edges

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class ImprovedEnhancedUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc1_bn = nn.BatchNorm2d(64)

        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_bn = nn.BatchNorm2d(128)

        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_bn = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)

        self.attn1 = SelfAttention(256)
        self.ssm1 = SSMBlock(256, 512)
        self.attn2 = SelfAttention(256)
        self.ssm2 = SSMBlock(256, 512)
        self.detail_enhance = DetailPreservingBlock(256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.dec3_bn = nn.BatchNorm2d(128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.dec2_bn = nn.BatchNorm2d(64)

        self.dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_bn = nn.BatchNorm2d(64)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

        self.refine1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.refine_out = nn.Conv2d(64, 3, kernel_size=1)

        self.sharpen = ImprovedSharpnessModule(3)

        self.vgg = nn.Identity()  # Replace with actual VGG for perceptual loss if needed

    def extract_features(self, x):
        # Used for perceptual loss
        return self.vgg(x)

    def forward(self, x):
        orig = x  # Save original input for residual connection

        # Encoder path
        e1 = F.leaky_relu(self.enc1_bn(self.enc1(x)), 0.2)
        e2 = F.leaky_relu(self.enc2_bn(self.enc2(self.pool(e1))), 0.2)
        e3 = F.leaky_relu(self.enc3_bn(self.enc3(self.pool(e2))), 0.2)

        # Bottleneck with attention and SSM
        middle = self.attn1(e3)
        middle = self.ssm1(middle)
        middle = self.attn2(middle)
        middle = self.ssm2(middle)
        middle = self.detail_enhance(middle)

        # Decoder path with skip connections
        d3 = self.up3(middle)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = F.leaky_relu(self.dec3_bn(self.dec3(d3)), 0.2)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = F.leaky_relu(self.dec2_bn(self.dec2(d2)), 0.2)

        d1 = F.leaky_relu(self.dec1_bn(self.dec1(d2)), 0.2)
        output = self.out_conv(d1)

        # Refinement and sharpening
        refined = output + orig
        refined = F.leaky_relu(self.refine1(refined), 0.2)
        refined = F.leaky_relu(self.refine2(refined), 0.2)
        refined = self.refine_out(refined) + output
        refined = self.sharpen(refined)

        return refined

class ImprovedNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        # Milder beta schedule for better image quality
        self.betas = torch.linspace(1e-6, 0.008, timesteps)  # Gentler noise schedule
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        
        noise = torch.randn_like(x0).to(device)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1).to(device)
        sqrt_1_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1).to(device)
        return sqrt_ab * x0 + sqrt_1_ab * noise, noise

class PerceptualLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.model = model  # VGG or other feature extractor
        
    def forward(self, pred, target):
        # Extract deep features
        pred_features = self.model.extract_features(pred)
        target_features = self.model.extract_features(target)
        
        # Perceptual loss (feature matching)
        perceptual_loss = self.mse(pred_features, target_features)
        
        # Direct pixel losses
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Total loss with weighted components
        return 0.5 * mse_loss + 0.2 * l1_loss + 0.3 * perceptual_loss

# Calculate PSNR and SSIM correctly
def calculate_metrics(img1, img2):
    # Convert from [-1,1] to [0,1]
    img1 = (img1.clamp(-1, 1) + 1) / 2
    img2 = (img2.clamp(-1, 1) + 1) / 2
    
    # Move to CPU and convert to numpy
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    
    # Calculate PSNR
    psnr = psnr_metric(img1, img2, data_range=1.0)
    
    # Calculate SSIM (channel by channel)
    ssim_vals = []
    for ch in range(img1.shape[1]):
        ssim_val = ssim_metric(img1[0, ch], img2[0, ch], data_range=1.0)
        ssim_vals.append(ssim_val)
    
    ssim = np.mean(ssim_vals)
    
    return psnr, ssim

def apply_blur(images, severity=0.5):
    """Apply blur effects with configurable severity
    
    Args:
        images: Input tensor of shape [B, C, H, W]
        severity: Float between 0 and 1 indicating blur intensity
        
    Returns:
        Blurred images tensor of same shape as input
    """
    device = images.device
    # Random mix of different blur types
    blur_type = np.random.choice(['gaussian', 'motion', 'defocus'])

    if blur_type == 'gaussian':
        sigma = torch.tensor(np.random.uniform(0.5, 2.0) * severity, device=device, dtype=torch.float32)
        return kf.gaussian_blur2d(images, kernel_size=(5, 5), sigma=(sigma, sigma))

    elif blur_type == 'motion':
        # Ensure kernel_size is always odd and at least 3
        base_size = 3
        additional = int(4 * severity * np.random.random())
        kernel_size = base_size + 2 * additional  # This formula always produces odd numbers (3, 5, 7, 9, 11)
        
        # Generate random angle in degrees (0-360)
        angle = np.random.uniform(0, 360)
        
        # Direction parameter should be a scalar between -1 and 1
        direction = -1.0 + 2.0 * np.random.random()  # -1 to 1
        
        return kf.motion_blur(images, kernel_size=kernel_size, angle=angle, direction=direction)

    else:  # defocus
        # Ensure kernel_size is always odd
        base_size = 3
        additional = int(4 * severity * np.random.random())
        kernel_size = base_size + 2 * additional  # This formula always produces odd numbers
        return kf.box_blur(images, kernel_size=(kernel_size, kernel_size))

@torch.no_grad()
def sample_and_evaluate(model, val_loader, device, num_samples=4):
    model.eval()
    
    # Process a few validation samples
    val_images = []
    blurred_images = []
    enhanced_images = []
    metrics = []
    
    for val_x, _ in val_loader:
        if len(val_images) >= num_samples:
            break
            
        val_x = val_x.to(device)
        
        # Apply moderate blur
        blurred = apply_blur(val_x, severity=0.7)
        
        # Run through model
        enhanced = model(blurred)
        
        # Calculate metrics
        for i in range(val_x.size(0)):
            if len(val_images) >= num_samples:
                break
                
            psnr, ssim = calculate_metrics(enhanced[i:i+1], val_x[i:i+1])
            
            val_images.append(val_x[i:i+1])
            blurred_images.append(blurred[i:i+1])
            enhanced_images.append(enhanced[i:i+1])
            metrics.append((psnr, ssim))
    
    return val_images, blurred_images, enhanced_images, metrics

def deblur_image(input_path, model, device, output_path=None):
    """Deblur a single image"""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced.png"

    # Load image
    img = Image.open(input_path).convert('RGB')

    # Preprocess - resize to model input size while maintaining aspect ratio
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match training size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        enhanced = model(input_tensor)

    # Post-process and save
    output_img = (enhanced[0].clamp(-1, 1) + 1) / 2
    output_img = output_img.cpu().permute(1, 2, 0).numpy() * 255
    output_img = Image.fromarray(output_img.astype(np.uint8))
    
    # Resize back to original size
    original_size = img.size
    output_img = output_img.resize(original_size, Image.LANCZOS)
    output_img.save(output_path)

    print(f"Enhanced image saved to {output_path}")
    return output_path

if __name__ == '__main__':
    # This is required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    # ======== SETUP ========
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
        
    print(f"Using device: {device}")
    os.makedirs("enhanced_samples", exist_ok=True)

    # ======== DATA - STANFORD DOGS ========
    img_size = 224  # Standard size for dog images
    batch_size = 8   # Smaller batch size for larger images
    learning_rate = 1e-4
    epochs = 30

    # Transform for Stanford Dogs - includes data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create Stanford Dogs datasets
    data_root = './data/stanford_dogs'
    
    print("Loading Stanford Dogs dataset...")
    train_dataset = StanfordDogsDataset(
        root_dir=data_root, 
        split='train', 
        transform=train_transform,
        download=True
    )
    
    val_dataset = StanfordDogsDataset(
        root_dir=data_root, 
        split='test', 
        transform=val_transform,
        download=False  # Already downloaded
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create improved model
    model = ImprovedEnhancedUNet().to(device)

    # Calculate parameters
    macs, params = get_model_complexity_info(ImprovedEnhancedUNet(), (3, img_size, img_size), as_strings=True, print_per_layer_stat=False)
    print(f"Model GFLOPs: {macs}, Params: {params}")

    # Initialize noise scheduler
    noise_scheduler = ImprovedNoiseScheduler()
    
    # Initialize perceptual loss function
    loss_fn = PerceptualLoss(model)

    # ======== PRETRAINING PHASE ========
    print("Starting deblurring pretraining phase...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate*2, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    best_metrics = {'psnr': 0, 'ssim': 0}

    # Pretrain specifically for deblurring
    for epoch in range(5):
        model.train()
        running_loss = 0
        with tqdm(train_loader) as pbar:
            for x, _ in pbar:
                x = x.to(device)
                
                # Apply synthetic blur
                blurred = apply_blur(x, severity=np.random.uniform(0.3, 0.7))
                
                # Direct image reconstruction
                output = model(blurred)
                
                # Calculate MSE loss for pretraining
                loss = F.mse_loss(output, x)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        
        # Validation with comprehensive metrics
        model.eval()
        val_psnr = 0
        val_ssim = 0
        samples_processed = 0
        
        with torch.no_grad():
            for val_x, _ in tqdm(val_loader, desc="Validating"):
                if samples_processed >= 100:  # Evaluate on 100 samples
                    break
                    
                val_x = val_x.to(device)
                blurred = apply_blur(val_x, severity=0.6)  # Consistent test difficulty
                output = model(blurred)
                
                # Calculate metrics
                for i in range(val_x.size(0)):
                    if samples_processed >= 100:
                        break
                        
                    psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                    val_psnr += psnr
                    val_ssim += ssim
                    samples_processed += 1
        
        avg_psnr = val_psnr / samples_processed
        avg_ssim = val_ssim / samples_processed
        
        # Save model if metrics improve
        improved = False
        if avg_psnr > best_metrics['psnr']:
            best_metrics['psnr'] = avg_psnr
            improved = True
            
        if avg_ssim > best_metrics['ssim']:
            best_metrics['ssim'] = avg_ssim
            improved = True
            
        if improved:
            torch.save(model.state_dict(), "best_deblur_model_dogs.pth")
            print(f"✓ New best model saved!")
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.6f} | " 
              f"Val PSNR: {avg_psnr:.2f}dB | Val SSIM: {avg_ssim:.4f} | "
              f"Best PSNR: {best_metrics['psnr']:.2f}dB | Best SSIM: {best_metrics['ssim']:.4f}")
        
        # Visualize samples periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_images, blurred_images, enhanced_images, metrics = sample_and_evaluate(model, val_loader, device)
            
            for i, (val_img, blur_img, enh_img, (psnr, ssim)) in enumerate(zip(val_images, blurred_images, enhanced_images, metrics)):
                # Convert to display format
                orig = (val_img[0].clamp(-1, 1) + 1) / 2
                blur = (blur_img[0].clamp(-1, 1) + 1) / 2
                enhanced = (enh_img[0].clamp(-1, 1) + 1) / 2
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(orig.permute(1, 2, 0).cpu().numpy())
                plt.title("Original Dog")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(blur.permute(1, 2, 0).cpu().numpy())
                plt.title("Blurred Dog")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(enhanced.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Enhanced Dog (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"enhanced_samples/dogs_epoch_{epoch+1}_sample_{i+1}.png", dpi=150)
                plt.close()

    # ======== FINAL MODEL EVALUATION ========
    # Load best model
    model.load_state_dict(torch.load("best_deblur_model_dogs.pth"))
    model.eval()

    # Final evaluation with varying levels of blur
    print("Performing final quality evaluation on Stanford Dogs...")
    
    blur_levels = [0.3, 0.6, 0.9]  # Light, medium, heavy blur
    results = {level: {'psnr': 0, 'ssim': 0, 'count': 0} for level in blur_levels}
    
    with torch.no_grad():
        for val_x, _ in tqdm(val_loader, desc="Final evaluation"):
            val_x = val_x.to(device)
            
            # Test each blur level
            for blur_level in blur_levels:
                blurred = apply_blur(val_x, severity=blur_level)
                output = model(blurred)
                
                # Calculate metrics for each image
                for i in range(val_x.size(0)):
                    psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                    
                    results[blur_level]['psnr'] += psnr
                    results[blur_level]['ssim'] += ssim
                    results[blur_level]['count'] += 1
                    
                    # Save some examples for each blur level
                    if results[blur_level]['count'] <= 5:
                        # Convert to display format
                        orig = (val_x[i].clamp(-1, 1) + 1) / 2
                        blur = (blurred[i].clamp(-1, 1) + 1) / 2
                        enhanced = (output[i].clamp(-1, 1) + 1) / 2
                        
                        # Create visualization
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(orig.permute(1, 2, 0).cpu().numpy())
                        plt.title("Original Dog")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(blur.permute(1, 2, 0).cpu().numpy())
                        plt.title(f"Blurred Dog (Level: {blur_level:.1f})")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(enhanced.permute(1, 2, 0).cpu().numpy())
                        plt.title(f"Enhanced Dog (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(f"enhanced_samples/dogs_final_blur_{blur_level:.1f}_sample_{results[blur_level]['count']}.png", dpi=150)
                        plt.close()
                        
                # Stop after processing a reasonable number of images
                if results[blur_level]['count'] >= 100:
                    break
            
            # Check if we've processed enough images for all blur levels
            if all(results[level]['count'] >= 100 for level in blur_levels):
                break

    # Print final results
    print("\n===== FINAL RESULTS ON STANFORD DOGS =====")
    for level in blur_levels:
        avg_psnr = results[level]['psnr'] / results[level]['count']
        avg_ssim = results[level]['ssim'] / results[level]['count']
        print(f"Blur Level {level:.1f}: PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")

    # ======== DEMONSTRATION FUNCTIONS ========
    def batch_deblur_dogs(input_dir, output_dir=None):
        """Batch process dog images for deblurring"""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "enhanced_dogs")
        os.makedirs(output_dir, exist_ok=True)

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        print(f"Found {len(image_files)} dog images to process")

        for img_path in tqdm(image_files, desc="Batch processing dog images"):
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_enhanced.png")
            
            try:
                deblur_image(img_path, model, device, output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Batch processing complete. Enhanced dog images saved to {output_dir}")

    # ======== SAVE MODEL AND CREATE DEMO ========
    print("\nSaving final model for Stanford Dogs...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'img_size': img_size,
        'architecture': 'ImprovedEnhancedUNet',
        'dataset': 'Stanford Dogs',
        'date_trained': 'July 2025'
    }, "deblur_model_dogs_complete.pth")
    
    # Save smaller model for deployment
    torch.save(model.state_dict(), "deblur_model_dogs_state.pth")

    # Generate final demonstration with best dog examples
    print("\nGenerating final demonstration with dog images...")
    
    # Select best examples from dogs dataset
    best_examples = []
    best_metrics = []
    
    with torch.no_grad():
        for val_x, _ in val_loader:
            if len(best_examples) >= 8:  # More examples for dogs
                break
                
            val_x = val_x.to(device)
            # Apply strong blur for impressive demo
            blurred = apply_blur(val_x, severity=0.8)
            output = model(blurred)
            
            # Select images with good metric improvements
            for i in range(val_x.size(0)):
                if len(best_examples) >= 8:
                    break
                    
                psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                psnr_blurred, ssim_blurred = calculate_metrics(blurred[i:i+1], val_x[i:i+1])
                
                # Calculate improvement
                psnr_improvement = psnr - psnr_blurred
                ssim_improvement = ssim - ssim_blurred
                
                if psnr_improvement > 5.0 and ssim_improvement > 0.1:
                    best_examples.append((val_x[i:i+1], blurred[i:i+1], output[i:i+1]))
                    best_metrics.append((psnr, ssim, psnr_improvement, ssim_improvement))

    # Create detailed visualization grid for dogs
    for idx, ((orig, blur, enhanced), (psnr, ssim, psnr_imp, ssim_imp)) in enumerate(zip(best_examples, best_metrics)):
        # Convert tensors for display
        orig_img = (orig[0].clamp(-1, 1) + 1) / 2
        blur_img = (blur[0].clamp(-1, 1) + 1) / 2
        enhanced_img = (enhanced[0].clamp(-1, 1) + 1) / 2
        
        # Create advanced visualization
        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 3, 1)
        plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        plt.title("Original Dog Image", fontsize=14)
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(blur_img.permute(1, 2, 0).cpu().numpy())
        plt.title("Blurred Dog Image", fontsize=14)
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(enhanced_img.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Enhanced Dog Image\nPSNR: {psnr:.2f}dB (+{psnr_imp:.2f}dB)\nSSIM: {ssim:.4f} (+{ssim_imp:.4f})", fontsize=14)
        plt.axis('off')
        
        # Zoom in on a region of interest (center crop for detail)
        h, w = orig_img.shape[1], orig_img.shape[2]
        crop_h, crop_w = h // 2, w // 2
        start_h, start_w = h // 4, w // 4
        
        # Get zoom crops
        orig_crop = orig_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        blur_crop = blur_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        enhanced_crop = enhanced_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        plt.subplot(2, 3, 4)
        plt.imshow(orig_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Original (Detail View)", fontsize=12)
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(blur_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Blurred (Detail View)", fontsize=12)
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(enhanced_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Enhanced (Detail View)", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"enhanced_samples/dogs_final_demo_{idx+1}_detailed.png", dpi=200, bbox_inches='tight')
        plt.close()

    # Create a summary grid showing multiple dog examples
    plt.figure(figsize=(20, 15))
    for idx, ((orig, blur, enhanced), _) in enumerate(zip(best_examples[:6], best_metrics[:6])):
        if idx >= 6:
            break
            
        orig_img = (orig[0].clamp(-1, 1) + 1) / 2
        blur_img = (blur[0].clamp(-1, 1) + 1) / 2
        enhanced_img = (enhanced[0].clamp(-1, 1) + 1) / 2
        
        # Original
        plt.subplot(6, 3, idx*3 + 1)
        plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        if idx == 0:
            plt.title("Original Dogs", fontsize=16)
        plt.axis('off')
        
        # Blurred
        plt.subplot(6, 3, idx*3 + 2)
        plt.imshow(blur_img.permute(1, 2, 0).cpu().numpy())
        if idx == 0:
            plt.title("Blurred Dogs", fontsize=16)
        plt.axis('off')
        
        # Enhanced
        plt.subplot(6, 3, idx*3 + 3)
        plt.imshow(enhanced_img.permute(1, 2, 0).cpu().numpy())
        if idx == 0:
            plt.title("Enhanced Dogs", fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("enhanced_samples/dogs_summary_grid.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Training complete! Stanford Dogs deblurring examples saved to 'enhanced_samples' directory.")
    print(f"Model saved as 'deblur_model_dogs_complete.pth'")
    print("Use deblur_image('path/to/dog_image.jpg', model, device) to enhance individual dog images")
    print("Use batch_deblur_dogs('path/to/dog_folder') to process multiple dog images")
   
   # ======= REMOVE OR COMMENT OUT THIS =========
# loss.backward()
# optimizer.step()
#                 
# running_loss += loss.item()
# pbar.set_description(f"Pretrain {epoch+1}/5 | Loss: {loss.item():.6f}")
#         
# avg_loss = running_loss / len(train_loader)
# print(f"Pretrain Epoch [{epoch+1}/5] | Avg Loss: {avg_loss:.6f}")


    # ======== MAIN TRAINING WITH PROGRESSIVE DIFFICULTY ========
    print("Starting main training phase...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=learning_rate/10)

    best_metrics = {'psnr': 0, 'ssim': 0}
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        # Progressive difficulty - gradually increase blur severity
        min_severity = 0.3
        max_severity = min(0.3 + epoch * 0.02, 0.8)
        
        with tqdm(train_loader) as pbar:
            for x, _ in pbar:
                x = x.to(device)
                
                # Apply synthetic blur with progressive difficulty
                severity = np.random.uniform(min_severity, max_severity)
                blurred = apply_blur(x, severity=severity)
                
                # Model prediction
                pred = model(blurred)
                
                # Use perceptual loss for better visual quality
                loss = loss_fn(pred, x)
                
                optimizer.zero_grad()
		
		# Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        
        # Validation with comprehensive metrics
        model.eval()
        val_psnr = 0
        val_ssim = 0
        samples_processed = 0
        
        with torch.no_grad():
            for val_x, _ in tqdm(val_loader, desc="Validating"):
                if samples_processed >= 100:  # Evaluate on 100 samples
                    break
                    
                val_x = val_x.to(device)
                blurred = apply_blur(val_x, severity=0.6)  # Consistent test difficulty
                output = model(blurred)
                
                # Calculate metrics
                for i in range(val_x.size(0)):
                    if samples_processed >= 100:
                        break
                        
                    psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                    val_psnr += psnr
                    val_ssim += ssim
                    samples_processed += 1
        
        avg_psnr = val_psnr / samples_processed
        avg_ssim = val_ssim / samples_processed
        
        # Save model if metrics improve
        improved = False
        if avg_psnr > best_metrics['psnr']:
            best_metrics['psnr'] = avg_psnr
            improved = True
            
        if avg_ssim > best_metrics['ssim']:
            best_metrics['ssim'] = avg_ssim
            improved = True
            
        if improved:
            torch.save(model.state_dict(), "best_deblur_model_dogs.pth")
            print(f"✓ New best model saved!")
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.6f} | " 
              f"Val PSNR: {avg_psnr:.2f}dB | Val SSIM: {avg_ssim:.4f} | "
              f"Best PSNR: {best_metrics['psnr']:.2f}dB | Best SSIM: {best_metrics['ssim']:.4f}")
        
        # Visualize samples periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_images, blurred_images, enhanced_images, metrics = sample_and_evaluate(model, val_loader, device)
            
            for i, (val_img, blur_img, enh_img, (psnr, ssim)) in enumerate(zip(val_images, blurred_images, enhanced_images, metrics)):
                # Convert to display format
                orig = (val_img[0].clamp(-1, 1) + 1) / 2
                blur = (blur_img[0].clamp(-1, 1) + 1) / 2
                enhanced = (enh_img[0].clamp(-1, 1) + 1) / 2
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(orig.permute(1, 2, 0).cpu().numpy())
                plt.title("Original Dog")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(blur.permute(1, 2, 0).cpu().numpy())
                plt.title("Blurred Dog")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(enhanced.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Enhanced Dog (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"enhanced_samples/dogs_epoch_{epoch+1}_sample_{i+1}.png", dpi=150)
                plt.close()

    # ======== FINAL MODEL EVALUATION ========
    # Load best model
    model.load_state_dict(torch.load("best_deblur_model_dogs.pth"))
    model.eval()

    # Final evaluation with varying levels of blur
    print("Performing final quality evaluation on Stanford Dogs...")
    
    blur_levels = [0.3, 0.6, 0.9]  # Light, medium, heavy blur
    results = {level: {'psnr': 0, 'ssim': 0, 'count': 0} for level in blur_levels}
    
    with torch.no_grad():
        for val_x, _ in tqdm(val_loader, desc="Final evaluation"):
            val_x = val_x.to(device)
            
            # Test each blur level
            for blur_level in blur_levels:
                blurred = apply_blur(val_x, severity=blur_level)
                output = model(blurred)
                
                # Calculate metrics for each image
                for i in range(val_x.size(0)):
                    psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                    
                    results[blur_level]['psnr'] += psnr
                    results[blur_level]['ssim'] += ssim
                    results[blur_level]['count'] += 1
                    
                    # Save some examples for each blur level
                    if results[blur_level]['count'] <= 5:
                        # Convert to display format
                        orig = (val_x[i].clamp(-1, 1) + 1) / 2
                        blur = (blurred[i].clamp(-1, 1) + 1) / 2
                        enhanced = (output[i].clamp(-1, 1) + 1) / 2
                        
                        # Create visualization
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(orig.permute(1, 2, 0).cpu().numpy())
                        plt.title("Original Dog")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(blur.permute(1, 2, 0).cpu().numpy())
                        plt.title(f"Blurred Dog (Level: {blur_level:.1f})")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(enhanced.permute(1, 2, 0).cpu().numpy())
                        plt.title(f"Enhanced Dog (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(f"enhanced_samples/dogs_final_blur_{blur_level:.1f}_sample_{results[blur_level]['count']}.png", dpi=150)
                        plt.close()
                        
                # Stop after processing a reasonable number of images
                if results[blur_level]['count'] >= 100:
                    break
            
            # Check if we've processed enough images for all blur levels
            if all(results[level]['count'] >= 100 for level in blur_levels):
                break

    # Print final results
    print("\n===== FINAL RESULTS ON STANFORD DOGS =====")
    for level in blur_levels:
        avg_psnr = results[level]['psnr'] / results[level]['count']
        avg_ssim = results[level]['ssim'] / results[level]['count']
        print(f"Blur Level {level:.1f}: PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")

    # ======== DEMONSTRATION FUNCTIONS ========
    def batch_deblur_dogs(input_dir, output_dir=None):
        """Batch process dog images for deblurring"""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "enhanced_dogs")
        os.makedirs(output_dir, exist_ok=True)

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        print(f"Found {len(image_files)} dog images to process")

        for img_path in tqdm(image_files, desc="Batch processing dog images"):
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_enhanced.png")
            
            try:
                deblur_image(img_path, model, device, output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Batch processing complete. Enhanced dog images saved to {output_dir}")

    # ======== SAVE MODEL AND CREATE DEMO ========
    print("\nSaving final model for Stanford Dogs...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'img_size': img_size,
        'architecture': 'ImprovedEnhancedUNet',
        'dataset': 'Stanford Dogs',
        'date_trained': 'July 2025'
    }, "deblur_model_dogs_complete.pth")
    
    # Save smaller model for deployment
    torch.save(model.state_dict(), "deblur_model_dogs_state.pth")

    # Generate final demonstration with best dog examples
    print("\nGenerating final demonstration with dog images...")
    
    # Select best examples from dogs dataset
    best_examples = []
    best_metrics = []
    
    with torch.no_grad():
        for val_x, _ in val_loader:
            if len(best_examples) >= 8:  # More examples for dogs
                break
                
            val_x = val_x.to(device)
            # Apply strong blur for impressive demo
            blurred = apply_blur(val_x, severity=0.8)
            output = model(blurred)
            
            # Select images with good metric improvements
            for i in range(val_x.size(0)):
                if len(best_examples) >= 8:
                    break
                    
                psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                psnr_blurred, ssim_blurred = calculate_metrics(blurred[i:i+1], val_x[i:i+1])
                
                # Calculate improvement
                psnr_improvement = psnr - psnr_blurred
                ssim_improvement = ssim - ssim_blurred
                
                if psnr_improvement > 5.0 and ssim_improvement > 0.1:
                    best_examples.append((val_x[i:i+1], blurred[i:i+1], output[i:i+1]))
                    best_metrics.append((psnr, ssim, psnr_improvement, ssim_improvement))

    # Create detailed visualization grid for dogs
    for idx, ((orig, blur, enhanced), (psnr, ssim, psnr_imp, ssim_imp)) in enumerate(zip(best_examples, best_metrics)):
        # Convert tensors for display
        orig_img = (orig[0].clamp(-1, 1) + 1) / 2
        blur_img = (blur[0].clamp(-1, 1) + 1) / 2
        enhanced_img = (enhanced[0].clamp(-1, 1) + 1) / 2
        
        # Create advanced visualization
        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 3, 1)
        plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        plt.title("Original Dog Image", fontsize=14)
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(blur_img.permute(1, 2, 0).cpu().numpy())
        plt.title("Blurred Dog Image", fontsize=14)
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(enhanced_img.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Enhanced Dog Image\nPSNR: {psnr:.2f}dB (+{psnr_imp:.2f}dB)\nSSIM: {ssim:.4f} (+{ssim_imp:.4f})", fontsize=14)
        plt.axis('off')
        
        # Zoom in on a region of interest (center crop for detail)
        h, w = orig_img.shape[1], orig_img.shape[2]
        crop_h, crop_w = h // 2, w // 2
        start_h, start_w = h // 4, w // 4
        
        # Get zoom crops
        orig_crop = orig_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        blur_crop = blur_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        enhanced_crop = enhanced_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        plt.subplot(2, 3, 4)
        plt.imshow(orig_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Original (Detail View)", fontsize=12)
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(blur_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Blurred (Detail View)", fontsize=12)
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(enhanced_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Enhanced (Detail View)", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"enhanced_samples/dogs_final_demo_{idx+1}_detailed.png", dpi=200, bbox_inches='tight')
        plt.close()

    # Create a summary grid showing multiple dog examples
    plt.figure(figsize=(20, 15))
    for idx, ((orig, blur, enhanced), _) in enumerate(zip(best_examples[:6], best_metrics[:6])):
        if idx >= 6:
            break
            
        orig_img = (orig[0].clamp(-1, 1) + 1) / 2
        blur_img = (blur[0].clamp(-1, 1) + 1) / 2
        enhanced_img = (enhanced[0].clamp(-1, 1) + 1) / 2
        
        # Original
        plt.subplot(6, 3, idx*3 + 1)
        plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        if idx == 0:
            plt.title("Original Dogs", fontsize=16)
        plt.axis('off')
        
        # Blurred
        plt.subplot(6, 3, idx*3 + 2)
        plt.imshow(blur_img.permute(1, 2, 0).cpu().numpy())
        if idx == 0:
            plt.title("Blurred Dogs", fontsize=16)
        plt.axis('off')
        
        # Enhanced
        plt.subplot(6, 3, idx*3 + 3)
        plt.imshow(enhanced_img.permute(1, 2, 0).cpu().numpy())
        if idx == 0:
            plt.title("Enhanced Dogs", fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("enhanced_samples/dogs_summary_grid.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Training complete! Stanford Dogs deblurring examples saved to 'enhanced_samples' directory.")
    print(f"Model saved as 'deblur_model_dogs_complete.pth'")
    print("Use deblur_image('path/to/dog_image.jpg', model, device) to enhance individual dog images")
    print("Use batch_deblur_dogs('path/to/dog_folder') to process multiple dog images")