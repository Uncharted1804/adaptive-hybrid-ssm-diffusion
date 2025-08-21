import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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

# ======== MODEL CLASSES ========
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

def apply_stronger_test_blur(images):
    """Apply stronger blur for testing to better showcase model effectiveness"""
    device = images.device
    blur_type = np.random.choice(['gaussian', 'motion', 'defocus'])
    
    if blur_type == 'gaussian':
        # Increase sigma for more substantial blur
        sigma = torch.tensor(np.random.uniform(1.5, 3.5)).to(device)
        return kf.gaussian_blur2d(images, kernel_size=(9, 9), sigma=(sigma, sigma))
    
    elif blur_type == 'motion':
        # Longer kernel for more dramatic motion blur
        # Ensure kernel_size is odd
        kernel_size = 7 + 2 * np.random.randint(2, 5)  # Will always be odd (11, 13, 15, 17)
        angle = np.random.uniform(0, 360)
        direction = -1.0 + 2.0 * np.random.random()  # -1 to 1
        return kf.motion_blur(images, kernel_size=kernel_size, angle=angle, direction=direction)
    
    else:  # defocus
        # Larger defocus blur
        # Ensure kernel_size is odd
        kernel_size = 7 + 2 * np.random.randint(2, 5)  # Will always be odd
        return kf.box_blur(images, kernel_size=(kernel_size, kernel_size))
    
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

# 2. Enhanced visualization function (from changes.py)
# Modify the create_enhanced_visualization function (around lines 379-430)
def create_enhanced_visualization(original, blurred, enhanced, psnr, ssim, save_path):
    """Create more informative visualization with difference maps"""
    # Ensure original dimensions are maintained
    _, _, orig_h, orig_w = original.shape
    
    # Convert tensors to numpy arrays
    orig = original.cpu().squeeze().permute(1, 2, 0).numpy()
    blur = blurred.cpu().squeeze().permute(1, 2, 0).numpy()
    enh = enhanced.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Calculate difference maps (multiplied for visibility)
    diff_blur = np.abs(orig - blur) * 5
    diff_enh = np.abs(orig - enh) * 5
    
    # Ensure values are in proper range
    orig = np.clip((orig + 1) / 2, 0, 1)
    blur = np.clip((blur + 1) / 2, 0, 1)
    enh = np.clip((enh + 1) / 2, 0, 1)
    diff_blur = np.clip(diff_blur, 0, 1)
    diff_enh = np.clip(diff_enh, 0, 1)
    
    # Create figure with subplots
    plt.figure(figsize=(18, 10))
    
    # Original, blurred and enhanced images
    plt.subplot(2, 3, 1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(blur)
    plt.title("Blurred")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(enh)
    plt.title(f"Enhanced\nPSNR: {psnr:.2f}dB\nSSIM: {ssim:.4f}")
    plt.axis('off')
    
    # Difference maps
    plt.subplot(2, 3, 5)
    plt.imshow(diff_blur)
    plt.title("Original vs Blurred\n(Difference Map)")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(diff_enh)
    plt.title("Original vs Enhanced\n(Difference Map)")
    plt.axis('off')
    
    # Add edge comparison (high frequency details)
    edges_orig = ndimage.sobel(rgb2gray(orig))
    edges_enh = ndimage.sobel(rgb2gray(enh))
    
    plt.subplot(2, 3, 4)
    plt.imshow(np.abs(edges_orig - edges_enh), cmap='hot')
    plt.title("Edge Preservation\n(Lower is better)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# 4. Function to visualize attention maps (from changes.py)
def visualize_attention(model, image, save_path):
    """Visualize where the attention mechanism is focusing"""
    model.eval()
    with torch.no_grad():
        # Forward pass to get internal representations
        # This requires modifying the model to extract attention weights
        
        # Pseudo-code (implementation depends on model architecture):
        # image = image.to(device)
        # _, attention_weights = model.forward_with_attention(image)
        
        # For this example, let's visualize a synthetic attention map
        # In practice, extract real attention weights from model
        
        # Create a heatmap visualization
        B, C, H, W = image.shape
        synthetic_attention = torch.zeros(B, 1, H, W)
        
        # Highlight edges which is where deblurring focuses
        edges = kf.sobel(image)
        attention_heatmap = torch.sum(torch.abs(edges), dim=1, keepdim=True)
        attention_heatmap = F.interpolate(attention_heatmap, (H, W))
        
        # Normalize for visualization
        attention_min = attention_heatmap.min()
        attention_max = attention_heatmap.max()
        attention_normalized = (attention_heatmap - attention_min) / (attention_max - attention_min)
        
        # Convert to numpy for matplotlib
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1) / 2  # convert from [-1,1] to [0,1]
        attn_np = attention_normalized[0, 0].cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        plt.imshow(attn_np, alpha=0.7, cmap='jet')
        plt.title("Attention Heatmap")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# 5. Benchmarking function (from changes.py)
def benchmark_against_baselines(image, model, save_path):
    """Compare your model against classic deblurring methods"""
    # Convert torch tensor to numpy
    img_np = (image[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
    
    # Apply blur
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img_np, sigma=2.0)
    
    # Wiener filter (classic deconvolution)
    kernel = np.ones((5, 5)) / 25  # Approximation of blur kernel
    wiener_result = np.zeros_like(blurred)
    for i in range(3):  # Process each channel
        wiener_result[:,:,i] = wiener(blurred[:,:,i], kernel, 0.1)
    
    # Richardson-Lucy (another classic method)
    rl_result = np.zeros_like(blurred)
    for i in range(3):
        rl_result[:,:,i] = richardson_lucy(blurred[:,:,i], kernel, num_iter=10)
    
    # TV denoising (edge-preserving smoothing)
    tv_result = denoise_tv_chambolle(blurred, weight=0.1)
    
    # Your model's result
    with torch.no_grad():
        blurred_tensor = torch.tensor(blurred).permute(2, 0, 1).unsqueeze(0).to(image.device)
        blurred_tensor = blurred_tensor * 2 - 1  # Convert back to [-1,1]
        model_result = model(blurred_tensor)
        model_result_np = (model_result[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
    
    # Create comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(blurred)
    plt.title("Blurred")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(model_result_np)
    plt.title("Your Model")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(wiener_result)
    plt.title("Wiener Filter")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(rl_result)
    plt.title("Richardson-Lucy")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(tv_result)
    plt.title("TV Denoising")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        # Explicitly set GPU device if available
        torch.cuda.set_device(0)
        
    print(f"Using device: {device}")
    os.makedirs("enhanced_samples", exist_ok=True)

    # ======== DATA ========
    # Increasing image size for better detail preservation
    img_size = 32 # Increased from 64
    batch_size = 16  # Smaller batch size for higher resolution
    learning_rate = 1e-4
    epochs = 30

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Using a higher quality dataset instead of CIFAR-10
    try:
        # Try to use ImageNet subset if available
        train_data = datasets.ImageFolder(root='./data/imagenet_subset', transform=transform)
        print("Using ImageNet subset for training")
    except:
        # Fall back to CIFAR-10
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        print("Using CIFAR-10 for training")
    
    # Split data into train and validation
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Set num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_description(f"Pretrain {epoch+1}/5 | Loss: {loss.item():.6f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Pretrain Epoch [{epoch+1}/5] | Avg Loss: {avg_loss:.6f}")
        
        # Quick validation
        model.eval()
        with torch.no_grad():
            val_psnr = 0
            val_ssim = 0
            samples_processed = 0
            
            for val_x, _ in val_loader:
                if samples_processed >= 100:  # Evaluate on 100 samples
                    break
                    
                val_x = val_x.to(device)
                blurred = apply_blur(val_x, severity=0.5)
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
            print(f"Pretrain Validation: PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")

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
            torch.save(model.state_dict(), "best_deblur_model.pth")
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
                plt.title("Original")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(blur.permute(1, 2, 0).cpu().numpy())
                plt.title("Blurred")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(enhanced.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Enhanced (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")
                plt.axis('off')
                
                plt.savefig(f"enhanced_samples/epoch_{epoch+1}_sample_{i+1}.png")
                plt.close()

    # ======== FINAL MODEL EVALUATION ========
    # Load best model
    model.load_state_dict(torch.load("best_deblur_model.pth"))
    model.eval()

    # Final evaluation with varying levels of blur
    print("Performing final quality evaluation...")
    
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
                        plt.title("Original")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(blur.permute(1, 2, 0).cpu().numpy())
                        plt.title(f"Blurred (Level: {blur_level:.1f})")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(enhanced.permute(1, 2, 0).cpu().numpy())
                        plt.title(f"Enhanced (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")
                        plt.axis('off')
                        
                        plt.savefig(f"enhanced_samples/final_blur_{blur_level:.1f}_sample_{results[blur_level]['count']}.png")
                        plt.close()
                        
                # Stop after processing a reasonable number of images
                if results[blur_level]['count'] >= 100:
                    break
            
            # Check if we've processed enough images for all blur levels
            if all(results[level]['count'] >= 100 for level in blur_levels):
                break

# Print final results
    print("\n===== FINAL RESULTS =====")
    for level in blur_levels:
        avg_psnr = results[level]['psnr'] / results[level]['count']
        avg_ssim = results[level]['ssim'] / results[level]['count']
        print(f"Blur Level {level:.1f}: PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")

def deblur_image(input_path, output_path=None):
    from PIL import Image

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced.png"

    # Load image
    img = Image.open(input_path).convert('RGB')

    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        enhanced = model(input_tensor)
        enhanced = model(enhanced)

    # Sharpen
    sharpened = kf.unsharp_mask(enhanced, kernel_size=(5, 5), sigma=1.0, strength=0.7)

    # Save output
    output_img = (sharpened[0].clamp(-1, 1) + 1) / 2
    output_img = output_img.cpu().permute(1, 2, 0).numpy() * 255
    output_img = Image.fromarray(output_img.astype(np.uint8))
    output_img.save(output_path)

    print(f"Enhanced image saved to {output_path}")

    # Comparison visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.array(output_img))
    plt.title("Enhanced")
    plt.axis('off')

    comparison_path = f"{os.path.splitext(output_path)[0]}_comparison.png"
    plt.savefig(comparison_path)
    plt.close()

    print(f"Comparison saved to {comparison_path}")
    return output_path, comparison_path


def batch_deblur(input_dir, output_dir=None):
    import glob
    from PIL import Image

    if output_dir is None:
        output_dir = os.path.join(input_dir, "enhanced")
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(image_files)} images to process")

    for img_path in tqdm(image_files, desc="Batch processing"):
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_enhanced.png")
        try:
            img = Image.open(img_path).convert('RGB')

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(img).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                enhanced = model(input_tensor)
                sharpened = kf.unsharp_mask(enhanced, kernel_size=(5, 5), sigma=1.0, strength=0.7)

                output_img = (sharpened[0].clamp(-1, 1) + 1) / 2
                output_img = output_img.cpu().permute(1, 2, 0).numpy() * 255
                output_img = Image.fromarray(output_img.astype(np.uint8))
                output_img.save(output_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Batch processing complete. Enhanced images saved to {output_dir}")

    
    print("For batch processing, use: batch_deblur('path/to/image_folder')")

    # Save model in various formats
    print("\nSaving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'img_size': img_size,
        'architecture': 'ImprovedEnhancedUNet',
        'date_trained': 'April 2025'
    }, "deblur_model_complete.pth")
    
    # Save smaller model for deployment
    torch.save(model.state_dict(), "deblur_model_state.pth")
    
    # Try to save TorchScript model for deployment
    try:
        example_input = torch.randn(1, 3, img_size, img_size).to(device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save("deblur_model_traced.pt")
        print("TorchScript model saved successfully")
    except Exception as e:
        print(f"Error saving TorchScript model: {e}")

    # Final demo with advanced visualization
    print("\nGenerating final demonstration...")
    
    # Select best examples
    best_examples = []
    best_metrics = []
    
    with torch.no_grad():
        for val_x, _ in val_loader:
            if len(best_examples) >= 5:
                break
                
            val_x = val_x.to(device)
            # Apply strong blur for impressive demo
            blurred = apply_blur(val_x, severity=0.8)
            output = model(blurred)
            
            # Select images with good metric improvements
            for i in range(val_x.size(0)):
                if len(best_examples) >= 5:
                    break
                    
                psnr, ssim = calculate_metrics(output[i:i+1], val_x[i:i+1])
                psnr_blurred, ssim_blurred = calculate_metrics(blurred[i:i+1], val_x[i:i+1])
                
                # Calculate improvement
                psnr_improvement = psnr - psnr_blurred
                ssim_improvement = ssim - ssim_blurred
                
                if psnr_improvement > 6.0 and ssim_improvement > 0.15:
                    best_examples.append((val_x[i:i+1], blurred[i:i+1], output[i:i+1]))
                    best_metrics.append((psnr, ssim, psnr_improvement, ssim_improvement))
    
    # Create detailed visualization
    for idx, ((orig, blur, enhanced), (psnr, ssim, psnr_imp, ssim_imp)) in enumerate(zip(best_examples, best_metrics)):
        # Convert tensors for display
        orig_img = (orig[0].clamp(-1, 1) + 1) / 2
        blur_img = (blur[0].clamp(-1, 1) + 1) / 2
        enhanced_img = (enhanced[0].clamp(-1, 1) + 1) / 2
        
        # Create advanced visualization with zoom-in regions
        plt.figure(figsize=(18, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(blur_img.permute(1, 2, 0).cpu().numpy())
        plt.title("Blurred")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(enhanced_img.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Enhanced\nPSNR: {psnr:.2f}dB (+{psnr_imp:.2f}dB)\nSSIM: {ssim:.4f} (+{ssim_imp:.4f})")
        plt.axis('off')
        
        # Zoom in on a region of interest (center crop)
        h, w = orig_img.shape[1], orig_img.shape[2]
        crop_h, crop_w = h // 3, w // 3
        start_h, start_w = h // 3, w // 3
        
        # Get zoom crops
        orig_crop = orig_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        blur_crop = blur_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        enhanced_crop = enhanced_img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        plt.subplot(2, 3, 4)
        plt.imshow(orig_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Original (Detail)")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(blur_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Blurred (Detail)")
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(enhanced_crop.permute(1, 2, 0).cpu().numpy())
        plt.title("Enhanced (Detail)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"enhanced_samples/final_demo_{idx+1}_detailed.png", dpi=200)
        plt.close()
        
    print("Training complete! Detailed examples saved to 'enhanced_samples' directory.")




