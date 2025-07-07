import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Re-use exactly the same classes you defined before:
# - RadarNetCDFDataset
# - mask_patches
# - ConvMAE

# 1) Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- Real Radar Dataset from NetCDF ----------
class RadarNetCDFDataset(Dataset):
    def __init__(self,
                 file_path,
                 variable='echo_strength',
                 normalize=True):
        """
        Args:
            file_path (str): Path to .nc file.
            variable (str): Name of the radar field.
            normalize (bool): If True, scale raw counts to [0,1].
        """
        print(f"[Dataset] Loading dataset from {file_path}, variable = '{variable}'")
        # Lazy-load with xarray
        self.ds = xr.open_dataset(file_path)
        self.data = self.ds[variable]
        self.normalize = normalize
        # Number of sweeps = length along 'time'
        self.num_samples = self.data.sizes['time']
        print(f"[Dataset] Loaded. Number of samples (time steps): {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Extract a single sweep: shape (angle, range)
        arr = self.data.isel(time=idx).values.astype('float32')
        if self.normalize:
            # Raw counts are 4-bit (0–15); scale to [0,1]
            arr = arr / 15.0
        x = torch.from_numpy(arr).unsqueeze(0)  # shape [1, H(angle), W(range)]
        if idx < 3:  # only print first few to avoid flooding
            print(f"[Dataset] __getitem__ idx={idx} → tensor shape {x.shape}, "
                  f"min={x.min().item():.3f}, max={x.max().item():.3f}")
        return x

# ---------- Patch Masking (with debug prints) ----------
def mask_patches(x, patch_size=16, mask_ratio=0.2):
    B, C, H, W = x.shape
    print(f"[mask_patches] Input x.shape={x.shape}, patch_size={patch_size}, mask_ratio={mask_ratio}")
    out = x.clone()
    nH, nW = H // patch_size, W // patch_size
    total = nH * nW
    mask = torch.rand(B, total, device=x.device) < mask_ratio

    masked_count = mask.sum().item()
    print(f"[mask_patches] Total patches per image={total}, "
          f"Total masked this batch={masked_count}/{B*total}")

    for b in range(B):
        idx = 0
        for i in range(nH):
            for j in range(nW):
                if mask[b, idx]:
                    out[b, :, i*patch_size:(i+1)*patch_size,
                           j*patch_size:(j+1)*patch_size] = 0
                idx += 1
    return out

# ---------- ConvMAE Model (with forward debug) ----------
class ConvMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        print(f"[Model] Encoded to z.shape = {z.shape}")
        out = self.decoder(z)
        print(f"[Model] Decoded to out.shape = {out.shape}")
        return out

# 2) Reload model architecture and weights
model = ConvMAE().to(device)
state_dict = torch.load('convmae_radar.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded and set to eval() mode.")

# 3) Create a small test loader (here we'll just reuse your Dataset but point to e.g. a 'test' file
#    or the same file if you didn’t split data).
test_ds = RadarNetCDFDataset('radar_all_sweeps.nc',
                             variable='echo_strength',
                             normalize=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=True)

# 4) Grab one batch, run inference, and plot
with torch.no_grad():
    # take first batch
    x = next(iter(test_loader)).to(device)       # [B,1,H,W]
    x_mask = mask_patches(x, patch_size=4, mask_ratio=0.7)
    recon = model(x_mask)                        # [B,1,H,W]

# 5) Visualize the first sample in the batch
orig = x[0,0].cpu().numpy()
masked = x_mask[0,0].cpu().numpy()
pred  = recon[0,0].cpu().numpy()

fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(orig, cmap='viridis')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(masked, cmap='viridis')
axes[1].set_title('Masked Input')
axes[1].axis('off')

axes[2].imshow(pred, cmap='viridis')
axes[2].set_title('Reconstruction')
axes[2].axis('off')

plt.tight_layout()
plt.show()



# … everything through your current step 5 …
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt


H, W = orig.shape

# 1) build angle (0–360°) and range axes
angles = np.deg2rad(np.linspace(0, 360, H, endpoint=False))
ranges = np.arange(W)
theta, r = np.meshgrid(angles, ranges, indexing='ij')

# 2) set up three polar subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4),
                        subplot_kw={'projection': 'polar'})
titles = ['Original', 'Masked Input', 'Reconstruction']
data   = [orig, masked, pred]

for ax, title, d in zip(axs, titles, data):
    # zero at top, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, W)
    pcm = ax.pcolormesh(theta, r, d, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')    # turn off grid/labels for cleaner look

plt.tight_layout()
plt.show()