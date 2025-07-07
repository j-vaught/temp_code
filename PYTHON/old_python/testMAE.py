# Install any extra dependencies (only if not already present):
# !pip install --quiet xarray netcdf4 torch torchvision matplotlib

import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

# ---------- Training Scaffold (don’t run on this machine) ----------
if __name__ == '__main__':
    # Hyperparameters
    file_path     = 'radar_all_sweeps.nc'
    batch_size    = 8
    epochs        = 5
    patch_size    = 32   # adapt to your angular/range resolution
    mask_ratio    = 0.2
    learning_rate = 5e-4

    # Dataset + Loader
    dataset = RadarNetCDFDataset(file_path, variable='echo_strength')
    loader  = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=6,
                         pin_memory=True)
    print(f"[DataLoader] Created. Batches per epoch: {len(loader)}")

    # Model, optimizer, loss
    model     = ConvMAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---- LOAD 16-PATCH PRETRAIN ----
    # (skip this block if you really do want training from scratch)
    state = torch.load('convmae_radar.pth', map_location=device)
    model.load_state_dict(state)
    print('[Init] loaded 16×16-patch weights. Starting fine-tune.')

    # Training loop
    for ep in range(1, epochs+1):
        print(f"\n========== Starting Epoch {ep}/{epochs} ==========")
        running_loss = 0.0

        for batch_idx, x in enumerate(loader, 1):
            if batch_idx == 1:
                print(f"[Epoch {ep}] First batch raw x.shape: {x.shape}")
            x = x.to(device)

            x_mask = mask_patches(x,
                                  patch_size=patch_size,
                                  mask_ratio=mask_ratio)
            if batch_idx == 1:
                zeros = int((x_mask == 0).sum().item())
                total = x.numel()
                print(f"[Epoch {ep}] First batch masked: {zeros}/{total} pixels → "
                      f"{zeros/total:.3%}")

            rec = model(x_mask)

            if batch_idx == 1:
                print(f"[Epoch {ep}] First batch reconstruction stats: "
                      f"min={rec.min().item():.3f}, max={rec.max().item():.3f}")

            # compute masked-pixels-only, weighted MSE
            pix_mask = (x_mask == 0)
            w = torch.where(x > 0.5, 10.0, 1.0).to(device)
            loss = ((rec - x)**2 * w)[pix_mask].mean()

            # every 10 batches, print loss
            if batch_idx % 10 == 0:
                print(f"[Epoch {ep}] Batch {batch_idx}/{len(loader)} — loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"[Epoch {ep}] Finished. Average Loss: {avg_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), 'convmae_radar.pth')
    print("Model weights saved to 'convmae_radar.pth'")
