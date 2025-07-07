# main.py
import random, torch, torch.nn as nn, timm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim as ssim_fn
from tqdm import tqdm
import torch.nn.functional as F


# ----------------- GLOBAL CONFIG ---------------------------
ROOT_X = "/Users/jacobvaught/Downloads/frames_parallel_9g/03/low"
ROOT_Y = "/Users/jacobvaught/Downloads/frames_parallel_9g/03/mid"
ROOT_Z = "/Users/jacobvaught/Downloads/frames_parallel_9g/03/high"
IMG_SIZE = 224
BATCH_SIZE = 16
MASK_RATIO = 0.65
EPOCHS = 50
VISUALIZE = True
OUT_DIR = "results"
# -----------------------------------------------------------

# ---------- 1. DATASET --------------------------------------------------------
def load_image_black_bg(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        # Black, fully opaque
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    else:
        img = img.convert('RGB')
    return img
def psnr(x, y):
    mse = F.mse_loss(x, y)
    return 10 * torch.log10(1.0 / mse)

def ssim_metric(x, y):
    # expects x, y in [0,1], (B,C,H,W)
    return ssim_fn(x, y, data_range=1.0)


class TrioDataset(Dataset):
    def __init__(self, root_x, root_y, root_z, img_size=224):
        self.fx = sorted(Path(root_x).glob('*.png'))
        self.fy = sorted(Path(root_y).glob('*.png'))
        self.fz = sorted(Path(root_z).glob('*.png'))
        assert len({len(self.fx), len(self.fy), len(self.fz)}) == 1, "Folder lengths differ"
        self.t = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])



    def __len__(self): return len(self.fx)

    def __getitem__(self, idx):
        im1 = self.t(load_image_black_bg(self.fx[idx]))
        im2 = self.t(load_image_black_bg(self.fy[idx]))
        im3 = self.t(load_image_black_bg(self.fz[idx]))
        trio = torch.cat([im1, im2, im3], dim=0)   # shape (9,H,W)
        return trio

# ---------- 2. SIMPLE MAE MODULE ---------------------------------------------
class SimpleMAE(nn.Module):
    def __init__(self, img_size=224, patch=16, mask_ratio=0.5):
        super().__init__()
        self.patch = patch
        self.mask_ratio = mask_ratio
        # tiny ViT encoder, no pretrained weights
        self.encoder = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            img_size=img_size,
            in_chans=9,      # 3 frames × 3 channels
            num_classes=0    # we only need features
        )
        embed_dim = self.encoder.embed_dim
        self.decoder = nn.Linear(embed_dim, patch*patch*9)  # reconstruct 9-channel patch

    def _patchify(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch
        assert H % p == 0 and W % p == 0
        imgs = imgs.reshape(B, C, H//p, p, W//p, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1).flatten(1, 2)  # (B, N, p,p,C)
        return imgs.reshape(B, -1, p*p*C)                    # (B, Npatch, patch_dim)
    
    def _unpatchify(self, patches, H, W):
        """Inverse of _patchify: (B, N, p*p*C) → (B, C, H, W)."""
        B, N, D = patches.shape
        p = self.patch
        C = D // (p * p)
        h, w = H // p, W // p
        patches = patches.reshape(B, h, w, p, p, C)          # (B,h,w,p,p,C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)          # (B,C,h,p,w,p)
        return patches.reshape(B, C, H, W)


    def _make_image_mask(self, imgs):
        """Return a binary mask the same H×W as the image, with `mask_ratio`
        of the *patches* set to 1 (masked)."""
        B, _, H, W = imgs.shape
        p = self.patch
        n_h, n_w = H // p, W // p
        N = n_h * n_w
        keep = int(N * (1 - self.mask_ratio))
        ids = torch.stack([torch.randperm(N, device=imgs.device)
                           for _ in range(B)], dim=0)          # 👈 on-device
        mask = torch.ones((B, N), device=imgs.device, dtype=torch.bool)
        mask.scatter_(1, ids[:, :keep], False)                  # False = keep
        # blow mask up from patch grid → pixel grid
        mask = mask.view(B, n_h, n_w).\
               repeat_interleave(p, 1).repeat_interleave(p, 2)
        return mask.unsqueeze(1)                                # shape (B,1,H,W)

    def forward(self, imgs, return_recon=False):
        """If return_recon=True, also return masked-input and reconstruction."""
        target_patches = self._patchify(imgs)

        pix_mask = self._make_image_mask(imgs).expand(-1, imgs.shape[1], -1, -1)
        imgs_masked = imgs.clone()
        imgs_masked[pix_mask] = 0

        # --- encoder --------------------------------------------------------
        x = self.encoder.patch_embed(imgs_masked)
        cls = self.encoder.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        for blk in self.encoder.blocks:
            x = blk(x)
        tokens = self.encoder.norm(x)[:, 1:]            # (B, N, 384)
        # --------------------------------------------------------------------

        recon_patches = self.decoder(tokens)            # (B, N, p*p*C)
        loss = ((recon_patches - target_patches) ** 2).mean()

        if return_recon:
            B, _, H, W = imgs.shape
            recon_img = self._unpatchify(recon_patches, H, W)
            return loss, imgs_masked, recon_img
        return loss


writer = SummaryWriter(log_dir="runs/mae")


# ------------------ TRAIN LOOP -----------------------------
def train():
    ds = TrioDataset(ROOT_X, ROOT_Y, ROOT_Z, img_size=IMG_SIZE)
    pin = torch.cuda.is_available()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=pin)
    device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
    model = SimpleMAE(img_size=IMG_SIZE, mask_ratio=MASK_RATIO).to(device)
    print(f"Using device: {device}")
    print(f"Dataset size: {len(ds)} images, batch size: {BATCH_SIZE}, mask ratio: {MASK_RATIO}")
    print(f"Model: {model.encoder.num_features} features, {model.patch}x{model.patch} patches")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:
            batch = batch.to(device)
            loss, imgs_masked, recon = model(batch, return_recon=True)
            opt.zero_grad(); loss.backward(); opt.step()

            # --- Metrics (CPU for logging) ---
            recon = recon.clamp(0,1)
            batch = batch.clamp(0,1)
            mse = F.mse_loss(recon, batch)
            psnr_val = psnr(recon, batch)
            ssim_val = ssim_metric(recon, batch)

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/psnr", psnr_val, global_step)
            writer.add_scalar("train/ssim", ssim_val, global_step)
            writer.add_scalar("train/lr", opt.param_groups[0]['lr'], global_step)

            if step % 25 == 0:
                # Visual grid logging
                strips = []
                for clip in (batch[0], imgs_masked[0], recon[0]):
                    frames = [clip[c*3:(c+1)*3] for c in range(3)]
                    strips.append(make_grid(frames, nrow=3, normalize=True))
                panel = make_grid(strips, nrow=1)
                writer.add_image("train/recon_panel", panel, global_step)
            
            pbar.set_description(f"Ep{epoch} step{step} loss:{loss.item():.4f} psnr:{psnr_val:.2f} ssim:{ssim_val:.3f}")
            global_step += 1


        # -------------- NEW: lightweight visuals -------------
        if VISUALIZE:
            show_triplet(model, batch[0].cpu(), OUT_DIR)
            plot_decoder_filters(model, OUT_DIR)
            embed_tsne(model, loader, OUT_DIR)
            # (Optionally log these images to TensorBoard as well)
            writer.add_image("train/decoder_filters", plt.imread(f"{OUT_DIR}/decoder_filters.png"), epoch, dataformats='HWC')
            writer.add_image("train/tsne", plt.imread(f"{OUT_DIR}/tsne.png"), epoch, dataformats='HWC')

        # ------------------------------------------------------

# ---------- 4. VISUALISATION HELPERS -----------------------------------------
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
import numpy as np

@torch.no_grad()
def show_triplet(model, trio, out_dir):
    """Save one PNG: original | masked | reconstruction (each strip = 3 frames)."""
    model.eval()
    device = next(model.parameters()).device
    trio = trio.unsqueeze(0).to(device)                 # (1,9,H,W)

    loss, imgs_masked, recon = model(trio, return_recon=True)
    trio, imgs_masked, recon = trio[0], imgs_masked[0], recon[0]  # drop batch

    def split_frames(x):                                # (9,H,W) → three (3,H,W)
        return [x[c*3:(c+1)*3] for c in range(3)]

    strips = []
    for clip in (trio, imgs_masked, recon):
        frames = split_frames(clip.cpu())
        strips.append(make_grid(frames, nrow=3, normalize=True))

    grid = make_grid(strips, nrow=1)                    # stack horizontally
    out_path = Path(out_dir) / "reconstruction.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, grid.permute(1, 2, 0).numpy())
    print(f"✔ reconstruction.png written to {out_path}")


def plot_decoder_filters(model, out_dir):
    """
    Save a mosaic of the decoder’s learned filters.

    * 384 tiles (one per embedding dimension)
    * Each tile is the 9-channel patch averaged to grayscale
    * Handles Matplotlib’s 2-D/3-D array rules safely
    """
    W = model.decoder.weight.detach().cpu()                 # (p²*9, 384)
    W = W.T.reshape(384, 9, model.patch, model.patch)       # (384,9,p,p)
    W = W.mean(1, keepdim=True)                             # → (384,1,p,p)

    grid = make_grid(W, nrow=16, normalize=True)      # (3, H, W) every time
    img  = grid.permute(1, 2, 0).numpy()              # → (H, W, 3)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decoder_filters.png"
    plt.imsave(path, img)                             # RGB image → no cmap needed
    print(f"✔ decoder-filter grid saved to {path}")




@torch.no_grad()
def embed_tsne(model, loader, out_dir, max_batches=16):
    model.eval(); device = next(model.parameters()).device
    feats = []
    for i,(batch) in enumerate(loader):
        if i==max_batches: break
        batch = batch.to(device)
        x = model.encoder.patch_embed(batch)
        cls = model.encoder.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = model.encoder.pos_drop(x + model.encoder.pos_embed)
        for blk in model.encoder.blocks: x = blk(x)
        x = model.encoder.norm(x)[:,1:].mean(1)   # average over patches
        feats.append(x.cpu())
    feats = torch.cat(feats).numpy()
    emb2d = TSNE(n_components=2, perplexity=30).fit_transform(feats)
    plt.figure(figsize=(6,6)); plt.scatter(emb2d[:,0], emb2d[:,1], s=8)
    path = Path(out_dir)/"tsne.png"; plt.tight_layout(); plt.savefig(path)
    print(f"✔ t-SNE embedding saved to {path}")


# ------------------ MAIN ENTRY -----------------------------
if __name__ == "__main__":
    train()
