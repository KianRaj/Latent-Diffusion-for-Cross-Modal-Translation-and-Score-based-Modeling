import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print("Device:", device, " DType:", dtype)

import os, glob, random
from PIL import Image
from tqdm import tqdm

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=dtype,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

total_train_steps = int(pipe.scheduler.config.num_train_timesteps)
print("Total train timesteps:", total_train_steps)

@torch.no_grad()
def get_text_embeds(prompt: str):
    tokens = pipe.tokenizer(
        prompt or "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeds = pipe.text_encoder(tokens.input_ids.to(device))[0]  # (1,77,C)
    return text_embeds

from torchvision import transforms
transform_image = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@torch.no_grad()
def pil_to_vae_latent(img_pil):
    img = img_pil.convert("RGB")
    x = transform_image(img)[None].to(device=device, dtype=dtype)
    posterior = pipe.vae.encode(x)
    z0 = posterior.latent_dist.sample() * pipe.vae.config.scaling_factor
    return z0

@torch.no_grad()
def ddim_invert_to_t(z0, prompt, target_t):
    assert 0 <= target_t < total_train_steps, "within the training range"
    text_embeds = get_text_embeds(prompt)
    z = z0.clone()

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=device, dtype=z.dtype)
    for t in range(target_t):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        model_in = z
        eps = pipe.unet(model_in, t_tensor, encoder_hidden_states=text_embeds).sample

        alpha_t   = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t+1]

        z = ((alpha_next.sqrt() / alpha_t.sqrt()) * (z - (1 - alpha_t).sqrt() * eps)
             + (1 - alpha_next).sqrt() * eps)
    return z

@torch.no_grad()
def _expand_text_embeds_for_batch(text_embeds, batch):
    if text_embeds.shape[0] != batch:
        text_embeds = text_embeds.expand(batch, -1, -1)
    return text_embeds

@torch.no_grad()
def ddim_denoise_to_0(z_t, prompt, start_t):
    text_embeds = get_text_embeds(prompt)
    text_embeds = _expand_text_embeds_for_batch(text_embeds, z_t.shape[0])

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=z_t.device, dtype=z_t.dtype)
    z = z_t.clone()

    for t in range(start_t, 0, -1):
        t_tensor = torch.tensor([t], device=z.device, dtype=torch.long)
        eps = pipe.unet(z, t_tensor, encoder_hidden_states=text_embeds).sample

        alpha_t   = alphas_cumprod[t]
        alpha_tm1 = alphas_cumprod[t-1]

        x0 = (z - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        z  = alpha_tm1.sqrt() * x0 + (1.0 - alpha_tm1).sqrt() * eps

    return z

@torch.no_grad()
def vae_decode_latent(z0):
    z0 = z0.to(device=device, dtype=dtype)
    x  = pipe.vae.decode(z0 / pipe.vae.config.scaling_factor).sample
    x  = (x.clamp(-1, 1) + 1) / 2.0
    return x


rgb_image_data = "/home/suraj/adl_assign/a3/data/m3fd/M3FD_Detection/Vis"
ir_image_data  = "/home/suraj/adl_assign/a3/data/m3fd/M3FD_Detection/Ir"
output_latent_dir = "/home/suraj/adl_assign/a3/data/latents_t400"
os.makedirs(output_latent_dir, exist_ok=True)

random.seed(42)
target_t = 400

all_pairs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(rgb_image_data, "*"))])
all_pairs = [f for f in all_pairs if os.path.exists(os.path.join(ir_image_data, f))]
k = int(0.10 * len(all_pairs))
pairs = random.sample(all_pairs, k)
print(f"Total pairs found: {len(all_pairs)}  Using 10 percent subset: {len(pairs)}")

for fname in tqdm(pairs, desc="DDIM inverting to t=400"):
    rgb_path = os.path.join(rgb_image_data, fname)
    ir_path  = os.path.join(ir_image_data,  fname)
    out_pt = os.path.join(output_latent_dir, fname.rsplit(".", 1)[0] + ".pt")
    if os.path.exists(out_pt):
        continue

    z0_rgb = pil_to_vae_latent(Image.open(rgb_path))
    z0_ir  = pil_to_vae_latent(Image.open(ir_path))

    z400_rgb = ddim_invert_to_t(z0_rgb, prompt="a photo", target_t=target_t)
    z400_ir  = ddim_invert_to_t(z0_ir,  prompt="infrared photo", target_t=target_t)

    torch.save(
        {
            "rgb400": z400_rgb.half().cpu(),
            "ir400": z400_ir.half().cpu(),
            "shape": list(z400_rgb.shape),
            "t": target_t
        },
        out_pt
    )
print("Saved latents to:", output_latent_dir)

class T400LatentPairs(Dataset):
    def __init__(self, latent_dir):
        self.files = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt latents found in {latent_dir}.")
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if path in self._cache:
            return self._cache[path]
        d = torch.load(path, map_location="cpu", weights_only=True)
        z_rgb = d["rgb400"].squeeze(0).float()
        z_ir  = d["ir400"].squeeze(0).float()
        pair = (z_rgb, z_ir)
        self._cache[path] = pair
        return pair

full_ds = T400LatentPairs(output_latent_dir)
n_total = len(full_ds)
n_train = max(1, int(0.8 * n_total))
n_test  = n_total - n_train

g = torch.Generator().manual_seed(42)
train_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_test], generator=g)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4, pin_memory=(device=="cuda"))
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=4, pin_memory=(device=="cuda"))
print(f"training count: {n_train}  test count: {n_test}")

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )

class LatentUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base=64):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)
        self.enc2 = conv_block(base, base*2)
        self.down2 = nn.Conv2d(base*2, base*2, 4, 2, 1)
        self.enc3 = conv_block(base*2, base*4)
        self.down3 = nn.Conv2d(base*4, base*4, 4, 2, 1)

        self.bott = conv_block(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec1 = conv_block(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        x  = self.down1(e1)
        e2 = self.enc2(x)
        x  = self.down2(e2)
        e3 = self.enc3(x)
        x  = self.down3(e3)

        x  = self.bott(x)

        x  = self.up3(x)
        x  = self.dec3(torch.cat([x, e3], dim=1))
        x  = self.up2(x)
        x  = self.dec2(torch.cat([x, e2], dim=1))
        x  = self.up1(x)
        x  = self.dec1(torch.cat([x, e1], dim=1))
        return self.out(x)

torch.backends.cuda.matmul.allow_tf32 = True

model = LatentUNet(in_ch=4, out_ch=4, base=64).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
crit = nn.MSELoss()
scaler = torch.amp.GradScaler(enabled=(device=="cuda"))

num_epochs = 25
save_dir = "/home/suraj/adl_assign/a3/checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, num_epochs+1):
    model.train()
    running = 0.0
    for z_rgb, z_ir in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        z_rgb = z_rgb.to(device=device, dtype=dtype, non_blocking=True)
        z_ir  = z_ir.to(device=device, dtype=dtype, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            pred_ir = model(z_rgb)
            loss = crit(pred_ir, z_ir)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running += loss.item()

    avg = running / max(1, len(train_loader))
    print(f"Epoch {epoch}: train MSE={avg:.6f}")
    torch.save(model.state_dict(), os.path.join(save_dir, f"latent_unet_t400_e{epoch}.pt"))

print("training complete.")

from torchmetrics.functional.image import (
    peak_signal_noise_ratio as tm_psnr,
    structural_similarity_index_measure as tm_ssim,
)
def batch_psnr(pred, target):
    preds = pred.clamp(0, 1).to(dtype=torch.float32)
    targs = target.clamp(0, 1).to(dtype=torch.float32)
    B = preds.size(0)
    vals = []
    for i in range(B):
        v = tm_psnr(preds[i:i+1], targs[i:i+1], data_range=1.0) 
        vals.append(v.detach().reshape(1))                       
    return torch.cat(vals, dim=0)                               


def batch_ssim(pred, target):
    preds = pred.clamp(0, 1).to(dtype=torch.float32)
    targs = target.clamp(0, 1).to(dtype=torch.float32)
    B = preds.size(0)
    vals = []
    for i in range(B):
        v = tm_ssim(preds[i:i+1], targs[i:i+1], data_range=1.0)  
        vals.append(v.detach().reshape(1))                       
    return torch.cat(vals, dim=0)                                

from torchvision.utils import save_image, make_grid

model.eval()
psnr_list, ssim_list = [], []

pair_dir = os.path.join(save_dir, "samples_eval", "pairs_grids")
os.makedirs(pair_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, (z_rgb, z_ir) in enumerate(tqdm(test_loader, desc="Evaluating")):
        z_rgb = z_rgb.to(device=device, dtype=dtype, non_blocking=True)
        z_ir  = z_ir.to(device=device, dtype=dtype, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            pred_ir_t = model(z_rgb)

        z0_pred_ir = ddim_denoise_to_0(pred_ir_t, prompt="infrared photo", start_t=target_t)
        z0_gt_ir   = ddim_denoise_to_0(z_ir,        prompt="infrared photo", start_t=target_t)

        img_pred = vae_decode_latent(z0_pred_ir)
        img_gt   = vae_decode_latent(z0_gt_ir)

        psnr_list.append(batch_psnr(img_pred, img_gt))
        ssim_list.append(batch_ssim(img_pred, img_gt))

        tiles = []
        B = img_pred.size(0)
        for i in range(B):
            tiles.append(img_gt[i])
            tiles.append(img_pred[i])
        grid = make_grid(tiles, nrow=2, padding=2)
        save_image(grid, os.path.join(pair_dir, f"batch_{batch_idx:04d}.png"))

psnr_mean = torch.cat(psnr_list).mean().item() if psnr_list else float('nan')
ssim_mean = torch.cat(ssim_list).mean().item() if ssim_list else float('nan')
print(f"Test PSNR: {psnr_mean:.2f} dB | Test SSIM: {ssim_mean:.4f}")