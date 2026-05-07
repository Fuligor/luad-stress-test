import argparse
import io
import os
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur, crop
from tqdm.auto import tqdm
import math

# Imports
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, AutoPipelineForInpainting
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# --- 1. Dataset Class ---
class HistopathologyDataset(Dataset):
    def __init__(self, data_records, tokenizer, size=512):
        self.data = data_records
        self.tokenizer = tokenizer
        self.size = size

        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
        except Exception as e:
            print(f"Error loading image index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Check if mask is empty (Clean Image)
        raw_mask = Image.open(io.BytesIO(item["mask"])).convert("L")
        is_clean_image = np.array(raw_mask).max() == 0

        pixel_values = self.image_transforms(image)

        # --- TRAINING LOGIC ---
        # 1. Clean Images: Synthesize mask to teach "Regularization" (Filling holes with tissue)
        if is_clean_image:
            mask_values = torch.zeros((1, self.size, self.size))
            h = random.randint(50, 150)
            w = random.randint(50, 150)
            y = random.randint(0, self.size - h)
            x = random.randint(0, self.size - w)
            mask_values[:, y:y+h, x:x+w] = 1.0
            prompt = "Histopathological image of lung tissue, hematoxylin and eosin staining, with no fold artifact visible."
        
        # 2. Artifact Images: Use REAL mask to teach "Generation" (Drawing folds)
        else:
            mask_values = self.mask_transforms(raw_mask)
            # mask_values = (mask_values > 0.5).float()
            prompt = item["text"]

        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        return {
            "pixel_values": pixel_values, 
            "mask_values": mask_values, 
            "input_ids": text_input.input_ids[0],
            "is_clean": is_clean_image
        }

# --- 2. Helper Functions ---
def prepare_mask(pil_mask):
    tensor = transforms.ToTensor()(pil_mask)
    blurred = gaussian_blur(tensor, kernel_size=(21, 21), sigma=3.0)
    return blurred

def get_mask_bbox(mask_tensor):
    indices = torch.nonzero(mask_tensor > 0)
    if len(indices) == 0:
        return 0, 0, mask_tensor.shape[1], mask_tensor.shape[2]
    
    y_min, y_max = indices[:, 1].min(), indices[:, 1].max()
    x_min, x_max = indices[:, 2].min(), indices[:, 2].max()
    
    h, w = y_max - y_min, x_max - x_min
    margin = 10
    top = max(0, int(y_min) - margin)
    left = max(0, int(x_min) - margin)
    height = min(mask_tensor.shape[1] - top, int(h) + 2*margin)
    width = min(mask_tensor.shape[2] - left, int(w) + 2*margin)
    return top, left, height, width

def evaluate_loss(unet, vae, text_encoder, noise_scheduler, dataloader, accelerator):
    unet.eval()
    total_loss = 0
    num_batches = 0
    
    print("Running Validation Loss Check...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Loss"):
            latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            mask = batch["mask_values"]
            mask = torch.nn.functional.interpolate(mask, size=latents.shape[-2:], mode="bilinear")
            masked_image_latents = latents * (1 - mask)
            unet_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
            
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            noise_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches

# --- 3. Main Metric & Gallery Function (UPDATED) ---
def run_metrics_and_gallery(vae, text_encoder, tokenizer, unet, args, accelerator, val_dataloader, fixed_visual_samples, mask_bank):
    print("--- Running Full Validation Suite ---")
    
    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    
    P_CLEAN = "Histopathological image of lung tissue, hematoxylin and eosin staining, with no fold artifact visible."
    P_ARTIFACT = "Histopathological image of lung tissue, hematoxylin and eosin staining, with a fold artifact visible."

    # --- PART A: Visual Gallery (Now with Masks!) ---
    gallery_images = []
    
    for i, sample in enumerate(fixed_visual_samples):
        original_img = Image.open(io.BytesIO(sample['image'])).convert("RGB").resize((512,512))
        original_mask = Image.open(io.BytesIO(sample['mask'])).convert("L").resize((512,512))
        is_clean = np.array(original_mask).max() == 0
        
        # 1. Case: Clean Image -> Add Artifact
        if is_clean:
            if 'borrowed_mask_bytes' in sample:
                 mask_input = Image.open(io.BytesIO(sample['borrowed_mask_bytes'])).convert("L").resize((512,512))
            else:
                 mask_input = Image.open(io.BytesIO(random.choice(mask_bank))).convert("L").resize((512,512))
            prompt = P_ARTIFACT 
            case_name = f"Sample {i} (Add Artifact)"
            
        # 2. Case: Artifact Image -> Restore/Regularize (Check if it can draw prompt)
        # Note: If we use P_ARTIFACT here, it should reconstruct the fold. 
        # If we use P_CLEAN, it tries to remove it (which we agreed is not the goal, but interesting to see).
        # Let's stick to P_ARTIFACT (Restoration) to see if it learns the texture.
        else:
            mask_input = original_mask
            prompt = P_ARTIFACT 
            case_name = f"Sample {i} (Restore Artifact)"

        with torch.autocast("cuda"):
            out = pipeline(prompt=prompt, image=original_img, mask_image=prepare_mask(mask_input),
                           num_inference_steps=20, guidance_scale=7.5, generator=generator).images[0]
        
        # LOGGING: Input | Mask | Output
        gallery_images.append(wandb.Image(original_img, caption=f"{case_name}: Input"))
        gallery_images.append(wandb.Image(mask_input, caption=f"{case_name}: Mask"))
        gallery_images.append(wandb.Image(out, caption=f"{case_name}: Result"))
    
    accelerator.get_tracker("wandb").log({"validation_gallery": gallery_images})

    # --- PART B: METRICS ---
    print("Computing Metrics...")
    
    # FID Generation: [Generated Folds] vs [Real Folds]
    fid_gen = FrechetInceptionDistance(feature=64).to(accelerator.device)
    
    # FID Regularization: [Inpainted Clean Tissue] vs [Real Clean Tissue]
    fid_reg = FrechetInceptionDistance(feature=64).to(accelerator.device)
    
    # Pairwise (Only for Regularization Clean->Clean)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(accelerator.device)
    
    BATCH_LIMIT = 20
    count = 0
    acc_lpips = 0.0
    n_reg_samples = 0
    
    for batch in tqdm(val_dataloader, desc="Metrics"):
        if count >= BATCH_LIMIT: break
        
        real_imgs = batch["pixel_values"].to(accelerator.device)
        is_clean_batch = batch["is_clean"].to(accelerator.device)
        bsz = real_imgs.shape[0]

        pil_images = [transforms.ToPILImage()((real_imgs[k]*0.5+0.5).clamp(0,1).cpu()) for k in range(bsz)]

        # Identify clean and artifact indices in this batch
        clean_indices = [k for k in range(bsz) if is_clean_batch[k]]
        artifact_indices = [k for k in range(bsz) if not is_clean_batch[k]]

        # --- TASK 1: GENERATION (Clean + Borrowed Mask + Prompt"Fold" -> Artifact) ---
        if len(clean_indices) > 0:
            borrowed_masks_pil = [Image.open(io.BytesIO(random.choice(mask_bank))).convert("L").resize((512,512)) for _ in clean_indices]
            borrowed_masks_blurred = [prepare_mask(m) for m in borrowed_masks_pil]
            clean_imgs_pil = [pil_images[k] for k in clean_indices]

            with torch.autocast("cuda"):
                fake_artifacts = pipeline(
                    prompt=[P_ARTIFACT]*len(clean_indices), image=clean_imgs_pil, mask_image=borrowed_masks_blurred,
                    num_inference_steps=20, output_type="pt"
                ).images.to(accelerator.device)

            # FID Gen: Add Fakes
            for idx, k in enumerate(clean_indices):
                fid_gen.update((fake_artifacts[idx].clamp(0,1)*255).to(torch.uint8).unsqueeze(0), real=False)

        # --- TASK 2: REGULARIZATION (Clean + Borrowed Mask + Prompt"No Fold" -> Clean) ---
        # We reuse the clean images but ask for NO FOLD. The model should ignore the mask shape and fill with tissue.
        if len(clean_indices) > 0:
            # Re-use borrowed masks to see if model ignores the specific fold shape
            with torch.autocast("cuda"):
                fake_clean = pipeline(
                    prompt=[P_CLEAN]*len(clean_indices), image=clean_imgs_pil, mask_image=borrowed_masks_blurred,
                    num_inference_steps=20, output_type="pt"
                ).images.to(accelerator.device)

            for idx, k in enumerate(clean_indices):
                # FID Reg: Add Fakes
                fid_reg.update((fake_clean[idx].clamp(0,1)*255).to(torch.uint8).unsqueeze(0), real=False)
                
                # FID Reg: Add Reals (Ground Truth Clean)
                real_uint8 = ((real_imgs[k]*0.5+0.5).clamp(0,1)*255).to(torch.uint8).unsqueeze(0)
                fid_reg.update(real_uint8, real=True)

                # LPIPS (Crop to mask)
                m_tensor = transforms.ToTensor()(borrowed_masks_pil[idx]).to(accelerator.device)
                top, left, h, w = get_mask_bbox(m_tensor)
                if h > 8 and w > 8:
                    real_crop = crop(real_imgs[k]*0.5+0.5, top, left, h, w).unsqueeze(0).clamp(0,1)
                    fake_crop = crop(fake_clean[idx], top, left, h, w).unsqueeze(0).clamp(0,1)
                    
                    if h < 32 or w < 32: # Resize tiny crops
                        real_crop = F.interpolate(real_crop, size=(64,64), mode='bilinear')
                        fake_crop = F.interpolate(fake_crop, size=(64,64), mode='bilinear')

                    acc_lpips += lpips(fake_crop, real_crop).item()
                    n_reg_samples += 1

        # --- UPDATE FID GEN REAL STATS ---
        # We need Real Artifacts to compare against. We get them from the artifact_indices in the batch.
        for k in artifact_indices:
            real_uint8 = ((real_imgs[k]*0.5+0.5).clamp(0,1)*255).to(torch.uint8).unsqueeze(0)
            fid_gen.update(real_uint8, real=True)

        count += 1
        
    # Finalize
    metrics = {}
    try:
        # Check if we saw enough samples
        metrics["fid_generation"] = fid_gen.compute().item()
        metrics["fid_regularization"] = fid_reg.compute().item()
        if n_reg_samples > 0:
            metrics["val_lpips_reg"] = acc_lpips / n_reg_samples
            
        print(f"Metrics: {metrics}")
        accelerator.get_tracker("wandb").log(metrics)
    except Exception as e:
        print(f"Metric Error (probably empty batch for one class): {e}")

    del pipeline, fid_gen, fid_reg, lpips
    torch.cuda.empty_cache()

# --- 4. Main Training Loop ---
def main(data_records):
    args = argparse.Namespace()
    
    # === CONFIG ===
    args.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    args.custom_checkpoint_path = "./sd-histopathology-artefact-gen-v2" 
    args.output_dir = "./sd-histopathology-artefact-gen-FINAL-v2-BILINEAR"
    args.train_batch_size = 2           
    args.gradient_accumulation_steps = 2 
    args.learning_rate = 1e-5           
    args.lr_scheduler = "cosine"        
    args.num_train_epochs = 10          
    args.resolution = 512
    args.mixed_precision = "fp16" 
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with="wandb")
    if accelerator.is_main_process: accelerator.init_trackers("histopathology_artefact_gen_v2", config=vars(args))

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if args.custom_checkpoint_path and os.path.exists(args.custom_checkpoint_path):
        try: unet = UNet2DConditionModel.from_pretrained(args.custom_checkpoint_path, subfolder="unet"); print("Loaded custom checkpoint.")
        except: print("Could not load custom checkpoint. Starting fresh.")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # --- DATA PREP ---
    train_data = [d for d in data_records if d['split'] == 'train']
    val_data = [d for d in data_records if d['split'] == 'val']
    
    # 1. Mask Bank
    print("Creating Mask Bank...")
    all_artifact_masks = [d['mask'] for d in data_records if np.array(Image.open(io.BytesIO(d['mask'])).convert("L")).max() > 0]
    mask_bank = random.sample(all_artifact_masks, min(200, len(all_artifact_masks)))
    
    # 2. Visual Gallery
    val_artifacts = [d for d in val_data if np.array(Image.open(io.BytesIO(d['mask'])).convert("L")).max() > 0]
    val_cleans = [d for d in val_data if np.array(Image.open(io.BytesIO(d['mask'])).convert("L")).max() == 0]
    fixed_visual_samples = val_artifacts[:4] + val_cleans[:4]
    for i in range(4, len(fixed_visual_samples)):
        fixed_visual_samples[i]['borrowed_mask_bytes'] = random.choice(mask_bank)

    train_dl = DataLoader(HistopathologyDataset(train_data, tokenizer, size=args.resolution), batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(HistopathologyDataset(val_data, tokenizer, size=args.resolution), batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    unet, optimizer, train_dl, val_dl = accelerator.prepare(unet, optimizer, train_dl, val_dl)
    vae.to(accelerator.device); text_encoder.to(accelerator.device)

    from diffusers.optimization import get_scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=args.num_train_epochs * num_update_steps_per_epoch)

    print("Starting training...")
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                mask = batch["mask_values"]
                mask = torch.nn.functional.interpolate(mask, size=latents.shape[-2:], mode="bilinear")
                masked_image_latents = latents * (1 - mask)
                unet_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
                
                loss = F.mse_loss(unet(unet_input, timesteps, text_encoder(batch["input_ids"])[0]).sample.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if accelerator.is_main_process: wandb.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

        if accelerator.is_main_process:
            val_loss = evaluate_loss(unet, vae, text_encoder, noise_scheduler, val_dl, accelerator)
            wandb.log({"val_loss": val_loss, "epoch": epoch})
            
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            AutoPipelineForInpainting.from_pretrained(
                args.pretrained_model_name_or_path, unet=accelerator.unwrap_model(unet), text_encoder=text_encoder, vae=vae
            ).save_pretrained(save_path)
            
            run_metrics_and_gallery(vae, text_encoder, tokenizer, unet, args, accelerator, val_dl, fixed_visual_samples, mask_bank)

    accelerator.end_training()

if __name__ == "__main__":
    try:
        parquet_file = "combined_dataset_folds_and_clean_MPP_09_ttvsplit.parquet"
        df = pd.read_parquet(parquet_file)
        main(df.to_dict('records'))
    except Exception as e:
        print(f"Error: {e}")