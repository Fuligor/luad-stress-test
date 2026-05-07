import argparse
import io
import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
import random
import itertools
from PIL import Image
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance

# Data & IO
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
import pyarrow.parquet as pq
import wandb

# Modeling
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoPipelineForInpainting,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore

# ============================================================
# 1. DATASET CLASSES
# ============================================================


class HybridPrecipitateDataset(IterableDataset):
    """
    Script 2 Original Architecture:
    Streams data and handles mixing Real/Synthetic/Clean sources.
    """

    def __init__(self, dataset_path, split, tokenizer, size=512, probs=[0.2, 0.4, 0.4]):
        self.tokenizer = tokenizer
        self.size = size
        self.dataset_path = dataset_path
        self.split_name = split
        self.probs = probs
        self.source_map = {"real": 0, "synthetic": 1, "clean": 2}

        self.img_tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.mask_tf = transforms.Compose(
            [
                transforms.Resize(
                    (size, size), interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )

        # Preload real data for mixing and for Metric/Validation extraction later
        self.real_cache = self._preload_real_data()

    def _preload_real_data(self):
        # FIX: Handle case where dataset_path is None (used by wrapper class)
        if self.dataset_path is None:
            return []

        print(f"[{self.split_name}] Pre-loading 'Real' samples into RAM...")
        cache = []
        try:
            pf = pq.ParquetFile(self.dataset_path)
            # Ensure we only load what we need
            iterator = pf.iter_batches(
                batch_size=100, columns=["image", "mask", "source", "split", "text"]
            )
            for batch in iterator:
                rows = batch.to_pylist()
                for row in rows:
                    if row["split"] == self.split_name and row["source"] == "real":
                        cache.append(row)
        except Exception as e:
            print(f"Warning: Could not load real data cache: {e}")

        print(f"Loaded {len(cache)} Real samples into memory.")
        return cache

    def get_large_data_stream(self):
        pf = pq.ParquetFile(self.dataset_path)
        iterator = pf.iter_batches(
            batch_size=4, columns=["image", "mask", "source", "split", "text"]
        )
        for batch in iterator:
            rows = batch.to_pylist()
            for row in rows:
                if row["split"] != self.split_name:
                    continue
                if row["source"] == "real":
                    continue  # Served from cache
                yield row

    def process_item(self, item):
        source_str = item["source"]
        source_id = self.source_map.get(source_str, 1)

        try:
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
            raw_mask = Image.open(io.BytesIO(item["mask"])).convert("L")
        except Exception:
            # print(f"Skipping corrupt image: {e}")
            return None

        pixel_values = self.img_tf(image)
        mask_tensor = self.mask_tf(raw_mask)

        # --- ARCHITECTURAL CHANGE PRESERVED: MASK DILATION ---
        # Dilate mask to force edge blending for precipitates
        if source_str in ["real", "synthetic"]:
            mask_tensor = F.max_pool2d(
                mask_tensor, kernel_size=21, stride=1, padding=10
            )

        # --- ARCHITECTURAL CHANGE PRESERVED: HOLE FILLING ---
        # For clean data, create random box masks
        if source_str == "clean":
            mask_tensor.fill_(0.0)
            h = random.randint(64, 160)
            w = random.randint(64, 160)
            y = random.randint(0, self.size - h)
            x = random.randint(0, self.size - w)
            mask_tensor[:, y : y + h, x : x + w] = 1.0

        mask = (mask_tensor > 0.5).float()

        # --- ARCHITECTURAL CHANGE PRESERVED: PROMPT ENGINEERING ---
        base_prompt = item.get("text", "")
        transparency_keywords = (
            "semi-transparent overlay, see-through texture, blending with tissue"
        )

        if source_str == "clean":
            prompt = "Histopathological image of lung tissue, hematoxylin and eosin staining, clean tissue."
        elif source_str in ["real", "synthetic"]:
            if not base_prompt or len(base_prompt) < 5:
                prompt = f"Histopathological image of lung tissue, hematoxylin and eosin staining, {transparency_keywords}, precipitate artifact."
            else:
                prompt = f"{base_prompt}, {transparency_keywords}"
        else:
            prompt = base_prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "mask_values": mask,
            "input_ids": text_inputs.input_ids[0],
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "is_clean": (source_str == "clean"),
        }

    def __iter__(self):
        if len(self.real_cache) > 0:
            real_iterator = itertools.cycle(self.real_cache)
        else:
            real_iterator = iter([])

        large_data_iterator = self.get_large_data_stream()

        while True:
            val = random.random()
            target_is_real = val < self.probs[0]

            try:
                if target_is_real and len(self.real_cache) > 0:
                    item = next(real_iterator)
                else:
                    item = next(large_data_iterator)

                processed = self.process_item(item)
                if processed is None:
                    continue
                yield processed

            except StopIteration:
                large_data_iterator = self.get_large_data_stream()


class ValidationMetricDataset(Dataset):
    """
    New Helper Class: Allows us to turn the list of cached validation data
    into a standard Map-Style dataset required for the metric evaluation loop.
    """

    def __init__(self, data_list, tokenizer, size=512):
        self.data = data_list
        self.tokenizer = tokenizer
        self.size = size
        # Re-use logic from main dataset
        self.processor = HybridPrecipitateDataset(None, None, tokenizer, size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.processor.process_item(item)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================


def get_mask_bbox(mask_tensor):
    """From Script 1: Used for LPIPS cropping"""
    indices = torch.nonzero(mask_tensor > 0)
    if len(indices) == 0:
        return 0, 0, mask_tensor.shape[1], mask_tensor.shape[2]

    y_min, y_max = indices[:, 1].min(), indices[:, 1].max()
    x_min, x_max = indices[:, 2].min(), indices[:, 2].max()

    h, w = y_max - y_min, x_max - x_min
    margin = 10
    top = max(0, int(y_min) - margin)
    left = max(0, int(x_min) - margin)
    height = min(mask_tensor.shape[1] - top, int(h) + 2 * margin)
    width = min(mask_tensor.shape[2] - left, int(w) + 2 * margin)
    return top, left, height, width


def apply_noise_offset(latents, noise, strength=0.1):
    """From Script 2: Preserved for dark/bright artifact generation"""
    if random.random() < 0.25:
        offset = torch.randn(
            latents.shape[0], latents.shape[1], 1, 1, device=latents.device
        )
        return noise + strength * offset
    return noise


def prepare_mask_dilated(pil_mask, size=(512, 512)):
    """
    ADAPTED: Uses Script 2's Dilation logic instead of Script 1's Blur.
    Ensures metrics use the same mask style as training.
    """
    tensor = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )(pil_mask).unsqueeze(0)  # Add batch dim

    # Kernel 21, Padding 10 match training logic
    dilated = F.max_pool2d(tensor, kernel_size=21, stride=1, padding=10)
    return dilated.squeeze(0)  # Remove batch dim


def evaluate_loss(unet, vae, text_encoder, noise_scheduler, dataloader, accelerator):
    """Computes validation loss on the provided dataloader"""
    unet.eval()
    total_loss = 0
    num_batches = 0

    print("Running Validation Loss Check...")
    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= 20:
                break  # Limit val loss calc to 20 batches to save time

            # Prepare inputs
            pixel_values = batch["pixel_values"].to(accelerator.device)
            mask_values = batch["mask_values"].to(accelerator.device)
            input_ids = batch["input_ids"].to(accelerator.device)

            latents = (
                vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
                * 0.18215
            )
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Interpolate mask to latent size (nearest neighbor)
            mask = F.interpolate(mask_values, size=latents.shape[-2:], mode="nearest")
            masked_image_latents = latents * (1 - mask)

            unet_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
            encoder_hidden_states = text_encoder(input_ids)[0]

            noise_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def calculate_wasserstein_dist(gen_tensor, real_tensor):
    """
    Calculates Wasserstein Distance on Color Histograms for a single pair of images.
    Input: [C, H, W] tensors in range [0, 1]
    Output: Scalar distance
    """
    # Convert to CPU numpy uint8 [0, 255]
    gen_np = (gen_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    real_np = (real_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

    d = 0
    for c in range(3):
        # Compute histogram for each channel
        hist_gen, _ = np.histogram(
            gen_np[c].flatten(), bins=256, range=(0, 256), density=True
        )
        hist_real, _ = np.histogram(
            real_np[c].flatten(), bins=256, range=(0, 256), density=True
        )

        # Calculate Earth Mover's Distance between distributions
        d += wasserstein_distance(np.arange(256), np.arange(256), hist_gen, hist_real)

    return d / 3.0


# ============================================================
# 3. METRICS AND GALLERY (MERGED)
# ============================================================


def run_metrics_and_gallery(
    vae,
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    val_dataloader,
    fixed_gallery_batch,
    mask_bank,
    step,
):
    # FIX: Safety check for empty dataloader
    if val_dataloader is None or len(val_dataloader) == 0:
        print(f"--- Skipping Metrics (Step {step}): No Validation Data Available ---")
        return

    print(f"--- Running Validation & Metrics (Step {step}) ---")

    # 1. Ensure deterministic behavior for metrics by seeding
    # This coupled with shuffle=False in the dataloader ensures "Constant masks" in metrics loop
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.pretrained_model,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    # PROMPTS
    transparency_kw = (
        "semi-transparent overlay, blending with tissue, precipitate artifact"
    )
    P_CLEAN = "Histopathological image of lung tissue, hematoxylin and eosin staining, clean tissue."
    P_ARTIFACT = f"Histopathological image of lung tissue, hematoxylin and eosin staining, {transparency_kw}"

    # --- PART A: VISUAL GALLERY (FIXED SAMPLES) ---
    gallery_images = []

    if fixed_gallery_batch:
        print("Generating Fixed Gallery...")
        for i, item in enumerate(fixed_gallery_batch):
            # item has: 'pixel_values', 'mask_values', 'is_clean', 'original_mask_pil'

            orig_tensor = item["pixel_values"]
            orig_img = transforms.ToPILImage()((orig_tensor * 0.5 + 0.5).clamp(0, 1))
            is_clean = item["is_clean"]

            g_type = item.get("type", "unknown")

            if g_type == "clean_gen":
                # Case 1: Clean -> Generate Artifact
                # Use the PRE-ASSIGNED fixed mask
                if "fixed_mask_bytes" in item:
                    mask_input = Image.open(
                        io.BytesIO(item["fixed_mask_bytes"])
                    ).convert("L")
                elif mask_bank:
                    mask_input = Image.open(
                        io.BytesIO(random.choice(mask_bank))
                    ).convert("L")
                else:
                    mask_input = Image.new("L", (512, 512), 0)

                mask_tensor_prepared = prepare_mask_dilated(mask_input)
                mask_pil_prepared = transforms.ToPILImage()(mask_tensor_prepared)
                prompt = P_ARTIFACT
                case_name = f"Sample {i} (Gen Artifact)"

            elif g_type == "artifact_restore":
                # Case 2: Artifact -> Restore
                mask_input = item["original_mask_pil"]
                mask_tensor_prepared = prepare_mask_dilated(mask_input)
                mask_pil_prepared = transforms.ToPILImage()(mask_tensor_prepared)
                prompt = P_ARTIFACT
                case_name = f"Sample {i} (Restore)"

            elif g_type == "clean_identity":
                # Case 3: Identity Check
                mask_input = Image.new("L", (512, 512), 0)
                mask_tensor_prepared = prepare_mask_dilated(mask_input)
                mask_pil_prepared = transforms.ToPILImage()(mask_tensor_prepared)
                prompt = P_CLEAN
                case_name = f"Sample {i} (Identity)"

            else:
                continue

            with torch.autocast("cuda"):
                out = pipeline(
                    prompt=prompt,
                    image=orig_img,
                    mask_image=mask_pil_prepared,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]

            gallery_images.append(wandb.Image(orig_img, caption=f"{case_name}: Input"))
            gallery_images.append(
                wandb.Image(mask_pil_prepared, caption=f"{case_name}: Mask")
            )
            gallery_images.append(wandb.Image(out, caption=f"{case_name}: Result"))

        accelerator.log({"validation_gallery": gallery_images}, step=step)

    # --- PART B: METRICS ---
    print("Computing Metrics (FID/KID/CLIP/Wasserstein/LPIPS)...")

    # Initialize Metrics
    fid_gen = FrechetInceptionDistance(feature=64).to(accelerator.device)
    kid_gen = KernelInceptionDistance(subset_size=10, feature=64).to(accelerator.device)

    fid_reg = FrechetInceptionDistance(feature=64).to(accelerator.device)
    kid_reg = KernelInceptionDistance(subset_size=10, feature=64).to(accelerator.device)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(accelerator.device)

    # NEW: CLIP Score
    try:
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(
            accelerator.device
        )
        use_clip = True
    except Exception as e:
        print(f"Warning: Could not initialize CLIP Metric (download fail?): {e}")
        use_clip = False

    BATCH_LIMIT = 25
    count = 0

    # Accumulators for scalar metrics
    acc_lpips = 0.0
    acc_color_dist = 0.0
    acc_clip_gen = 0.0
    acc_clip_reg = 0.0

    n_reg_samples = 0
    n_gen_samples = 0

    for batch in tqdm(val_dataloader, desc="Metrics"):
        if count >= BATCH_LIMIT:
            break

        real_imgs = batch["pixel_values"].to(accelerator.device)
        is_clean_batch = batch["is_clean"]  # boolean tensor
        bsz = real_imgs.shape[0]

        pil_images = [
            transforms.ToPILImage()((real_imgs[k] * 0.5 + 0.5).clamp(0, 1).cpu())
            for k in range(bsz)
        ]

        clean_indices = [k for k in range(bsz) if is_clean_batch[k]]
        artifact_indices = [k for k in range(bsz) if not is_clean_batch[k]]

        # TASK 1: GENERATION (Clean -> Artifact)
        if len(clean_indices) > 0 and mask_bank:
            # Deterministic selection due to seed(42)
            borrowed_masks_pil = [
                Image.open(io.BytesIO(random.choice(mask_bank))).convert("L")
                for _ in clean_indices
            ]
            borrowed_masks_prepared = [
                transforms.ToPILImage()(prepare_mask_dilated(m))
                for m in borrowed_masks_pil
            ]
            clean_imgs_pil = [pil_images[k] for k in clean_indices]

            with torch.autocast("cuda"):
                fake_artifacts = pipeline(
                    prompt=[P_ARTIFACT] * len(clean_indices),
                    image=clean_imgs_pil,
                    mask_image=borrowed_masks_prepared,
                    num_inference_steps=20,
                    output_type="pt",
                ).images.to(accelerator.device)

            # Update Gen Metrics
            for idx, k in enumerate(clean_indices):
                fake_uint8 = (
                    (fake_artifacts[idx].clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0)
                )
                fid_gen.update(fake_uint8, real=False)
                kid_gen.update(fake_uint8, real=False)

            # CLIP Score (Generation)
            if use_clip:
                # Need uint8 tensors [N, C, H, W] or range [0, 255]
                imgs_for_clip = (fake_artifacts.clamp(0, 1) * 255).to(torch.uint8)
                score = clip_metric(imgs_for_clip, [P_ARTIFACT] * len(clean_indices))
                acc_clip_gen += score.item() * len(clean_indices)
                n_gen_samples += len(clean_indices)

        # TASK 2: REGULARIZATION (Clean -> Clean)
        if len(clean_indices) > 0:
            clean_masks_pil = []
            for _ in clean_indices:
                m = torch.zeros((1, 512, 512))
                h, w = random.randint(64, 160), random.randint(64, 160)
                y, x = random.randint(0, 512 - h), random.randint(0, 512 - w)
                m[:, y : y + h, x : x + w] = 1.0
                clean_masks_pil.append(transforms.ToPILImage()(m))

            clean_imgs_pil = [pil_images[k] for k in clean_indices]

            with torch.autocast("cuda"):
                fake_clean = pipeline(
                    prompt=[P_CLEAN] * len(clean_indices),
                    image=clean_imgs_pil,
                    mask_image=clean_masks_pil,
                    num_inference_steps=20,
                    output_type="pt",
                ).images.to(accelerator.device)

            for idx, k in enumerate(clean_indices):
                fake_uint8 = (
                    (fake_clean[idx].clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0)
                )
                real_uint8 = (
                    ((real_imgs[k] * 0.5 + 0.5).clamp(0, 1) * 255)
                    .to(torch.uint8)
                    .unsqueeze(0)
                )

                # Distribution Metrics
                fid_reg.update(fake_uint8, real=False)
                kid_reg.update(fake_uint8, real=False)
                fid_reg.update(real_uint8, real=True)
                kid_reg.update(real_uint8, real=True)

                # Color Distance (Wasserstein)
                # Calculate on normalized tensors [0, 1]
                fake_norm = fake_clean[idx].clamp(0, 1)
                real_norm = (real_imgs[k] * 0.5 + 0.5).clamp(0, 1)

                w_dist = calculate_wasserstein_dist(fake_norm, real_norm)
                acc_color_dist += w_dist

                # LPIPS
                m_tensor = transforms.ToTensor()(clean_masks_pil[idx]).to(
                    accelerator.device
                )
                top, left, h, w = get_mask_bbox(m_tensor)

                if h > 8 and w > 8:
                    real_crop = crop(real_norm, top, left, h, w).unsqueeze(0)
                    fake_crop = crop(fake_norm, top, left, h, w).unsqueeze(0)
                    if h < 32 or w < 32:
                        real_crop = F.interpolate(
                            real_crop, size=(64, 64), mode="bilinear"
                        )
                        fake_crop = F.interpolate(
                            fake_crop, size=(64, 64), mode="bilinear"
                        )
                    acc_lpips += lpips(fake_crop, real_crop).item()

                n_reg_samples += 1

            # CLIP Score (Regularization)
            if use_clip:
                imgs_for_clip = (fake_clean.clamp(0, 1) * 255).to(torch.uint8)
                score = clip_metric(imgs_for_clip, [P_CLEAN] * len(clean_indices))
                acc_clip_reg += score.item() * len(clean_indices)

        # TASK 3: GEN METRICS REAL STATS
        for k in artifact_indices:
            real_uint8 = (
                ((real_imgs[k] * 0.5 + 0.5).clamp(0, 1) * 255)
                .to(torch.uint8)
                .unsqueeze(0)
            )
            fid_gen.update(real_uint8, real=True)
            kid_gen.update(real_uint8, real=True)

        count += 1

    # Compute & Log Metrics
    metrics = {}
    try:
        metrics["fid_generation"] = fid_gen.compute().item()
        metrics["kid_generation_mean"], metrics["kid_generation_std"] = (
            kid_gen.compute()
        )

        metrics["fid_regularization"] = fid_reg.compute().item()
        metrics["kid_regularization_mean"], metrics["kid_regularization_std"] = (
            kid_reg.compute()
        )

        if n_reg_samples > 0:
            metrics["val_lpips_reg"] = acc_lpips / n_reg_samples
            metrics["val_color_wasserstein_reg"] = acc_color_dist / n_reg_samples

        if n_gen_samples > 0 and use_clip:
            metrics["val_clip_score_gen"] = acc_clip_gen / n_gen_samples

        if n_reg_samples > 0 and use_clip:
            metrics["val_clip_score_reg"] = acc_clip_reg / n_reg_samples

    except Exception as e:
        print(f"Metric calculation failed (insufficient samples?): {e}")

    try:
        val_loss = evaluate_loss(
            unet, vae, text_encoder, pipeline.scheduler, val_dataloader, accelerator
        )
        metrics["val_loss"] = val_loss
    except Exception as e:
        print(f"Val loss calculation failed: {e}")

    print(f"Metrics: {metrics}")
    accelerator.log(metrics, step=step)

    del pipeline, fid_gen, fid_reg, kid_gen, kid_reg, lpips
    if use_clip:
        del clip_metric
    torch.cuda.empty_cache()


# ============================================================
# 4. TRAINING LOOP
# ============================================================


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="fp16",
        log_with="wandb",
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.init_trackers("sd-inpaint-precipitates-metrics", config=vars(args))

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    # --- DATA SETUP ---
    # 1. Train Dataset
    train_ds = HybridPrecipitateDataset(
        dataset_path=args.data,
        split="train",
        tokenizer=tokenizer,
        probs=[0.4, 0.4, 0.2],
    )

    # 2. Validation Dataset
    print("Preparing Validation Data for Metrics...")
    val_cache_real = []
    val_cache_clean = []

    try:
        temp_val_ds = HybridPrecipitateDataset(args.data, "val", tokenizer)
        val_cache_real = temp_val_ds.real_cache
    except Exception as e:
        print(f"Warning: Failed to load val dataset wrapper: {e}")

    try:
        pf = pq.ParquetFile(args.data)
        for batch in pf.iter_batches(
            batch_size=100, columns=["image", "mask", "source", "split", "text"]
        ):
            rows = batch.to_pylist()
            for row in rows:
                if row["split"] == "val" and row["source"] == "clean":
                    val_cache_clean.append(row)
                if len(val_cache_clean) > 100:
                    break
    except Exception as e:
        print(f"Warning: Failed to load clean val samples: {e}")

    # --- FALLBACK LOGIC ---
    if len(val_cache_real) == 0 and len(val_cache_clean) == 0:
        print("!! WARNING !! No validation data found. Falling back to TRAIN data.")
        val_cache_real = train_ds.real_cache[:50]
        try:
            pf = pq.ParquetFile(args.data)
            for batch in pf.iter_batches(
                batch_size=100, columns=["image", "mask", "source", "split", "text"]
            ):
                rows = batch.to_pylist()
                for row in rows:
                    if row["split"] == "train" and row["source"] == "clean":
                        val_cache_clean.append(row)
                    if len(val_cache_clean) > 50:
                        break
                if len(val_cache_clean) > 50:
                    break
        except Exception as e:
            print(f"Error reading parquet for fallback: {e}")

    val_data_list = val_cache_real[:100] + val_cache_clean[:100]

    # Create DataLoader with shuffle=False for deterministic metrics
    if len(val_data_list) > 0:
        val_dataset = ValidationMetricDataset(val_data_list, tokenizer)
        val_dl = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
    else:
        print("CRITICAL: No data for metrics found. Metrics will be skipped.")
        val_dl = None

    # --- PREPARE FIXED VISUAL SAMPLES ---
    print("Selecting Fixed Gallery Samples...")
    fixed_gallery_batch = []
    processor_ref = (
        train_ds if hasattr(train_ds, "process_item") else val_dataset.processor
    )

    # 3. Create Mask Bank
    print("Building Mask Bank...")
    mask_bank = [row["mask"] for row in train_ds.real_cache[:300]]

    # 1. Clean for Gen
    for raw_item in val_cache_clean[:2]:
        proc = processor_ref.process_item(raw_item)
        if proc:
            proc["type"] = "clean_gen"
            # CONSTANT MASK ASSIGNMENT for Gallery
            if mask_bank:
                proc["fixed_mask_bytes"] = random.choice(mask_bank)
            fixed_gallery_batch.append(proc)

    # 2. Real for Restore
    for raw_item in val_cache_real[:2]:
        proc = processor_ref.process_item(raw_item)
        if proc:
            proc["type"] = "artifact_restore"
            proc["original_mask_pil"] = (
                Image.open(io.BytesIO(raw_item["mask"])).convert("L").resize((512, 512))
            )
            fixed_gallery_batch.append(proc)

    # 3. Clean for Identity
    if len(val_cache_clean) > 2:
        raw_item = val_cache_clean[2]
    elif len(val_cache_clean) > 0:
        raw_item = val_cache_clean[0]
    else:
        raw_item = None

    if raw_item:
        proc = processor_ref.process_item(raw_item)
        if proc:
            proc["type"] = "clean_identity"
            fixed_gallery_batch.append(proc)

    print(f"Fixed Gallery Batch prepared with {len(fixed_gallery_batch)} items.")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    unet, optimizer, train_dl, val_dl = accelerator.prepare(
        unet, optimizer, train_dl, val_dl
    )

    global_step = 0
    max_steps = args.max_steps

    print(f"Starting training for {max_steps} steps...")
    data_iter = iter(train_dl)
    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        unet.train()

        with accelerator.accumulate(unet):
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            noise = apply_noise_offset(latents, noise, strength=0.1)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            mask = F.interpolate(
                batch["mask_values"], size=noisy_latents.shape[-2:], mode="nearest"
            )
            masked_image_latents = latents * (1 - mask)

            model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            noise_pred = unet(model_input, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.is_local_main_process:
            accelerator.log({"loss": loss.item()}, step=global_step)
            progress_bar.update(1)

        if global_step > 0 and global_step % args.validation_steps == 0:
            if accelerator.is_main_process:
                gc.collect()
                torch.cuda.empty_cache()
                run_metrics_and_gallery(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    val_dl,
                    fixed_gallery_batch,
                    mask_bank,
                    global_step,
                )

        if global_step > 0 and global_step % args.save_steps == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.unwrap_model(unet).save_pretrained(
                    os.path.join(save_path, "unet")
                )
                print(f"Saved checkpoint to {save_path}")

        global_step += 1

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-precipitates-metrics-v2-338_09MPP2_diff_ratio2",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    train(args)
