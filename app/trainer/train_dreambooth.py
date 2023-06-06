import argparse
import hashlib
import itertools
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from app.trainer.dreambooth_dataset import DreamBoothDataset, PromptDataset


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    """Helper function"""
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


@dataclass
class DreamboothTrainer:
    pretrained_model_path: str
    instance_data_dir: str
    instance_prompt: str
    with_prior_preservation: str = None
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    output_dir: str = "text-inversion-model"
    seed: int = None
    resolution: int = 512
    center_crop: bool = False
    class_data_dir: str = None
    train_text_encoder: bool = False
    class_prompt: str = None
    train_batch_size: int = 4
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: int = None
    hub_token: str = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    push_to_hub: bool = False
    hub_model_id: str = None
    logging_dir: str = "logs"
    mixed_precision: str = None
    local_rank: int = -1
    revision: str = None
    tokenizer_name: str = None

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

    def main_trainer(self):
        # Converted original dreambooth code to a class.
        logging_dir = Path(self.output_dir, self.logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            logging_dir=logging_dir,
        )
        if (
            self.train_text_encoder
            and self.gradient_accumulation_steps > 1
            and accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )
        if self.seed is not None:
            set_seed(self.seed)

        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                torch_dtype = (
                    torch.float16
                    if accelerator.device.type == "cuda"
                    else torch.float32
                )
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.pretrained_model_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=self.revision,
                )
                pipeline.set_progress_bar_config(disable=True)
                num_new_images = self.num_class_images - cur_class_images

                # TODO: change print statements to logging.info
                print(f"Number of class images to sample: {num_new_images}.")
                sample_dataset = PromptDataset(self.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=self.sample_batch_size
                )
                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images
                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if accelerator.is_main_process:
            if self.push_to_hub:
                if self.hub_model_id is None:
                    repo_name = get_full_repo_name(
                        Path(self.output_dir).name, token=self.hub_token
                    )
                else:
                    repo_name = self.hub_model_id
                repo = Repository(self.output_dir, clone_from=repo_name)

                with open(
                    os.path.join(self.output_dir, ".gitignore"), "w+"
                ) as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            elif self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        # Tokenizer
        if self.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.tokenizer_name, revision=self.revision
            )

        # This is the statement we will enter!
        # We won't need to load a pretrained tokenizer.
        elif self.pretrained_model_path:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained_model_path,
                subfolder="tokenizer",
                revision=self.revision,
            )

        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_path, subfolder="text_encoder", revision=self.revision
        )

        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_path,
            subfolder="vae",
            revision=self.revision,
        )

        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_path,
            subfolder="unet",
            revision=self.revision,
        )
        vae.requires_grad_(False)
        if not self.train_text_encoder:
            text_encoder.requires_grad_(False)

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        if self.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

        if self.scale_lr:
            self.learning_rate = (
                self.learning_rate
                * self.gradient_accumulation_steps
                * self.train_batch_size
                * accelerator.num_processes
            )
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters())
            if self.train_text_encoder
            else unet.parameters()
        )
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        noise_scheduler = DDPMScheduler.from_config(
            self.pretrained_model_path, subfolder="scheduler"
        )
        train_dataset = DreamBoothDataset(
            instance_data_root=self.instance_data_dir,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_dir
            if self.with_prior_preservation
            else None,
            class_prompt=self.class_prompt,
            tokenizer=tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if self.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()

            input_ids = tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=1,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.gradient_accumulation_steps
        )
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
        )

        if self.train_text_encoder:
            (
                unet,
                text_encoder,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        vae.to(accelerator.device, dtype=weight_dtype)
        if not self.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(
            self.max_train_steps / num_update_steps_per_epoch
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=self.dict())

        total_batch_size = (
            self.train_batch_size
            * accelerator.num_processes
            * self.gradient_accumulation_steps
        )

        print("***** Running training *****")
        print(f"Num examples = {len(train_dataset)}")
        print(f"Num batches each epoch = {len(train_dataloader)}")
        print(f"Num Epochs = {self.num_train_epochs}")
        print(f"Instantaneous batch size per device = {self.train_batch_size}")
        print(
            f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        print(f"Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(self.max_train_steps), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(self.num_train_epochs):
            unet.train()
            if self.train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    if self.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = (
                            F.mse_loss(
                                noise_pred.float(), noise.float(), reduction="none"
                            )
                            .mean([1, 2, 3])
                            .mean()
                        )

                        # Compute prior loss
                        prior_loss = F.mse_loss(
                            noise_pred_prior.float(),
                            noise_prior.float(),
                            reduction="mean",
                        )

                        # Add the prior loss to the instance loss.
                        loss = loss + self.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(
                            noise_pred.float(), noise.float(), reduction="mean"
                        )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                unet.parameters(), text_encoder.parameters()
                            )
                            if self.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.max_train_steps:
                    break

            accelerator.wait_for_everyone()
        if accelerator.is_main_process:

            pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=self.revision,
            )
            pipeline.save_pretrained(self.output_dir)

        accelerator.end_training()
