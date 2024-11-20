import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import intel_extension_for_pytorch as ipex
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class AnimateDiffModel:
    def __init__(self, device="xpu", dtype=torch.bfloat16, step=4):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.adapter = None
        self.logger = logging.getLogger(__name__)
        self.step = step

    def initialize(self):
        try:
            self.logger.info("Initializing AnimateDiff model...")
            repo = "ByteDance/AnimateDiff-Lightning"
            ckpt = f"animatediff_lightning_{self.step}step_diffusers.safetensors"
            base = "emilianJR/epiCRealism"
            self.adapter = MotionAdapter().to(self.device, self.dtype)

            self.adapter.load_state_dict(
                load_file(hf_hub_download(repo, ckpt), device=self.device)
            )
            self.adapter.eval()
            self.pipe = AnimateDiffPipeline.from_pretrained(
                base, motion_adapter=self.adapter, torch_dtype=self.dtype
            ).to(self.device)
            self.pipe.unet.eval()
            self.pipe.unet = ipex.optimize(self.pipe.unet)
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
                beta_schedule="linear",
            )
            self.logger.info("AnimateDiff model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize AnimateDiff model: {str(e)}")
            raise

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info(
                f"Starting animation generation for job {params.get('job_id')}"
            )
            output = self.pipe(
                prompt=params["prompt"],
                guidance_scale=params.get("guidance_scale", 1.0),
                num_inference_steps=params.get("num_inference_steps", self.step),
                num_frames=params.get("num_frames",  32),
            )
            return output.frames[0]
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "AnimateDiff",
            "device": self.device,
            "dtype": str(self.dtype),
        }
