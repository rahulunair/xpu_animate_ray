import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

import psutil
import ray
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from ray import serve

from animate_diff import AnimateDiffModel, export_to_gif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_CONFIG = {
    "animate_diff": {
        "default_steps": 4,
        "default_guidance": 1.0,
        "default_frames": 32,
        "output_format": "gif",
    }
}


class ModelStatus:
    def __init__(self):
        self.is_loaded = False
        self.error = None
        self.model = None
        self.last_error_time = None


@serve.deployment(
    ray_actor_options={"num_cpus": 8},
    num_replicas=1,
    max_ongoing_requests=100,
    max_queued_requests=50,
)
@serve.ingress(app)
class AnimationServer:
    def __init__(self):
        self.logger = logging.getLogger("AnimationServer")
        self.model_status = ModelStatus()
        self.output_dir = Path("/output/animations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.logger.info("Initializing AnimateDiff model...")
            self.model_status.model = AnimateDiffModel()
            self.model_status.model.initialize()
            self.model_status.is_loaded = True
            self.model_status.error = None
            self.logger.info("Model initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            self.logger.error(error_msg)
            self.model_status.is_loaded = False
            self.model_status.error = error_msg
            self.model_status.model = None

    @app.get("/info")
    def get_info(self) -> Dict[str, Any]:
        return {
            "model_status": {
                "is_loaded": self.model_status.is_loaded,
                "error": self.model_status.error,
                "config": MODEL_CONFIG["animate_diff"],
            },
            "system_info": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "gpu_memory": torch.xpu.memory_allocated() / 1024**3,
            },
        }

    @app.post("/generate")
    async def generate_animation(
        self,
        prompt: str,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> Response:
        try:
            if not self.model_status.is_loaded:
                self._initialize_model()
                if not self.model_status.is_loaded:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model is not available. Error: {self.model_status.error}",
                    )
            config = MODEL_CONFIG["animate_diff"]
            params = {
                "job_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "prompt": prompt,
                "guidance_scale": guidance_scale or config["default_guidance"],
                "num_inference_steps": num_inference_steps or config["default_steps"],
                "num_frames": num_frames or config["default_frames"],
            }
            frames = self.model_status.model.generate(params)
            output_path = self.output_dir / f"animation_{params['job_id']}.gif"
            export_to_gif(frames, str(output_path))
            with open(output_path, "rb") as f:
                content = f.read()
            gc.collect()
            torch.xpu.empty_cache()
            return Response(content=content, media_type="image/gif")
        except Exception as e:
            self.logger.error(f"Error generating animation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.model_status.is_loaded else "unhealthy",
            "error": self.model_status.error,
            "memory_usage": {
                "cpu": psutil.virtual_memory().percent,
                "gpu": torch.xpu.memory_allocated() / 1024**3,
            },
        }


entrypoint = AnimationServer.bind()
