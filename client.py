import logging
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnimateDiffClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.output_dir = Path("generated_animations")
        self.output_dir.mkdir(exist_ok=True)

    def check_health(self) -> Dict[str, Any]:
        """Check the health status of the server."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get server information and model configuration."""
        try:
            response = self.session.get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get info: {str(e)}")
            raise

    def generate_animation(
        self,
        prompt: str,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        num_frames: Optional[int] = None,
        save: bool = True,
    ) -> bytes:
        """Generate an animation based on the provided prompt and parameters."""
        try:
            params = {
                "prompt": prompt,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "num_frames": num_frames,
            }
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            response = self.session.post(f"{self.base_url}/generate", params=params)
            response.raise_for_status()

            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"animation_{timestamp}_{prompt[:30]}.gif"
                filepath = self.output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(response.content)
                logger.info(f"Animation saved to {filepath}")

            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Animation generation failed: {str(e)}")
            raise

    def batch_generate(self, prompts: list, max_workers: int = 3) -> Dict[str, bytes]:
        """Generate multiple animations in parallel."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(self.generate_animation, prompt): prompt
                for prompt in prompts
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_prompt),
                total=len(prompts),
                desc="Generating animations",
            ):
                prompt = future_to_prompt[future]
                try:
                    results[prompt] = future.result()
                except Exception as e:
                    logger.error(
                        f"Failed to generate animation for prompt '{prompt}': {str(e)}"
                    )
                    results[prompt] = None

        return results


def main():
    # Initialize client
    client = AnimateDiffClient()

    # Test health and info endpoints
    logger.info("Checking server health...")
    health_status = client.check_health()
    logger.info(f"Health status: {health_status}")

    logger.info("Getting server info...")
    server_info = client.get_info()
    logger.info(f"Server info: {server_info}")

    # Test animation generation with various prompts
    test_prompts = [
        # Nature animations
        "a beautiful cherry blossom tree swaying in the wind, anime style",
        "ocean waves crashing on a beach at sunset, realistic style",
        "snowflakes falling in a winter forest, watercolor style",
        # Character animations
        "a cat playing with a ball of yarn, cartoon style",
        "a dragon breathing fire into the night sky, fantasy style",
        "a ballet dancer performing a pirouette, elegant style",
        # Abstract animations
        "colorful geometric shapes morphing and transforming",
        "flowing liquid metal with rainbow reflections",
        "northern lights dancing in the night sky",
        # Sci-fi animations
        "a futuristic city with flying cars and neon lights",
        "a space station orbiting a beautiful nebula",
        "a robot transforming into a vehicle",
    ]

    # Test single animation generation
    logger.info("Testing single animation generation...")
    client.generate_animation(
        prompt=test_prompts[0], guidance_scale=1.5, num_inference_steps=8, num_frames=32
    )
    # Test batch generation
    logger.info("Testing batch animation generation...")
    results = client.batch_generate(test_prompts)
    # Log success rate
    success_count = sum(1 for result in results.values() if result is not None)
    logger.info(
        f"Batch generation complete. Success rate: {success_count}/{len(test_prompts)}"
    )


if __name__ == "__main__":
    main()
