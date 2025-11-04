"""
PixArt-Alpha Image Generation Script

This script generates high-quality images using the PixArt-Alpha model.
It supports batch generation with random seeds for varied outputs.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import logging

import torch
from diffusers import PixArtAlphaPipeline
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pipeline(model_name: str = "PixArt-alpha/PixArt-XL-2-1024-MS") -> Optional[PixArtAlphaPipeline]:
    """
    Load the PixArt-Alpha pipeline with error handling.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Initialized pipeline or None if loading fails
    """
    try:
        logger.info(f"Loading model: {model_name}")
        pipe = PixArtAlphaPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.enable_model_cpu_offload()
        logger.info("Model loaded successfully")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def generate_images(
    pipe: PixArtAlphaPipeline,
    prompt: str,
    num_images: int = 5,
    output_prefix: str = "generated",
    output_dir: str = "."
) -> int:
    """
    Generate multiple images from a prompt.

    Args:
        pipe: Initialized PixArt pipeline
        prompt: Text description for image generation
        num_images: Number of images to generate
        output_prefix: Prefix for output filenames
        output_dir: Directory to save images

    Returns:
        Number of successfully generated images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    successful_generations = 0

    for i in range(num_images):
        try:
            # Set a new random seed
            seed = random.randint(0, 1000000)
            generator = torch.manual_seed(seed)

            logger.info(f"Generating image {i+1}/{num_images} with seed {seed}")

            # Generate image
            image = pipe(prompt, generator=generator).images[0]

            # Save image with a unique filename
            output_file = output_path / f"{output_prefix}_{i}.png"
            image.save(output_file)
            logger.info(f"Saved: {output_file}")

            successful_generations += 1

        except Exception as e:
            logger.error(f"Failed to generate image {i+1}: {e}")
            continue

    return successful_generations


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate images using PixArt-Alpha model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Visualize an immense, semi-transparent being, composed entirely of flora and fauna native to an alien rainforest, majestically surfacing from a mirror-like lake of liquid mercury. This occurs under a kaleidoscopic sunset that forms infinite fractals in a sky filled not with stars, but with shimmering, multi-dimensional geometrical patterns in a dazzling array of neon colors reminiscent of an unabridged dream. This cosmic event is being observed by the mysterious shadow figures that live within the astral reflections of the lake's surface on the moon.",
        help="Text prompt for image generation"
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to generate (default: 5)"
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="generated",
        help="Prefix for output filenames (default: generated)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save generated images (default: current directory)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        help="HuggingFace model identifier (default: PixArt-alpha/PixArt-XL-2-1024-MS)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main execution function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load the pipeline
    pipe = load_pipeline(args.model)
    if pipe is None:
        logger.error("Failed to initialize pipeline. Exiting.")
        return 1

    # Generate images
    successful = generate_images(
        pipe=pipe,
        prompt=args.prompt,
        num_images=args.num_images,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir
    )

    logger.info(f"Generated {successful}/{args.num_images} images successfully")

    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
