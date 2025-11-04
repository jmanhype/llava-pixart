# llava-pixart

An AI image generation tool using PixArt-Alpha to create stunning, high-quality images from text prompts.

## Features

- Batch generation of multiple images from a single prompt
- Random seed generation for varied outputs
- GPU acceleration with CPU offloading support
- High-quality 1024x1024 image generation

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- ~10GB disk space for model weights

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llava-pixart.git
cd llava-pixart
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script to generate 5 images with the default prompt:
```bash
python dif.py
```

### Command-Line Arguments

The script supports various command-line arguments for customization:

```bash
python dif.py --prompt "A beautiful sunset over mountains" \
              --num-images 3 \
              --output-prefix "sunset" \
              --output-dir "./outputs" \
              --verbose
```

**Available Arguments:**

- `--prompt`: Text description for image generation (default: cosmic scene)
- `--num-images`: Number of images to generate (default: 5)
- `--output-prefix`: Prefix for output filenames (default: "generated")
- `--output-dir`: Directory to save images (default: current directory)
- `--model`: HuggingFace model identifier (default: PixArt-alpha/PixArt-XL-2-1024-MS)
- `--verbose`: Enable verbose logging

### Help

View all available options:
```bash
python dif.py --help
```

### Output

Images are saved as PNG files with sequential numbering (e.g., `generated_0.png`, `generated_1.png`, etc.).

## Model Information

This project uses the PixArt-Alpha/PixArt-XL-2-1024-MS model, which generates high-quality 1024x1024 images.

## License

See LICENSE file for details.

## Credits

Built with [PixArt-Alpha](https://github.com/PixArt-alpha/PixArt-alpha) and [Diffusers](https://github.com/huggingface/diffusers).
