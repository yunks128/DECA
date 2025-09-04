# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DECA (Detailed Expression Capture and Animation) is a PyTorch-based 3D face reconstruction system that creates detailed 3D head models from single images. It reconstructs head pose, facial geometry, expressions, and lighting from photographs using the FLAME parametric face model.

## Development Commands

### Running the Server
```bash
# Start the Flask API server (runs on port 5000)
python deca_server.py

# The server provides REST endpoints:
# POST /reconstruct_download - Process image and return ZIP with results
# GET /health - Health check
```

### Running Demos
```bash
# Basic face reconstruction from image
python demos/demo_reconstruct.py -i <input_image> --saveObj --saveImages --saveDepth

# Run with CPU only (if GPU issues)
CUDA_VISIBLE_DEVICES="" python demos/demo_reconstruct.py -i <input> --device cpu

# Expression transfer between faces
python demos/demo_transfer.py
```

### Dependencies Installation
```bash
# Core dependencies (if not already installed)
pip install torch torchvision scikit-image opencv-python kornia yacs face-alignment ninja chumpy scipy trimesh flask

# The system requires significant dependencies and may need compatibility fixes for newer Python/PyTorch versions
```

### Testing
```bash
# Test the server locally
curl -X POST http://localhost:5000/reconstruct_download \
  -F "image=@test_image.jpg" \
  -F "save_obj=true" \
  -F "save_images=true" \
  -o results.zip

# Working example with full parameter set (external IP)
curl -X POST http://34.70.137.109:5000/reconstruct_download \
  -F "image=@Amber.png" \
  -F "save_obj=true" \
  -F "save_kpt=true" \
  -F "save_images=true" \
  -F "save_mat=true" \
  -F "save_depth=true" \
  -o Amber_reconstruction.zip
```

## Architecture Overview

### Core Components

**DECA Class** (`decalib/deca.py`): Main orchestrator that combines all components
- Manages model loading, encoding, and decoding
- Handles different output formats (meshes, images, parameters)
- Single instance pattern used by server for efficiency

**Models** (`decalib/models/`):
- `FLAME.py`: Parametric 3D face model (shape, expression, pose parameters)
- `encoders.py`: ResNet-50 based encoders that extract parameters from images
- `decoders.py`: Detail generators for fine displacement maps

**Rendering** (`decalib/utils/renderer.py`):
- Differentiable renderer with custom CUDA kernels
- Supports texture extraction and UV mapping
- Multiple rasterizer backends (standard_rasterize, pytorch3d)

### Data Flow

1. **Input Processing**: Face detection → cropping → normalization
2. **Encoding**: Image → FLAME parameters (shape, expression, pose, camera, lighting, detail)  
3. **Decoding**: Parameters → 3D mesh + texture + renderings
4. **Output**: OBJ files, depth maps, keypoints, parameter files

### Configuration System

Uses YACS configuration files in `configs/release_version/`:
- `deca_coarse.yml`: Main model configuration
- Model parameters, training settings, loss weights all configurable
- Access via `cfg.model.*`, `cfg.dataset.*`, etc.

## Critical Files and Locations

**Essential Data** (must exist in `data/`):
- `deca_model.tar`: Pre-trained weights (434MB) 
- `generic_model.pkl`: FLAME model parameters
- `head_template.obj`: Base mesh topology
- UV mapping and texture files

**Server** (`deca_server.py`):
- Flask API with global model instance
- Handles file uploads, processing, and ZIP generation
- Comprehensive error handling with fallbacks

**Custom CUDA Code** (`decalib/utils/rasterizer/`):
- `standard_rasterize_cuda_kernel.cu`: Custom rasterization kernels
- Compiled at runtime via PyTorch JIT
- May need compatibility fixes for newer PyTorch versions

## Common Issues and Fixes

### PyTorch/CUDA Compatibility
- CUDA kernel compilation may fail with newer PyTorch versions
- Fix: Update `AT_DISPATCH_FLOATING_TYPES(tensor.type(), ...)` to `AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), ...)`
- The chumpy library may need NumPy compatibility fixes for newer versions

### Memory Management
- Model loads ~434MB into GPU memory
- Server uses single global instance to avoid reloading overhead
- Clean up temporary files in `/tmp/` for API requests

### Dependencies
- `face-alignment` required for face detection but not in requirements.txt
- `ninja` required for CUDA compilation
- Some packages may need specific versions for compatibility

## Development Patterns

### Model Initialization
```python
# Standard pattern used throughout codebase
device = 'cuda' if torch.cuda.is_available() else 'cpu'
deca_cfg = cfg.copy()
deca = DECA(config=deca_cfg, device=device)
```

### Error Handling
```python
# Server pattern for robust error handling
try:
    # Process image
    codedict = deca.encode(images)
    opdict, visdict = deca.decode(codedict)
except Exception as e:
    # Log error, clean up temp files, return meaningful error
    return jsonify({"error": str(e)}), 500
```

### Configuration Access
```python
# Access nested config values
model_cfg = cfg.model
flame_cfg = cfg.model.flame_config
device = cfg.device
```

This codebase requires careful dependency management and may need compatibility fixes when working with newer Python/PyTorch versions. The server architecture is designed for production use with proper error handling and resource management.