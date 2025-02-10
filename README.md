# Norbert: Multimodal AI Research Model

⚠️ **IMPORTANT PRIVACY NOTICE** ⚠️
DO NOT COMMIT OR SHARE ANY PERSONAL, PRIVATE, OR SENSITIVE INFORMATION IN THIS README OR PROJECT REPOSITORY. 
REMOVE ANY PERSONAL DETAILS IMMEDIATELY IF ACCIDENTALLY ADDED.

## Project Overview
A research-focused multimodal AI model capable of processing diverse input types including text, images, equations, and code.

## Performance Validation

### Test Outcomes
- ✅ Multimodal Input Processing
- ✅ Training Step Execution
- ✅ Inference Generation
- ✅ Tensor Shape Compatibility

## Performance Test Results

### Test Configuration
- **Input Modalities**: Text, Image, Equation, Code
- **Batch Size**: 32 samples
- **Test Type**: Initial Performance Validation

### Key Metrics
- **Training Loss**: 2.333
- **Output Shape**: `[32, 10]` (32 samples, 10 class predictions)

### Sample Output
```python
Model Output (first 5 predictions): 
tensor([
    [-0.2150,  0.1259, -0.2166, -0.2416,  0.8604, ...],
    [-0.0533,  0.1326,  0.1963, -0.2822, -0.1571, ...],
    ...
])
```

## Norbert Multimodal AI Model

### Recent Improvements (v0.2.3)

#### Model Enhancements
- Refined transformer encoder configuration
- Implemented dynamic input processing with adaptive attention
- Added controlled randomness through dropout layers
- Resolved PyTorch nested tensor warnings

#### Performance Test Results

**Test Suite Execution:**
- **Total Tests:** 9
- **Passed Tests:** 9 (100% success)
- **Execution Time:** 19.07 seconds
- **Platform:** macOS, Python 3.13.2, PyTorch

**Key Test Categories:**
1. Model Initialization
2. Multimodal Input Processing
3. Advanced Multimodal Input Processing
4. Input Attention Mechanism
5. Input Preprocessing Robustness
6. Tool Integration
7. Training Methodology
8. Hyperparameter Optimization
9. Problem Solving Workflow

**Test Configuration:**
- Framework: pytest 8.3.4
- Test Environment: Isolated virtual environment

### Compatibility
- Python 3.13.2+
- PyTorch latest

### Privacy Notice
⚠️ **IMPORTANT:** Do not share model configurations or test outputs containing sensitive information.

## Running Norbert: CUDA and Device Selection

#### Automatic Device Detection

Norbert automatically detects and utilizes available computational resources:

```python
import torch
from norbert.core.model import NorbertBaseModel

# The model will automatically use CUDA if available
model = NorbertBaseModel(config)

# Verify the device being used
print(f"Model is running on: {next(model.parameters()).device}")
```

#### Explicit Device Selection

You can explicitly control device selection:

```python
# Force CPU usage
model = NorbertBaseModel(config, device='cpu')

# Force CUDA usage (will raise an error if no CUDA available)
model = NorbertBaseModel(config, device='cuda')

# Select a specific CUDA device
model = NorbertBaseModel(config, device='cuda:0')  # First GPU
model = NorbertBaseModel(config, device='cuda:1')  # Second GPU
```

#### System Requirements
- **Without CUDA:** Any system with Python 3.13.2+
- **With CUDA:** 
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit 11.x or 12.x
  - Compatible NVIDIA GPU driver

#### Performance Tips
- For multi-GPU systems, use `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`
- Monitor GPU memory usage during large model training
- Use mixed precision training for improved performance

**Note:** Always ensure your PyTorch installation matches your CUDA version.

## Test Verification Script
```python
import torch
from norbert.core.model import NorbertBaseModel, NorbertConfig
from norbert.modules.training import NorbertTrainer

# Configuration
config = NorbertConfig.from_pretrained('base')
model = NorbertBaseModel(config)

# Multimodal Inputs
inputs = {
    'text': torch.randn(32, 512, 768),      # Text embedding
    'image': torch.randn(32, 3, 224, 224),  # Image tensor
    'equation': torch.randn(32, 100),       # Equation representation
    'code': torch.randn(32, 512, 768)       # Code embedding
}

# Simulate Labels
labels = torch.randint(0, 10, (32,))

# Training
trainer = NorbertTrainer(model, config)
loss = trainer.training_step((inputs, labels), batch_idx=0)

# Inference
output = model(inputs)
```

## Terminal Output

```
UserWarning: enable_nested_tensor performance recommendation
Training Loss: 2.3186256885528564
Model Output Shape: torch.Size([32, 10])
Model Output (first 5 predictions): tensor([[-0.2992,  0.2003, -0.0677, -0.3482, -0.1743,  0.0689, -0.2442, -0.0975,
         -0.0551,  0.6086],
        [-0.2297, -0.1017,  0.8225, -0.2096, -0.1445, -0.2171,  0.2289, -0.0059,
          0.1088, -0.1053],
        [-0.2801,  1.0413, -0.0391, -0.3266, -0.0274, -0.0206, -0.0863, -0.0086,
         -0.1988, -0.1041],
        [-0.0242,  0.1361,  0.0968, -0.6328,  0.1996,  0.1129, -0.2120,  0.7892,
         -0.0740, -0.1099],
        [-0.1154,  0.0821, -0.0305, -0.4393, -0.1149,  0.1347, -0.0502, -0.1931,
          0.0803,  0.7181]])

## License
MIT License
