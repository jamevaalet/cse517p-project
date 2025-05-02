# Hardware Requirements for Training

The character-level LSTM model can be trained on various hardware configurations. Here are the recommended specifications based on dataset size and expected training time:

## Minimum Requirements
- **CPU**: 4+ core CPU (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Training Time**: Slow (hours to days depending on dataset size)

## Recommended Configuration
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CPU**: 8+ core CPU
- **RAM**: 16GB+
- **Storage**: 50GB SSD
- **Training Time**: Medium (minutes to hours)

## Optimal Configuration
- **GPU**: NVIDIA RTX 4080/4090 or A100/H100 series
- **CPU**: 16+ core CPU
- **RAM**: 32GB+
- **Storage**: 100GB+ NVMe SSD
- **Training Time**: Fast (minutes)

## Notes
1. **GPU vs CPU**: Training on GPU is highly recommended and will be 10-50x faster than CPU-only training
2. **CUDA Compatibility**: The Docker image uses CUDA 12.4, so ensure your NVIDIA drivers support this version
3. **Multi-GPU**: This implementation supports single-GPU training, but could be modified for multi-GPU

## Cloud Options
If local hardware is insufficient, consider these cloud alternatives:
- **Google Colab**: Free option with T4 GPU (limited runtime)
- **Google Colab Pro**: More GPU time and better GPUs (V100)
- **AWS EC2**: p3.2xlarge or g4dn.xlarge instances
- **Azure**: NC series VMs
- **GCP**: N1 with NVIDIA T4 instances

To check if your GPU is being used during training, look for CUDA initialization messages in the output, or add this code to the training loop:
```python
print(f"Using device: {model.device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```
