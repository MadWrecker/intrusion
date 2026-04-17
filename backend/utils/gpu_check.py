import onnxruntime as ort
import sys

def check_gpu():
    print("=== ONNXRuntime GPU Verification ===")
    device = ort.get_device()
    providers = ort.get_available_providers()
    
    print(f"Device: {device}")
    print(f"Available Providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("\n[SUCCESS] CUDA Execution Provider is available! ONNX models will run on GPU.")
    else:
        print("\n[ERROR] CUDA Execution Provider is NOT available. Falling back to CPU.")
        print("Please ensure you have installed the correct cuDNN and CUDA Toolkit for your mapped onnxruntime-gpu version.")
        sys.exit(1)

if __name__ == "__main__":
    check_gpu()
