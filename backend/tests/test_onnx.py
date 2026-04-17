import onnxruntime as ort
session = ort.InferenceSession("models/weights/arcface.onnx", providers=['CPUExecutionProvider'])
for i in session.get_inputs():
    print(f"Input Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
