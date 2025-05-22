#!/usr/bin/env python3
import os
import time
import argparse
import torch
import pyvips
from collections import OrderedDict
import tensorrt as trt

# --------------------------------------------------------------------------------------------------
# Helpers: Time profiler
# --------------------------------------------------------------------------------------------------
class TimeProfiler:
    def __init__(self):
        self.total = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total += time.time() - self.start

# --------------------------------------------------------------------------------------------------
# Simple stub for TRTInference (replace with your full class)
# --------------------------------------------------------------------------------------------------
class TRTInference:
    def __init__(self, engine_path, device="cuda:0"):
        self.device = device
        # load engine, create context, allocate bindings...
        # For test purpose, we just echo back inputs
    def __call__(self, blob):
        # echo inputs as outputs
        return {"labels": torch.zeros(1,1), "boxes": torch.zeros(1,1,4), "scores": torch.zeros(1,1)}

# --------------------------------------------------------------------------------------------------
# Image preprocessing without NumPy
# --------------------------------------------------------------------------------------------------
def process_image_no_numpy(model, file_path, device, tp_load, tp_infer):
    # 1) Load & normalize via pyvips
    with tp_load:
        im = pyvips.Image.new_from_file(file_path, access="sequential", memory=True)
        # ensure RGB
        if im.bands == 1:
            im = im.colourspace("srgb")
        elif im.bands == 4:
            im = im[:3]
        # normalize to [0,1] float
        im = im.linear(1/255.0, [0] * im.bands)
        buf = im.write_to_memory()  # raw bytes of float32 pixels
        h, w, c = im.height, im.width, im.bands

        # Create tensor directly from memoryview:
        # - we know write_to_memory() gives float32 per pixel channel
        mv = memoryview(buf).cast('f')  # now mv is sequence of h*w*c floats
        tensor = torch.as_tensor(list(mv), dtype=torch.float32, device=device)
        tensor = tensor.view(h, w, c).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    # 2) Inference
    with tp_infer:
        outputs = model({"images": tensor, "orig_target_sizes": torch.tensor([[h, w]], device=device)})
    return outputs

# --------------------------------------------------------------------------------------------------
# Main test loop
# --------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-trt", "--trt_engine", required=False, help="(Unused stub)")
    parser.add_argument("-i", "--input_folder", required=True, help="Folder of PNG images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TRTInference(args.trt_engine, device=device)

    tp_load = TimeProfiler()
    tp_infer = TimeProfiler()

    files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(".png")]
    total_start = time.time()

    for fname in files:
        path = os.path.join(args.input_folder, fname)
        _ = process_image_no_numpy(model, path, device, tp_load, tp_infer)

    total_time = time.time() - total_start
    n = len(files)
    print(f"\nProcessed {n} images in {total_time:.3f}s → {n/total_time:.2f} FPS")
    print(f"Avg load+preprocess: {(tp_load.total/n)*1000:.3f} ms")
    print(f"Avg inference:       {(tp_infer.total/n)*1000:.3f} ms")

if __name__ == "__main__":
    main()
