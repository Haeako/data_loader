import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import numpy as np  # Nếu chưa có: pip install cnpy==1.0.0 (hoặc dùng np.savez thay thế)
import sys

class SimplePipeline(Pipeline):
    def __init__(self, image_path, batch_size=1, num_threads=1, device_id=0):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.input = fn.readers.file(file_root=".", files=[image_path])
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.resize = fn.resize(resize_x=640, resize_y=640)
        self.cmnorm = fn.crop_mirror_normalize(
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )

    def define_graph(self):
        raw, _ = self.input(name="Reader")
        decoded = self.decode(raw)
        resized = self.resize(decoded)
        output = self.cmnorm(resized)
        return output

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    output_npz = sys.argv[2] if len(sys.argv) > 2 else "output.npz"

    pipe = SimplePipeline(image_path)
    pipe.build()
    output = pipe.run()[0]  # [0] to get the output tensor
    arr = output.as_cpu().as_array()

    print(f"Image shape after preprocessing: {arr.shape}")  # e.g., (1, 3, 640, 640)
    np.savez(output_npz, image=arr)
    print(f"Saved to {output_npz}")
