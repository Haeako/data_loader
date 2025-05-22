# python tools/deployment/export_onnx.py  \
#     --config configs/deim_dfine/deim_hgnetv2_x_coco.yml \
#     --resume /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/weights/normal/output/checkpoint0047.pth \
#     --check --simplify

# trtexec --onnx="/mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/weights/normal/output/checkpoint0047.onnx" --saveEngine="/mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/weights/normal/output/checkpoint0047.engine" --fp16

CUDA_VISIBLE_DEVICES=5 trtexec \
  --onnx=/mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/weights/normal/output/checkpoint0047.onnx \
  --saveEngine=normal.checkpoint0047.engine \
  --explicitBatch \
  –-useDLACore 0\ 
  --fp16 \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:8x3x640x640,orig_target_sizes:8x2 \
  --maxShapes=images:32x3x640x640,orig_target_sizes:32x2 \
  --workspace=4096