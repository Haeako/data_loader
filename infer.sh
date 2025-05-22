# python predict_torch.py --config configs/deim_dfine/deim_hgnetv2_x_coco.yml \
#     --resume /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/weights/sync_excludeM/output/checkpoint0027.pth \
#     --input_dir /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/sample \
#     --device cuda:4 \
#    --visualize \
#    --visualize_dir visualize/e27_sync_excl

python predict_trt.py \
    --trt /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/weights/normal/output/checkpoint0047.engine \
    --input_dir /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/sample \
    --device cuda:4 \
   --visualize \
   --visualize_dir visualize/e27_sync_excl