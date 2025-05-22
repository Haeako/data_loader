# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --master_port=7777 --nproc_per_node=4 train.py \
#  -c /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/configs/deim_dfine/object365/deim_hgnetv2_x_obj2coco_24e.yml \
#  --use-amp --seed=0 \
#  -t /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/models/deim_dfine_hgnetv2_x_obj2coco_24e.pth \
#  --output-dir ./weights/normal/output \
#  --summary-dir ./weights/normal/summary

# python -m torch.distributed.run

CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/deim_dfine/deim_hgnetv2_m_coco.yml --use-amp --seed=0 \
 -t /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/ntmk/DEIM/models/deim_dfine_hgnetv2_m_coco_90e.pth \
  --output-dir ./weights/sync_excludeM_sizeM/output \
 --summary-dir ./weights/sync_excludeM_sizeM/summary 