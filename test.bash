#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 
EXECUTABLE="./sanity_check"  # Thay bằng tên chương trình của bạn
ITERATIONS=10                # Số lần chạy
TOTAL_TIME=0

for ((i=1; i<=ITERATIONS; i++)); do
    echo "Lần chạy $i..."
    START_TIME=$(date +%s.%N)

    # Chạy chương trình với ưu tiên CPU và I/O cao
    /usr/bin/nice -n 0 /usr/bin/ionice -c2 -n0 "$EXECUTABLE" /mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/datasets/fisheye1k

    END_TIME=$(date +%s.%N)
    echo "Thời gian: ${ELAPSED}"
    TOTAL_TIME=$(echo "$TOTAL_TIME + $ELAPSED")
done

AVG_TIME=$(echo "scale=4; $TOTAL_TIME / $ITERATIONS" )
