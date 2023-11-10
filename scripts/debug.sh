CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/mnt/ssd/ronak/datasets/imagenet_captions/imagenet_captions_train_c10.csv"  \
    --val-data="/mnt/ssd/ronak/datasets/imagenet_captions/imagenet_captions_val_c10.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 1000 \
    --batch-size=128 \
    --log-every-n-steps 10 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=1 \
    --workers=8 \
    --model ViT-Mini-32 \
    --name "ViT-Mini-32-Raking" \
    --quantization="/mnt/ssd/ronak/datasets/imagenet_captions/quantization/vit_b32_laion2b_kmeans_50" \
    --use-raking \
    --overwrite
    # --imagenet-val="/mnt/ssd/ronak/datasets/imagenet_captions/imagenet_validation" \
    # --pretrained="/mnt/ssd/ronak/models/laion2b_s34b_b79k.bin" \