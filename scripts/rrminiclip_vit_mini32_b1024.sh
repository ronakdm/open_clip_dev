CUDA_VISIBLE_DEVICES=1 python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/mnt/ssd/ronak/datasets/imagenet_captions/imagenet_captions_train_c10.csv"  \
    --val-data="/mnt/ssd/ronak/datasets/imagenet_captions/imagenet_captions_val_c10.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 100 \
    --batch-size=1024 \
    --log-every-n-steps 50 \
    --lr=1e-5 \
    --wd=0.01 \
    --epochs=40 \
    --workers=8 \
    --model ViT-Mini-32 \
    --name "RR-ViT-Mini-32-ImageNet-Captions-C10-b1024" \
    --quantization="/mnt/ssd/ronak/datasets/imagenet_captions/quantization/vit_b32_laion2b_kmeans_50" \
    --use-raking \
    --save-most-recent \
    --delete-previous-checkpoint
    # --imagenet-val="/mnt/ssd/ronak/datasets/imagenet_captions/imagenet_validation" \
    # --pretrained="/mnt/ssd/ronak/models/laion2b_s34b_b79k.bin" \