export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/ubuntu/3d_text_inversion/gt_images/crochet_octopus"

accelerate launch octopus_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --tokenizer_name= \
  --learnable_property="object" \
  --placeholder_token="<crochet-octopus>" \
  --initializer_token="octopus" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_octopus_2" \
  --resume_from_checkpoint="latest" \
  --num_vectors=5
