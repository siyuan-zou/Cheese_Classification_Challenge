export MODEL_NAME="runwayml/stable-diffusion-v1-5"
accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir="./datas/SCARMOZA" \
    --instance_prompt="" \
    --validation_prompt="there is a loaf of <SCARMOZA> with a piece cut out of it where the food is made of <SCARMOZA> cheese " \
    --output_dir="./out/SCARMOZA" \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=5 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --report_to="wandb" \
    --num_validation_images=2 \
    --validation_steps=500 \
    --max_train_steps=3000 

accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir="./datas/EMMENTAL" \
    --instance_prompt="" \
    --validation_prompt="there is a piece of <EMMENTAL> cheese that is on a slate block " \
    --output_dir="./out/EMMENTAL" \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=5 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --report_to="wandb" \
    --num_validation_images=2 \
    --validation_steps=500 \
    --max_train_steps=3000 

accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir="./datas/MIMOLETTE" \
    --instance_prompt="" \
    --validation_prompt="there is a piece of <MIMOLETTE> cheese on a plate with a walnut " \
    --output_dir="./out/MIMOLETTE" \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=5 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --report_to="wandb" \
    --num_validation_images=2 \
    --validation_steps=500 \
    --max_train_steps=3000 
