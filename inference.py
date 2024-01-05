from transformers import CLIPTokenizer
from modified_adaptation import ModifiedPLIPEncoder
import torch
import json, tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

pretrained_model = "CompVis/stable-diffusion-v1-4"
logs = "../openpath-50000-model-lora/logs"
output_dir = "../openpath-50000-model-lora/output"
checkpoint = "../openpath-50000-model-lora/checkpoint-50000"
unet = unet = UNet2DConditionModel.from_pretrained(
        pretrained_model, subfolder="unet", low_cpu_mem_usage=False
    )
text_encoder = ModifiedPLIPEncoder.from_pretrained("vinid/plip")
accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logs)

accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="wandb",
        project_config=accelerator_project_config,
    )
accelerator.load_state(checkpoint)

pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=CLIPTokenizer.from_pretrained('../tokenizer'),
                    torch_dtype=torch.float16,
                )
pipeline.to("cuda")
generator = torch.Generator(device=accelerator.device)
images = []

metadata_file = "./test/metadata.jsonl"
with open(metadata_file, "r") as f:
    metadata_list = [json.loads(line) for line in f]

# # For each entry in metadata
for i, metadata in tqdm(enumerate(metadata_list), total=len(metadata_list)):
    # Use the "text" field as prompt
    prompt = metadata["text"]
    
    try:
        # Generate image for the current prompt
        image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        
        # Save the generated image
        output_path = f"../test_out/lora/{metadata['file_name'].replace('.jpg', '')}_out.png"
        image.save(output_path)
    except Exception as e:
        print(f"Error processing entry {i}: {e}")

print("Inference completed.")