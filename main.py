import os
from dotenv import load_dotenv
import torch
from diffusers import FluxPipeline

load_dotenv()

hf_token = os.environ.get("HUGGINGFACE_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it in your .env file.")

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, use_auth_token=hf_token)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
