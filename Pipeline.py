# animate diff + ip adapter
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, AutoencoderKL
from diffusers.utils import export_to_gif, load_image

device = "cuda"


vae_model_path = "/content/AnimateDiff_IP_Adapter_LoRa/VAE/"

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)


# Загрузка значения переменной из файла
with open('file.json', 'r') as file:
    loaded_variable = json.load(file)
print(loaded_variable) 



pipe = AnimateDiffPipeline.from_pretrained(loaded_variabl, motion_adapter=adapter,vae=vae, torch_dtype=torch.float16)

pipe = pipe.to(device)

# scheduler
scheduler = DDIMScheduler(
    clip_sample=False,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    timestep_spacing="trailing",
    steps_offset=1
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# load ip_adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", adapter_name="zoom-in")



pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-down", adapter_name="tilt-down")





pipe.load_lora_weights("guoyww/animatediff-motion-lora-pan-left", adapter_name="pan-left")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-pan-right", adapter_name="pan-right")


pipe.load_lora_weights("guoyww/animatediff-motion-lora-rolling-clockwise", adapter_name="rolling-clockwise")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-rolling-anticlockwise", adapter_name="rolling-anticlockwise")
