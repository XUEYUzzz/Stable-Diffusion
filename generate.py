import os
import datetime
os.environ["HF_HOME"] = "E:/huggingface_cache"  # 模型缓存路径

from diffusers import StableDiffusionXLPipeline
import torch

# 使用英文提示词（模型才能理解！）
PROMPT = "a mystical forest with glowing mushrooms, fantasy illustration, masterpiece, best quality, intricate details"
NEGATIVE_PROMPT = "blurry, low quality, text, watermark, cartoon"

SEED = 42
HEIGHT = 768
WIDTH = 768
STEPS = 30
GUIDANCE_SCALE = 7.5

# 🔽 自动创建带时间戳的保存路径（避免覆盖）
output_dir = "E:/智能插画设计/outputs"  # ← 修改为你想保存的文件夹
os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(output_dir, f"sdxl_{timestamp}.png")

print("🔄 正在加载 SDXL 模型...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
print("✅ 模型已加载到 GPU 上！")

# 生成图像
generator = torch.Generator(device="cuda").manual_seed(SEED)
image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
).images[0]

# 保存结果
image.save(SAVE_PATH)
print(f"🎉 图片已成功保存至: {SAVE_PATH}")
print(f"📦 模型缓存位置: {os.environ['HF_HOME']}")