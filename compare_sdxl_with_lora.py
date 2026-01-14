import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import datetime
import torch
from diffusers import StableDiffusionXLPipeline

# === 配置区（按需修改）===
HF_CACHE_DIR = "E:/huggingface_cache"
OUTPUT_DIR = "E:\StableDiffusion\outputs\PICTURE"
LORA_PATH = "E:\StableDiffusion\LORA\output\GHIBLI.safetensors"

# 提示词（建议用风景类）
PROMPT = "ghibli style, studio ghibli anime, " + \
         "a serene mountain landscape with a crystal-clear lake, pine trees, morning mist, golden hour"
NEGATIVE_PROMPT = "blurry, low quality, text, watermark, cartoon, people, buildings"

SEED = 42
HEIGHT = 1024
WIDTH = 1024
STEPS = 30
GUIDANCE_SCALE = 7.5

# === 初始化 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print("🔄 正在加载 SDXL 基础模型...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=HF_CACHE_DIR
)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
print("✅ 模型加载完成！")

generator = torch.Generator(device="cuda").manual_seed(SEED)

#=== 1. 生成原始图像（无 LoRA）===
print("🖼️ 生成原始图像（无 LoRA）...")
image_original = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
).images[0]

path_original = os.path.join(OUTPUT_DIR, f"sdxl_{timestamp}_original.png")
image_original.save(path_original)
print(f"✅ 原始图像已保存: {path_original}")

# === 2. 注入 LoRA 并生成风格化图像 ===
print(f"🔗 正在加载 LoRA: {LORA_PATH}")
pipe.load_lora_weights(LORA_PATH)  # ⭐ 关键：注入 LoRA
pipe.fuse_lora(lora_scale=2.0)  # ⭐ 融合 LoRA 权重（提升推理速度）
print("Active adapters:", pipe.get_active_adapters())

print("🎨 生成 LoRA 风格图像...")
# 注意：重置 generator seed 保证可比性！
generator = torch.Generator(device="cuda").manual_seed(SEED)
image_lora = pipe(
    prompt=PROMPT+", ghibli style",
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
).images[0]

path_lora = os.path.join(OUTPUT_DIR, f"sdxl_{timestamp}_lora.png")
image_lora.save(path_lora)
print(f"✅ LoRA 图像已保存: {path_lora}")

# === 可选：卸载 LoRA（如果后续还要用原始模型）===
pipe.unfuse_lora()
pipe.unload_lora_weights()

print("\n🎉 对比实验完成！请查看输出文件夹中的两张图片。")