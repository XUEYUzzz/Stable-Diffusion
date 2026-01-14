from safetensors.torch import load_file
import torch

LORA_PATH = r"E:\StableDiffusion\LORA\output\GHIBLI.safetensors"

print("🔍 正在加载 LoRA 文件...")
state_dict = load_file(LORA_PATH)

print(f"✅ 成功加载 {len(state_dict)} 个参数")
print("\n前5个参数名示例:")
for i, key in enumerate(list(state_dict.keys())[:5]):
    print(f"  {i+1}. {key}")

# 检查是否有非零权重
all_zero = all(torch.allclose(v, torch.zeros_like(v), atol=1e-6) for v in state_dict.values())
print(f"\n⚠️ 所有权重为零？{all_zero}")

if not all_zero:
    print("✅ LoRA 包含有效权重！")
else:
    print("❌ LoRA 权重全为零 → 训练失败！")