import os
import json
import subprocess
import sys
import math

# ==========================================
# 1. CẤU HÌNH QLORA & THAM SỐ
# ==========================================
base_model_id = "Qwen/Qwen3.5-0.8B-Base" 
quantized_path = "mlx_quantized_0.8B" 
adapter_path = "mlx_adapter"            
final_model_path = "./qwen-lucbat-qlora" 
data_dir = "mlx_data"               

num_epochs = 3      
batch_size = 8      
# Lược bỏ lora_rank vì MLX bản mới tự động tối ưu rank cho chip M

# ==========================================
# 2. XỬ LÝ VÀ CHIA TÁCH DỮ LIỆU (TRAIN / VALID)
# ==========================================
print("Đang chuẩn bị và làm sạch dữ liệu cho MLX...")
os.makedirs(data_dir, exist_ok=True)

with open("train.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Lọc các bài thơ hợp lệ
poems = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 10]
num_samples = len(poems)

if num_samples == 0:
    print("[LỖI] Tập train.txt trống hoặc không có bài thơ nào hợp lệ!")
    sys.exit(1)

# Tách riêng tập Validation (10% dữ liệu, tối đa 20 bài) để tránh rò rỉ dữ liệu
val_size = min(20, max(1, num_samples // 10))
train_poems = poems[val_size:]
valid_poems = poems[:val_size]

# Lưu tập Train
with open(os.path.join(data_dir, "train.jsonl"), "w", encoding="utf-8") as f:
    for poem in train_poems:
        json.dump({"text": poem}, f, ensure_ascii=False)
        f.write("\n")

# Lưu tập Valid
with open(os.path.join(data_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
    for poem in valid_poems:
        json.dump({"text": poem}, f, ensure_ascii=False)
        f.write("\n")

# ==========================================
# 3. ÉP MÔ HÌNH XUỐNG 4-BIT (QUANTIZATION)
# ==========================================
if not os.path.exists(quantized_path):
    print("Đang lượng tử hóa (Quantize) mô hình xuống 4-bit...")
    quantize_cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", base_model_id,
        "--mlx-path", quantized_path,
        "--quantize", "--q-bits", "4"
    ]
    subprocess.run(quantize_cmd, check=True)

# ==========================================
# 4. HUẤN LUYỆN QLORA VỚI MLX (TỐI ƯU CHIP M)
# ==========================================
total_iters = math.ceil((len(train_poems) * num_epochs) / batch_size)
save_every = max(50, total_iters // 5)

print("\n🚀 BẮT ĐẦU HUẤN LUYỆN QLORA...")
mlx_train_cmd = [
    sys.executable, "-m", "mlx_lm", "lora",  
    "--model", quantized_path,           
    "--train",
    "--data", data_dir,
    "--batch-size", str(batch_size),
    "--num-layers", "12",               
    "--iters", str(total_iters),         
    "--max-seq-length", "256",           # [QUAN TRỌNG] Chặn cấp phát thừa RAM ảo
    "--learning-rate", "2e-4",
    "--adapter-path", adapter_path,
    "--save-every", str(save_every),
    "--grad-checkpoint"                  # [QUAN TRỌNG] Cân bằng VRAM trên Apple Silicon
]
subprocess.run(mlx_train_cmd, check=True)

# ==========================================
# 5. GỘP MÔ HÌNH (FUSE)
# ==========================================
print("\n📦 ĐANG GỘP QLORA ADAPTER VÀO MÔ HÌNH...")
mlx_fuse_cmd = [
    sys.executable, "-m", "mlx_lm.fuse",
    "--model", quantized_path,
    "--adapter-path", adapter_path,
    "--save-path", final_model_path
]
subprocess.run(mlx_fuse_cmd, check=True)
print("\n✅ HOÀN TẤT!")