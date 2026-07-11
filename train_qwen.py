import os
import json
import subprocess
import sys

# ==========================================
# 1. CẤU HÌNH MÔ HÌNH VÀ THAM SỐ
# ==========================================
# Sử dụng bản 1.5B kết hợp MLX sẽ giúp Mac Mini train với tốc độ bàn thờ
base_model_id = "Qwen/Qwen2.5-1.5B" 
data_dir = "mlx_data"                   # Thư mục chứa data dạng JSONL
adapter_path = "mlx_adapter"            # Thư mục lưu trọng số LoRA của MLX
final_model_path = "./qwen-lucbat-model" # Thư mục xuất mô hình chuẩn cuối cùng

# ==========================================
# 2. CHUYỂN ĐỔI DỮ LIỆU SANG CHUẨN MLX (JSONL)
# ==========================================
print("Đang chuẩn bị và làm sạch dữ liệu cho MLX...")
os.makedirs(data_dir, exist_ok=True)

# Đọc toàn bộ nội dung file thay vì đọc từng dòng rời rạc
with open("train.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Tách các bài thơ bằng dấu \n\n (Gộp nhiều câu lục bát lại thành 1 bài trọn vẹn)
poems = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 10]

# Lưu thành tập train.jsonl (Mỗi dòng data giờ đây là MỘT BÀI THƠ hoàn chỉnh)
with open(os.path.join(data_dir, "train.jsonl"), "w", encoding="utf-8") as f:
    for poem in poems:
        json.dump({"text": poem}, f, ensure_ascii=False)
        f.write("\n")

# Lấy 10 bài đầu tiên làm tập validation
with open(os.path.join(data_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
    for poem in poems[:10]:
        json.dump({"text": poem}, f, ensure_ascii=False)
        f.write("\n")

print(f"Đã tạo xong {len(poems):,} khối bài thơ chuẩn.")

# ==========================================
# 3. KÍCH HOẠT HUẤN LUYỆN BẰNG MLX-LM (SIÊU TỐC)
# ==========================================
print("\n🚀 BẮT ĐẦU HUẤN LUYỆN TRÊN APPLE SILICON...")
mlx_train_cmd = [
    sys.executable, "-m", "mlx_lm", "lora",  # Đã sửa cách gọi module
    "--model", base_model_id,
    "--train",
    "--data", data_dir,
    "--batch-size", "2",
    "--num-layers", "8",                     # Đã sửa tên tham số tại đây
    "--iters", "1500",           
    "--learning-rate", "2e-4",
    "--adapter-path", adapter_path,
    "--save-every", "500"
]

try:
    subprocess.run(mlx_train_cmd, check=True)
except subprocess.CalledProcessError:
    print("\n[LỖI] Đã xảy ra lỗi trong quá trình huấn luyện MLX.")
    sys.exit(1)

# ==========================================
# 4. XUẤT NGƯỢC MÔ HÌNH VỀ CHUẨN PYTORCH (HUGGING FACE)
# ==========================================
print("\n📦 ĐANG GỘP (FUSE) MÔ HÌNH ĐỂ TƯƠNG THÍCH VỚI BỘ ÉP LUẬT...")
mlx_fuse_cmd = [
    sys.executable, "-m", "mlx_lm.fuse",
    "--model", base_model_id,
    "--adapter-path", adapter_path,
    "--save-path", final_model_path
]

try:
    subprocess.run(mlx_fuse_cmd, check=True)
    print("\n" + "="*50)
    print(f"🎉 HOÀN TẤT! Mô hình PyTorch cuối cùng đã nằm tại: {final_model_path}")
    print("="*50)
except subprocess.CalledProcessError:
    print("\n[LỖI] Đã xảy ra lỗi khi Fuse mô hình.")
    sys.exit(1)