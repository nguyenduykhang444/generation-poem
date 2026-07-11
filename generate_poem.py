import torch
import random 
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

# IMPORT CLASS TỪ FILE VỪA TẠO
from lucbat_processor import StrictLucBatProcessor

# ==========================================
# KHỞI TẠO VÀ SINH THƠ
# ==========================================
model_path = "./qwen-lucbat-model" 
print("Đang tải Tokenizer và Mô hình...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

# --- CHỌN NGẪU NHIÊN SỐ DÒNG ---
target_lines = random.choice([4, 6, 8, 10])
print(f"\n=> AI đã được lệnh sáng tác ngẫu nhiên một bài thơ dài đúng {target_lines} câu!")

# GỌI CLASS ÉP LUẬT
processor = StrictLucBatProcessor(tokenizer, max_total_lines=target_lines)
logits_processor_list = LogitsProcessorList([processor])

prompt = "Đồng Tháp"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Đang sáng tác thơ...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300, 
        logits_processor=logits_processor_list,
        do_sample=True,
        temperature=0.85,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1
    )

print("\n" + "="*40 + "\nBÀI THƠ HOÀN CHỈNH\n" + "="*40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("="*40 + "\n")