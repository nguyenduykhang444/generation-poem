# evaluate_viverse.py
import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from bert_score import BERTScorer

# IMPORT CLASS TỪ FILE VỪA TẠO
from lucbat_processor import StrictLucBatProcessor

# ==========================================
# 1. HÀM CHẤM ĐIỂM LUẬT THƠ (Giữ nguyên các hàm check âm vần ở đây vì nó dùng để ĐÁNH GIÁ, không phải ép)
# ==========================================
TRAC_CHARS = set("áắấéếíóốớúứýảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ")

def get_tone(word):
    word = word.lower()
    has_alpha = any(c.isalpha() for c in word)
    if not has_alpha: return None
    return "T" if any(c in TRAC_CHARS for c in word) else "B"

def remove_tones(word):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in word:
        if c in s1: s += s0[s1.index(c)]
        else: s += c
    return s

def get_rhyme_part(word):
    word = remove_tones(word.lower().strip())
    consonants = ['ngh', 'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr', 
                  'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x']
    for c in consonants:
        if word.startswith(c): return word[len(c):]
    return word 

def is_rhyme(word1, word2):
    if not word1 or not word2: return False
    return get_rhyme_part(word1) == get_rhyme_part(word2)

def evaluate_luc_bat_rules(cau_luc, cau_bat_gen):
    luc_words = cau_luc.strip().split()
    bat_words = cau_bat_gen.strip().split()
    
    score_length = 0.0
    score_tone = 0.0
    score_rhyme = 0.0
    
    if len(bat_words) == 8: score_length = 10.0
        
    if len(bat_words) >= 8:
        tone_score_per_word = 30.0 / 4.0
        if get_tone(bat_words[1]) == "B": score_tone += tone_score_per_word
        if get_tone(bat_words[3]) == "T": score_tone += tone_score_per_word
        if get_tone(bat_words[5]) == "B": score_tone += tone_score_per_word
        if get_tone(bat_words[7]) == "B": score_tone += tone_score_per_word
        
    if len(luc_words) >= 6 and len(bat_words) >= 6:
        if is_rhyme(luc_words[5], bat_words[5]): score_rhyme = 60.0
            
    return score_length + score_tone + score_rhyme, score_length, score_tone, score_rhyme

# ==========================================
# 2. CHẠY KIỂM THỬ VÀ ĐÁNH GIÁ TRÊN TOÀN BỘ TẬP DỮ LIỆU
# ==========================================
DATASET_FILE = "test.txt"
test_data = []
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines) - 1:
    if len(lines[i].split()) == 6 and len(lines[i+1].split()) == 8:
        test_data.append((lines[i], lines[i+1]))
        i += 2
    else: i += 1

# CẮT LẤY 10% TẬP DỮ LIỆU Ở ĐÂY
limit = max(1, int(len(test_data) * 0.1))
test_data = test_data[:limit]

print("Đang tải Tokenizer và Mô hình Sinh thơ...")
model_path = "./qwen-lucbat-model" 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

# GỌI CLASS ÉP LUẬT TỪ FILE NGOÀI (Chỉ ép 2 dòng vì ta đang test sinh câu Bát từ câu Lục)
processor = StrictLucBatProcessor(tokenizer, max_total_lines=2)
logits_processor_list = LogitsProcessorList([processor])

print("Đang khởi tạo BERTScorer (PhoBERT)...")
scorer = BERTScorer(model_type="vinai/phobert-base", num_layers=9, rescale_with_baseline=False)

print("\n" + "="*60)
print(f"BẮT ĐẦU ĐÁNH GIÁ (CÓ BỘ ÉP LUẬT TỔNG HỢP) - TẤT CẢ {len(test_data)} MẪU (10% DỮ LIỆU)")
print("="*60)

total_rule = 0; total_len = 0; total_tone = 0; total_rhyme = 0
generated_bats = []
reference_bats = []

start_time = time.time()

for i, (cau_luc, cau_bat_ref) in enumerate(test_data):
    prompt = cau_luc + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25, 
            logits_processor=logits_processor_list, 
            do_sample=True,          
            temperature=0.8,        
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cau_bat_gen = full_text.replace(cau_luc, "").strip().split("\n")[0].strip()
    
    rule_score, len_s, tone_s, rhyme_s = evaluate_luc_bat_rules(cau_luc, cau_bat_gen)
    
    total_rule += rule_score; total_len += len_s; total_tone += tone_s; total_rhyme += rhyme_s
    generated_bats.append(cau_bat_gen)
    reference_bats.append(cau_bat_ref)
    
    if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
        print(f"Đã xử lý [{i+1}/{len(test_data)}] mẫu...")

print("\nĐang tính toán PhoBERTScore cho toàn bộ tập test (có thể mất vài phút)...")
P, R, F1 = scorer.score(generated_bats, reference_bats)
avg_phobert_f1 = F1.mean().item() * 100 

n = len(test_data)
eval_time = time.time() - start_time

print("\n" + "="*60)
print("BÁO CÁO ĐÁNH GIÁ CHUẨN VIVERSE-A1 (10% DỮ LIỆU)")
print("="*60)
print(f"Tổng số mẫu test:      {n}")
print(f"Thời gian chạy:        {eval_time:.1f} giây (~{eval_time/60:.1f} phút)")
print(f"1. ĐIỂM LUẬT THƠ (Rule-based): {total_rule/n:.1f} / 100")
print(f"   - Độ dài (10%):     {total_len/n:.1f}%")
print(f"   - Thanh điệu (30%): {total_tone/n:.1f}%")
print(f"   - Gieo vần (60%):   {total_rhyme/n:.1f}%")
print(f"2. PHO-BERT SCORE (Semantic) : {avg_phobert_f1:.1f}%")
print("="*60)