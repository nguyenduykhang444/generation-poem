import json
import re

# ==========================================
# 1. ĐỊNH NGHĨA LUẬT NGỮ ÂM TIẾNG VIỆT
# ==========================================
# Bảng phân loại dấu thanh
TONE_MAP = {
    'huyền': 'àằầèềìòồờùừỳ',
    'sắc': 'áắấéếíóốớúứý',
    'hỏi': 'ảẳẩẻểỉỏổởủửỷ',
    'ngã': 'ãẵẫẽễĩõỗỡũữỹ',
    'nặng': 'ạặậẹệịọộợụựỵ'
}

def get_tone_info(word):
    """Xác định Thanh (B/T) và Dấu thanh cụ thể của từ."""
    word_lower = word.lower()
    for sub_tone, chars in TONE_MAP.items():
        # Nếu từ chứa bất kỳ ký tự nào có dấu
        if any(char in word_lower for char in chars):
            tone = 'B' if sub_tone == 'huyền' else 'T'
            return tone, sub_tone
    # Nếu không chứa dấu nào ở trên -> Âm ngang (Bằng)
    return 'B', 'ngang'

def get_rhyme(word):
    """Tách vần của từ bằng cách bỏ phụ âm đầu và dấu thanh."""
    word_lower = word.lower()
    
    # Danh sách các phụ âm đầu trong Tiếng Việt
    consonants_pattern = r'^(qu|gi|tr|ch|ph|nh|kh|th|ngh|ng|gh|b|c|d|đ|g|h|k|l|m|n|p|q|r|s|t|v|x)'
    
    # Loại bỏ phụ âm đầu để lấy phần vần (có chứa dấu)
    rhyme_part = re.sub(consonants_pattern, '', word_lower)
    
    # Fallback trường hợp từ bị lỗi chỉ chứa phụ âm
    if not rhyme_part:
        rhyme_part = word_lower
        
    # Bảng mapping để loại bỏ dấu thanh, đưa vần về dạng gốc (VD: "ắng" -> "ang")
    s1 = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
    s0 = 'aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd'
    trans = str.maketrans(s1, s0)
    
    return rhyme_part.translate(trans)

# ==========================================
# 2. ĐỌC DỮ LIỆU TỪ TẬP TRAIN.TXT
# ==========================================
print("Đang đọc tập train.txt để trích xuất từ vựng...")
vocab = set()
with open('train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        # Tách dòng thành các chữ
        words = line.strip().split()
        for w in words:
            # Loại bỏ các dấu câu dính sát vào chữ (VD: "thương," -> "thương")
            w_clean = re.sub(r'[^\w]', '', w)
            # Chỉ giữ lại các chữ chứa chữ cái, bỏ qua số
            if w_clean and not w_clean.isdigit():
                vocab.add(w_clean.lower())

print(f"Tìm thấy {len(vocab):,} từ vựng độc nhất.")

# ==========================================
# 3. PHÂN TÍCH NGỮ ÂM VÀ LƯU DICTIONARY
# ==========================================
print("Đang phân tích thanh điệu và gieo vần...")
phonetic_dict = {}

for word in vocab:
    tone, sub_tone = get_tone_info(word)
    rhyme = get_rhyme(word)
    
    phonetic_dict[word] = {
        'tone': tone,
        'sub_tone': sub_tone,
        'rhyme': rhyme
    }

# Lưu kết quả ra file JSON
with open('text_phonetic_dict.json', 'w', encoding='utf-8') as f:
    json.dump(phonetic_dict, f, ensure_ascii=False, indent=4)

print("Hoàn tất! Đã lưu từ điển tại 'text_phonetic_dict.json'")