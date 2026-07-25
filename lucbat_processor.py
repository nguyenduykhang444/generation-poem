import torch
from transformers import LogitsProcessor

class StrictLucBatProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_total_lines=2):
        self.tokenizer = tokenizer
        self.trac_chars = set("áắấéếíóốớúứýảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ")
        self.vowels = set("aeiouyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ")
        self.nl_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
        self.eos_token_id = tokenizer.eos_token_id
        self.max_total_lines = max_total_lines 

    def get_tone(self, text):
        text = text.lower()
        if not any(c in self.vowels for c in text): return None 
        if any(c in self.trac_chars for c in text): return "T"
        return "B"

    def remove_tones(self, word):
        s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
        s = ''
        for c in word:
            if c in s1: s += s0[s1.index(c)]
            else: s += c
        return s

    def get_rhyme_part(self, word):
        word = self.remove_tones(word.lower().strip())
        consonants = ['ngh', 'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr', 
                      'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x']
        for c in consonants:
            if word.startswith(c): return word[len(c):]
        return word

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        text = self.tokenizer.decode(input_ids[0])
        lines = text.split('\n')
        current_line = lines[-1] 
        
        previous_lines = [l.strip() for l in lines[:-1] if l.strip()]
        completed_lines = len(previous_lines)
        
        if not previous_lines: target_length = 6
        else: target_length = 8 if len(previous_lines[-1].split()) == 6 else 6
        
        current_words = current_line.strip().split()
        current_word_count = len(current_words)
        is_finishing_poem = (completed_lines >= self.max_total_lines - 1) and (target_length == 8)

        top_k_val, top_indices = torch.topk(scores[0], 100)
        mask = torch.full_like(scores[0], -float('inf'))

        for token_id in top_indices:
            token_str = self.tokenizer.decode([token_id])
            is_valid = True
            
            candidate_line = current_line + token_str
            candidate_words = candidate_line.strip().split()
            candidate_word_count = len(candidate_words)
            
            # 1. LUẬT ĐỘ DÀI
            if '\n' in token_str or token_id == self.eos_token_id:
                if current_word_count < target_length: is_valid = False 
                elif current_word_count == target_length:
                    if current_words and not any(c in self.vowels for c in current_words[-1].lower()):
                        is_valid = False
                    if is_valid:
                        if is_finishing_poem and '\n' in token_str: is_valid = False 
                        elif not is_finishing_poem and token_id == self.eos_token_id: is_valid = False 
            else:
                if candidate_word_count > target_length: is_valid = False 
                
                # 2. LUẬT THANH ĐIỆU
                if is_valid and candidate_word_count in [2, 4, 6, 8]:
                    target_tone = "B" if candidate_word_count in [2, 6, 8] else "T"
                    tone = self.get_tone(candidate_words[-1])
                    if tone and tone != target_tone: is_valid = False

                # 3. LUẬT GIEO VẦN (MỚI THÊM)
                # Chỉ ép vần khi đang gõ chữ thứ 6 của câu Bát
                if is_valid and candidate_word_count == 6 and target_length == 8 and completed_lines > 0:
                    luc_words = previous_lines[-1].split()
                    if len(luc_words) == 6:
                        target_rhyme = self.get_rhyme_part(luc_words[5])
                        current_word = candidate_words[-1]
                        
                        # Chỉ check vần khi từ này ĐÃ CÓ NGUYÊN ÂM (không chém nhầm phụ âm dở dang)
                        if any(c in self.vowels for c in current_word.lower()):
                            current_rhyme = self.get_rhyme_part(current_word)
                            if current_rhyme and target_rhyme:
                                # Cho phép nếu vần hiện tại là tiền tố của vần đích (đang gõ dở token)
                                if not target_rhyme.startswith(current_rhyme):
                                    is_valid = False
            
            if is_valid: mask[token_id] = scores[0, token_id]

        scores[0] = mask
        
        # ĐIỀU HƯỚNG KẾT THÚC
        if current_word_count == target_length:
            if current_words and any(c in self.vowels for c in current_words[-1].lower()):
                if is_finishing_poem: scores[0, self.eos_token_id] += 50.0 
                else: scores[0, self.nl_token_id] += 25.0  

        if torch.all(scores[0] == -float('inf')):
            if current_word_count >= target_length:
                if is_finishing_poem: scores[0, self.eos_token_id] = 10.0
                else: scores[0, self.nl_token_id] = 10.0
            else: scores[0, self.eos_token_id] = 10.0 

        return scores