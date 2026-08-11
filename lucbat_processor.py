import unicodedata

import torch
from transformers import LogitsProcessor

__version__ = "2.0"

HUYEN, SAC, NGA, HOI, NANG = "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"
TONE_MARKS = {HUYEN, SAC, NGA, HOI, NANG}
TRAC_MARKS = {SAC, NGA, HOI, NANG}

VOWELS = set("aeiouyăâêôơư")

CONSONANTS = ["ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
              "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r",
              "s", "t", "v", "x"]

TARGET_TONE = {2: "B", 4: "T", 6: "B", 8: "B"}

WORD_START_MARKS = ("\u0120", "\u2581", " ")

MAX_SYLLABLE_LEN = 7
_LATIN_CACHE = {}

# Luật thơ
# L01 đến L05   luật độ dài
# L06 đến L09   luật thanh điệu tại vị trí chẵn
# L10, L12      luật gieo vần lưng và vần chân
# L11           miễn trừ khi tiếng chưa có nguyên âm
# L13           tiếng thứ 6 và thứ 8 của câu bát khác dấu thanh
# L14           không lặp tiếng liền kề
# L15           tiếng gieo vần không trùng nguyên tiếng vần đích
# L16           tiếng phải nằm trong danh sách tiếng hợp lệ
# L17           tiếng chỉ gồm chữ cái Latin và không dài quá bảy ký tự
#
# Luật điều hướng và chống bế tắc
# Đ01, Đ02      cộng thưởng cho token xuống dòng và token kết thúc
# Đ03           khôi phục token kết thúc khi không còn lựa chọn
# Đ04           mở rộng dần nhóm token ứng viên trước khi nới lỏng luật

TIER_FULL = 0
TIER_SOFT = 1
TIER_RHYME = 2


def _decompose(word):
    # Hàm nền: tách dấu thanh khỏi ký tự, phục vụ mọi luật về thanh điệu và vần
    return unicodedata.normalize("NFD", word.lower())


def has_vowel(word):
    # Phục vụ L02 và L11: xác định một tiếng đã hoàn chỉnh hay còn dở dang
    return any(c in VOWELS for c in _decompose(word) if c not in TONE_MARKS)


def get_tone(word):
    # Phục vụ L06 đến L09: xác định thanh bằng hay thanh trắc
    d = _decompose(word)
    if not any(c in VOWELS for c in d if c not in TONE_MARKS):
        return None
    return "T" if any(c in TRAC_MARKS for c in d) else "B"


def get_tone_mark(word):
    # Phục vụ L13: phân biệt thanh ngang với thanh huyền
    return "huyen" if HUYEN in _decompose(word) else "ngang"


def remove_tones(word):
    # Phục vụ L10, L12 và L15: chuẩn hóa tiếng trước khi so vần
    stripped = "".join(c for c in _decompose(word) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def _is_latin(ch):
    # Phục vụ L17: nhận diện ký tự thuộc bảng chữ cái Latin
    ok = _LATIN_CACHE.get(ch)
    if ok is None:
        ok = unicodedata.name(ch, "").startswith("LATIN")
        _LATIN_CACHE[ch] = ok
    return ok


def is_valid_syllable(word):
    # Phục vụ L17: một tiếng chỉ gồm chữ cái Latin và không quá dài
    if not word or len(word) > MAX_SYLLABLE_LEN:
        return False
    return all(_is_latin(c) for c in word)


def get_rhyme_part(word):
    # Phục vụ L10, L12 và L15: trích phần vần của một tiếng
    w = remove_tones(word.strip())
    for c in CONSONANTS:
        if w.startswith(c):
            return w[len(c):]
    return w


def _bytes_to_unicode():
    # Phục vụ L10 và L12: giải mã từ vựng của bộ tách từ byte level BPE
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def _token_surface(tok):
    # Phục vụ L10 và L12: đưa token thô về chuỗi ký tự thật
    if tok.startswith("\u2581"):
        return " " + tok[1:]
    if all(c in _BYTE_DECODER for c in tok):
        try:
            raw = bytes(_BYTE_DECODER[c] for c in tok)
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return tok


def _continuation_prefixes(tokenizer):
    # Phục vụ L10 và L12: tập tiền tố dùng kiểm tra khả năng hoàn tất vần
    cached = getattr(tokenizer, "_lucbat_cont_prefixes", None)
    if cached is not None:
        return cached

    prefixes = set()
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    for tok in vocab:
        if not tok or tok[0] == "<":
            continue
        surface = _token_surface(tok)
        if not surface or surface[0] in WORD_START_MARKS or surface[0] == "\n":
            continue
        t = remove_tones(surface)
        if not t or not t[0].isalpha():
            continue
        for k in (1, 2, 3):
            if len(t) >= k:
                prefixes.add(t[:k])
    try:
        setattr(tokenizer, "_lucbat_cont_prefixes", prefixes)
    except Exception:
        pass
    return prefixes


def load_syllables(path, min_count=1):
    # Phục vụ L16: dựng danh sách tiếng hợp lệ từ kho ngữ liệu huấn luyện
    counter = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for w in line.split():
                w = w.strip().lower()
                if w:
                    counter[w] = counter.get(w, 0) + 1
    return {w for w, c in counter.items() if c >= min_count}


class StrictLucBatProcessor(LogitsProcessor):

    def __init__(self, tokenizer, max_total_lines=2,
                 candidate_pool=(100, 400, 1600), end_bonus=25.0,
                 rhyme_bonus=3.0, syllables=None):
        # Chuẩn bị tài nguyên dùng chung cho toàn bộ tập luật
        self.tokenizer = tokenizer
        self.max_total_lines = max_total_lines
        self.candidate_pool = candidate_pool
        self.end_bonus = end_bonus
        self.rhyme_bonus = rhyme_bonus
        self.syllables = syllables

        self.eos_token_id = tokenizer.eos_token_id
        self.newline_ids = self._collect_newline_ids(tokenizer)
        if not self.newline_ids:
            raise ValueError("Không xác định được token xuống dòng của bộ tách từ")
        self.nl_token_id = self.newline_ids[0]

        self.cont_prefixes = _continuation_prefixes(tokenizer)
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.special_ids.discard(None)
        self._token_cache = {}

    @staticmethod
    def _collect_newline_ids(tokenizer):
        # Phục vụ L01 đến L04 và Đ01 đến Đ03: xác định token kết thúc dòng
        ids = []
        for probe in ("\n", "\n\n", "a\n", " \n"):
            try:
                enc = tokenizer.encode(probe, add_special_tokens=False)
            except Exception:
                continue
            for tid in enc:
                if tid in ids:
                    continue
                try:
                    if "\n" in tokenizer.decode([tid]):
                        ids.append(tid)
                except Exception:
                    pass
        for tok in ("\n", "\u010a", "<0x0A>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                if tid != getattr(tokenizer, "unk_token_id", None):
                    ids.append(tid)
        return ids

    def _tok_str(self, token_id):
        # Hàm nền: chuỗi ký tự của token, bỏ qua token đặc biệt
        s = self._token_cache.get(token_id)
        if s is None:
            s = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self._token_cache[token_id] = s
        return s

    def _can_reach(self, word, target_rhyme):
        # Phục vụ L10 và L12: tiếng đang dở còn khả năng đi tới vần đích hay không
        cur = get_rhyme_part(word)
        if cur == target_rhyme:
            return "exact"
        if not target_rhyme.startswith(cur):
            return None
        needed = target_rhyme[len(cur):]
        for k in (1, 2, 3):
            if needed[:k] and needed[:k] in self.cont_prefixes:
                return "prefix"
        return None

    def _target_rhyme(self, pos, ctx):
        # Phục vụ L10 và L12: phần vần đích tại vị trí đang xét
        if pos != 6:
            return None
        return ctx["luc_rhyme"] if ctx["target_len"] == 8 else ctx["bat_rhyme"]

    def _target_word(self, pos, ctx):
        # Phục vụ L15: tiếng vần đích tại vị trí đang xét
        if pos != 6:
            return None
        return ctx["luc_word"] if ctx["target_len"] == 8 else ctx["bat_word"]

    def _finalize_tier(self, pos, word, ctx):
        # Kiểm tra tiếng vừa được chốt: L06 đến L10, L12, L13, L15, L16
        # L06 đến L09: thanh điệu tại vị trí chẵn
        if pos in TARGET_TONE and get_tone(word) != TARGET_TONE[pos]:
            return None
        tier = TIER_FULL

        # L10 và L12: gieo vần lưng và vần chân
        target = self._target_rhyme(pos, ctx)
        if target and get_rhyme_part(word) != target:
            tier = max(tier, TIER_RHYME)

        # L15: không trùng nguyên tiếng vần đích
        same = self._target_word(pos, ctx)
        if same and word.lower() == same.lower():
            tier = max(tier, TIER_SOFT)

        # L16: tiếng phải nằm trong danh sách tiếng hợp lệ
        if self.syllables and word.lower() not in self.syllables:
            tier = max(tier, TIER_SOFT)

        # L13: tiếng thứ 6 và thứ 8 của câu bát khác dấu thanh
        if pos == 8 and ctx["word6"]:
            if get_tone_mark(word) == get_tone_mark(ctx["word6"]):
                tier = max(tier, TIER_SOFT)
        return tier

    def _partial_tier(self, pos, word, ctx):
        # Kiểm tra tiếng đang hình thành: L06 đến L13, L15
        tone = get_tone(word)
        # L11: tiếng chưa có nguyên âm thì bỏ qua kiểm tra
        if tone is None:
            return TIER_FULL
        # L06 đến L09: thanh điệu tại vị trí chẵn
        if pos in TARGET_TONE and tone != TARGET_TONE[pos]:
            return None

        tier = TIER_FULL
        # L10 và L12: gieo vần, cho phép tiền tố còn đi tới vần đích
        target = self._target_rhyme(pos, ctx)
        if target and self._can_reach(word, target) is None:
            tier = max(tier, TIER_RHYME)

        # L15: không trùng nguyên tiếng vần đích
        same = self._target_word(pos, ctx)
        if same and word.lower() == same.lower():
            tier = max(tier, TIER_SOFT)

        # L13: tiếng thứ 6 và thứ 8 của câu bát khác dấu thanh
        if pos == 8 and ctx["word6"] and tone == "B":
            if get_tone_mark(word) == get_tone_mark(ctx["word6"]):
                tier = max(tier, TIER_SOFT)
        return tier

    def _is_exact_rhyme(self, pos, word, ctx):
        # Phục vụ L10 và L12: cộng thưởng cho tiếng khớp vần chính xác
        target = self._target_rhyme(pos, ctx)
        return bool(target) and get_rhyme_part(word) == target

    def __call__(self, input_ids, scores):
        # Điều phối toàn bộ tập luật: L01 đến L17 và Đ01 đến Đ04
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        lines = text.split("\n")
        current_line = lines[-1]
        previous = [l.strip() for l in lines[:-1] if l.strip()]
        completed = len(previous)

        if not previous:
            target_len = 6
        else:
            target_len = 8 if len(previous[-1].split()) == 6 else 6

        words = current_line.split()
        n = len(words)
        is_last = (completed >= self.max_total_lines - 1) and target_len == 8

        ctx = {"target_len": target_len, "luc_rhyme": None, "bat_rhyme": None,
               "luc_word": None, "bat_word": None,
               "word6": words[5] if n >= 6 else None}
        if previous:
            prev_words = previous[-1].split()
            if target_len == 8 and len(prev_words) == 6:
                ctx["luc_word"] = prev_words[5]
                ctx["luc_rhyme"] = get_rhyme_part(prev_words[5])
            if target_len == 6 and len(prev_words) == 8:
                ctx["bat_word"] = prev_words[7]
                ctx["bat_rhyme"] = get_rhyme_part(prev_words[7])

        row = scores[0]
        mask = torch.full_like(row, -float("inf"))
        terminators = [t for t in
                       ([self.eos_token_id] if is_last else self.newline_ids)
                       if t is not None]

        # L01 đến L04, Đ01, Đ02: dòng đã đủ tiếng thì chỉ cho kết thúc
        if n == target_len and words and has_vowel(words[-1]):
            best = max(terminators, key=lambda t: row[t].item())
            mask[best] = row[best] + self.end_bonus
            scores[0] = mask
            return scores

        vocab = row.shape[-1]
        buckets = {TIER_FULL: [], TIER_SOFT: [], TIER_RHYME: []}
        seen = set()

        # Đ04: mở rộng dần nhóm ứng viên
        for k in self.candidate_pool:
            _, indices = torch.topk(row, min(k, vocab))
            for token_id in indices.tolist():
                if token_id in seen:
                    continue
                seen.add(token_id)

                # L01 và L04: token kết thúc chỉ hợp lệ khi dòng đã đủ tiếng
                if token_id in self.special_ids or token_id == self.eos_token_id:
                    continue
                s = self._tok_str(token_id)
                if not s or not s.strip() or "\n" in s:
                    continue

                # L05: không cho vượt quá số tiếng mục tiêu
                cand_words = (current_line + s).split()
                m = len(cand_words)
                if m == 0 or m > target_len:
                    continue

                # L17: tiếng đang hình thành phải là chuỗi chữ cái hợp lệ
                if not is_valid_syllable(cand_words[-1]):
                    continue

                tier = TIER_FULL
                if m > n and n >= 1:
                    t = self._finalize_tier(n, words[-1], ctx)
                    if t is None:
                        continue
                    tier = max(tier, t)
                    # L14: không lặp tiếng liền kề
                    if m >= 2 and cand_words[-1].lower() == cand_words[-2].lower():
                        tier = max(tier, TIER_SOFT)

                t = self._partial_tier(m, cand_words[-1], ctx)
                if t is None:
                    continue
                tier = max(tier, t)

                bonus = self.rhyme_bonus if self._is_exact_rhyme(
                    m, cand_words[-1], ctx) else 0.0
                buckets[tier].append((token_id, bonus))

            if buckets[TIER_FULL]:
                break

        # chọn theo tầng ưu tiên, đầy đủ trước rồi mới nới lỏng
        chosen = buckets[TIER_FULL] or buckets[TIER_SOFT] or buckets[TIER_RHYME]
        if chosen:
            for token_id, bonus in chosen:
                mask[token_id] = row[token_id] + bonus
        else:
            # Đ03: chống bế tắc
            best = max(terminators, key=lambda t: row[t].item())
            mask[best] = 10.0

        scores[0] = mask
        return scores