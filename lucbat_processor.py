import unicodedata
from functools import lru_cache

import torch
from transformers import LogitsProcessor

# BỘ TẬP LUẬT THƠ LỤC BÁT
# Nhóm Độ dài
# L01: Dòng hiện tại phải đã đủ số tiếng mục tiêu khi gặp token xuống dòng/kết thúc (Hành động: Loại token).
# L02: Tiếng cuối cùng phải chứa ít nhất một nguyên âm, và dòng chỉ được kết thúc khi tiếng cuối đã hoàn chỉnh (Hành động: Loại token).
# L03: Không được chọn token xuống dòng khi bài thơ đã đủ số dòng và dòng hiện tại là câu bát (Hành động: Loại token).
# L04: Không được chọn token kết thúc khi bài thơ chưa đủ số dòng (Hành động: Loại token).
# L05: Số tiếng sau khi ghép không vượt quá độ dài mục tiêu (Hành động: Loại token).
# Nhóm Thanh điệu
# L06: Tiếng thứ 2 phải mang thanh bằng (Hành động: Loại token).
# L07: Tiếng thứ 4 phải mang thanh trắc (Hành động: Loại token).
# L08: Tiếng thứ 6 phải mang thanh bằng (Hành động: Loại token).
# L09: Tiếng thứ 8 của câu bát phải mang thanh bằng (Hành động: Loại token).
# L13: Tiếng thứ 6 và tiếng thứ 8 của câu bát phải khác dấu thanh (Hành động: Hạ tầng ưu tiên).
# Nhóm Gieo vần
# L10: Đang sinh tiếng thứ 6 câu bát, phần vần phải là tiền tố của phần vần tiếng thứ 6 câu lục phía trên (Hành động: Loại token).
# L11: Bỏ qua kiểm tra vần nếu tiếng đang sinh chưa chứa nguyên âm (Hành động: Không loại token).
# L12: Đang sinh tiếng thứ 6 câu lục, phần vần phải khớp với phần vần của tiếng thứ 8 câu bát phía trên (Hành động: Loại token).
# L15: Tiếng gieo vần không được trùng nguyên tiếng vần đích (Hành động: Hạ tầng ưu tiên).
# Nhóm Từ vựng và Hình thức
# L14: Tiếng mới không được trùng tiếng liền trước trong cùng dòng (Hành động: Hạ tầng ưu tiên).
# L16: Tiếng phải nằm trong danh sách tiếng dựng từ kho ngữ liệu (Hành động: Hạ tầng ưu tiên).
# L17: Tiếng chỉ gồm chữ cái Latin, không dài quá bảy ký tự, bề mặt token chỉ chứa tiếng Việt và dấu cách; tiếng phải đúng cấu trúc âm tiết và tiếng đang dở phải có khả năng hoàn thiện (Hành động: Loại token).
# Nhóm Điều hướng và Chống bế tắc
# Đ01: Dòng đủ tiếng, tiếng cuối hoàn chỉnh, thơ chưa đủ dòng -> Cộng 25 điểm cho token xuống dòng. (Hỗ trợ cộng điểm thưởng khi token khép lại một tiếng trọn vẹn).
# Đ02: Dòng đủ tiếng, tiếng cuối hoàn chỉnh, thơ đã đủ dòng -> Cộng 50 điểm cho token kết thúc. (Hỗ trợ cộng điểm thưởng khi token khép lại một tiếng trọn vẹn).
# Đ03: Quét toàn bộ từ vựng viết nốt tiếng dở dang, nới lỏng theo tầng ưu tiên, khôi phục điểm cho token xuống dòng, kết thúc khi toàn bộ véc-tơ mang điểm âm vô cùng.

HUYEN, SAC, NGA, HOI, NANG = "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"
TONE_MARKS = {HUYEN, SAC, NGA, HOI, NANG}
TRAC_MARKS = {SAC, NGA, HOI, NANG}

# dấu phụ: dấu á, dấu ớ, dấu ơ
BREVE, CIRCUMFLEX, HORN = "\u0306", "\u0302", "\u031b"
MOD_MARKS = {BREVE, CIRCUMFLEX, HORN}

VOWELS = set("aeiouyăâêôơư")

# bảng chữ cái tiếng Việt
VIET_BASE = set("abcdeghiklmnopqrstuvxy") | {"đ"}

# âm đầu
CONSONANTS = ["ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
              "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r",
              "s", "t", "v", "x"]

# phần vần
NUCLEI = {
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "ia", "iê", "iu",
    "oa", "oă", "oe", "oi", "oo", "ôi", "ơi",
    "ua", "uâ", "uă", "ue", "uê", "ui", "uô", "uơ", "uy",
    "ưa", "ưi", "ươ", "ưu", "ya", "yê",
    "iêu", "yêu", "uôi", "ươi", "ươu", "uya", "uyê", "oai", "oay", "oao",
    "oeo", "uây", "uyu", "uai", "uao", "uau", "uay", "uôu",
}

# âm cuối
FINALS = {"", "c", "ch", "m", "n", "ng", "nh", "p", "t"}

TARGET_TONE = {2: "B", 4: "T", 6: "B", 8: "B"}

# ký hiệu dấu cách
WORD_START_MARKS = ("\u0120", "\u2581", " ")

MAX_SYLLABLE_LEN = 7
_LATIN_CACHE = {}
_VIET_CHAR_CACHE = {}

TIER_FULL = 0
TIER_SOFT = 1    # vi phạm luật phụ (Hạ tầng ưu tiên): L13, L14, L15.
TIER_LEX = 2     # đúng cấu trúc tiếng Việt nhưng không có trong danh sách L16 (Hạ tầng ưu tiên).
TIER_RHYME = 3   # sai vần L10, L12.
TIER_TONE = 4    # sai thanh điệu L06 đến L09.
TIER_ORDER = (TIER_FULL, TIER_SOFT, TIER_LEX, TIER_RHYME, TIER_TONE)

# giá trị canh chừng cho bộ nhớ đệm, phân biệt chưa tính với kết quả None
_MISS = object()

@lru_cache(maxsize=100000)
def _decompose(word):
    return unicodedata.normalize("NFD", word.lower())


@lru_cache(maxsize=100000)
def normalize(word):
    return unicodedata.normalize("NFC", word.strip().lower())


@lru_cache(maxsize=100000)
def has_vowel(word):
    # Hỗ trợ L02: Tiếng cuối cùng phải chứa ít nhất một nguyên âm.
    return any(c in VOWELS for c in _decompose(word) if c not in TONE_MARKS)


@lru_cache(maxsize=100000)
def get_tone(word):
    d = _decompose(word)
    if not any(c in VOWELS for c in d if c not in TONE_MARKS):
        return None
    return "T" if any(c in TRAC_MARKS for c in d) else "B"


@lru_cache(maxsize=100000)
def get_tone_mark(word):
    return "huyen" if HUYEN in _decompose(word) else "ngang"


@lru_cache(maxsize=100000)
def get_tone_char(word):
    for c in _decompose(word):
        if c in TONE_MARKS:
            return c
    return ""


@lru_cache(maxsize=100000)
def remove_tones(word):
    stripped = "".join(c for c in _decompose(word) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def is_viet_char(ch):
    # Hỗ trợ L17: Đảm bảo tiếng chỉ gồm chữ cái Latin hợp lệ
    ok = _VIET_CHAR_CACHE.get(ch)
    if ok is None:
        d = unicodedata.normalize("NFD", ch.lower())
        ok = bool(d) and d[0] in VIET_BASE and all(
            m in TONE_MARKS or m in MOD_MARKS for m in d[1:])
        _VIET_CHAR_CACHE[ch] = ok
    return ok


def is_viet_surface(text):
    # Phục vụ L17: bề mặt của token chỉ gồm chữ cái tiếng Việt và dấu cách
    return bool(text) and all(c == " " or is_viet_char(c) for c in text)


@lru_cache(maxsize=100000)
def is_valid_syllable(word):
    if not word or len(word) > MAX_SYLLABLE_LEN:
        return False
    return all(is_viet_char(c) for c in word)


@lru_cache(maxsize=100000)
def split_syllable(word):
    # Phục vụ L17: tách tiếng thành âm đầu, phần vần, âm cuối
    w = remove_tones(word.lower())
    onset = ""
    for c in CONSONANTS:
        if w.startswith(c) and len(w) > len(c):
            onset = c
            break
    rest = w[len(onset):]
    i = 0
    while i < len(rest) and rest[i] in VOWELS:
        i += 1
    return onset, rest[:i], rest[i:]


@lru_cache(maxsize=100000)
def has_viet_structure(word):
    # Phục vụ L17: kiểm tra cấu trúc âm tiết tiếng Việt
    if not is_valid_syllable(word):
        return False
    _, nucleus, coda = split_syllable(word)
    return bool(nucleus) and nucleus in NUCLEI and coda in FINALS


@lru_cache(maxsize=100000)
def is_structure_prefix(word):
    # Phục vụ L17: tiếng đang dở còn có thể hoàn tất thành tiếng hợp lệ
    if not is_valid_syllable(word):
        return False
    onset, nucleus, coda = split_syllable(word)
    if not nucleus:
        head = onset + coda
        return any(c.startswith(head) for c in CONSONANTS)
    if not any(n.startswith(nucleus) for n in NUCLEI):
        return False
    return any(f.startswith(coda) for f in FINALS)


@lru_cache(maxsize=100000)
def get_rhyme_part(word):
    w = remove_tones(word.strip())
    for c in CONSONANTS:
        if w.startswith(c):
            return w[len(c):]
    return w


_INDEX_CACHE = {}


def get_indices(syllables, cont_surfaces, tag):
    key = (id(syllables), len(syllables), tag)
    hit = _INDEX_CACHE.get(key)
    if hit is not None:
        return hit[1:]
    prefix_index, base_index = build_prefix_index(syllables)
    variants = build_variant_index(syllables)
    reach = build_reach_set(syllables, cont_surfaces)
    _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = (syllables, prefix_index, base_index, variants, reach)
    return prefix_index, base_index, variants, reach


def build_prefix_index(syllables):
    index = {}
    base = {}
    for s in syllables:
        for i in range(1, len(s) + 1):
            index.setdefault(s[:i], set()).add(s)
        b = remove_tones(s)
        for i in range(1, len(b) + 1):
            key = b[:i]
            if base.get(key, 0) < len(b):
                base[key] = len(b)
    return {k: tuple(v) for k, v in index.items()}, base


def build_reach_set(syllables, cont_surfaces):
    reach = set()
    for s in syllables:
        length = len(s)
        ok = [False] * (length + 1)
        ok[length] = True
        for i in range(length - 1, -1, -1):
            for j in range(i + 1, length + 1):
                if ok[j] and s[i:j] in cont_surfaces:
                    ok[i] = True
                    break
        for i in range(1, length + 1):
            if ok[i]:
                reach.add(s[:i])
    return reach


def build_variant_index(syllables):
    return {(remove_tones(s), get_tone_char(s)) for s in syllables}


def _bytes_to_unicode():
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
    if tok.startswith("\u2581"):
        return " " + tok[1:]
    if all(c in _BYTE_DECODER for c in tok):
        try:
            raw = bytes(_BYTE_DECODER[c] for c in tok)
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return tok


def _vocab_tables(tokenizer):
    cached = getattr(tokenizer, "_lucbat_tables_v2", None)
    if cached is not None:
        return cached

    allowed = []
    prefixes = set()
    surfaces = {}
    starts = set()
    by_surface = {}
    try:
        size = len(tokenizer)
    except Exception:
        size = getattr(tokenizer, "vocab_size", 0)

    step = 4096
    for start in range(0, size, step):
        ids = list(range(start, min(start + step, size)))
        try:
            chunk = tokenizer.batch_decode([[i] for i in ids],
                                           skip_special_tokens=True)
        except Exception:
            chunk = []
            for i in ids:
                try:
                    chunk.append(tokenizer.decode([i],
                                                  skip_special_tokens=True))
                except Exception:
                    chunk.append("")
        for tid, s in zip(ids, chunk):
            # L17: bề mặt token chỉ được gồm chữ cái tiếng Việt và dấu cách
            if not s or not s.strip():
                continue
            if not is_viet_surface(s):
                continue
            # mỗi token chỉ được mang tối đa một tiếng, nhờ ràng buộc này mọi
            # tiếng mới sinh ra đều đi qua đủ các bước kiểm tra
            pieces = s.split()
            if len(pieces) != 1 or len(pieces[0]) > MAX_SYLLABLE_LEN:
                continue
            allowed.append(tid)
            surfaces[tid] = s
            if s[0] in WORD_START_MARKS:
                starts.add(tid)
                continue
            by_surface.setdefault(s, tid)
            t = remove_tones(s)
            for k in (1, 2, 3):
                if len(t) >= k:
                    prefixes.add(t[:k])

    tables = (allowed, prefixes, surfaces, starts, by_surface)
    try:
        setattr(tokenizer, "_lucbat_tables_v2", tables)
    except Exception:
        pass
    return tables


def load_syllables(path, min_count=1): #Dựng danh sách tiếng hợp lệ từ kho ngữ liệu huấn luyện, phục vụ L16
    counter = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for w in line.split():
                w = normalize(w)
                w = "".join(c for c in w if is_viet_char(c))
                if w:
                    counter[w] = counter.get(w, 0) + 1
    return {w for w, c in counter.items()
            if c >= min_count and has_viet_structure(w)}


class StrictLucBatProcessor(LogitsProcessor):

    def __init__(self, tokenizer, max_total_lines=2,
                 candidate_pool=(100, 400, 1600), end_bonus=25.0,
                 rhyme_bonus=3.0, whole_bonus=1.0, syllables=None,
                 strict_lexicon=True, max_candidates=64):
        self.tokenizer = tokenizer
        self.max_total_lines = max_total_lines
        self.candidate_pool = candidate_pool
        self.top_k = max(candidate_pool) if candidate_pool else 1600
        self.max_candidates = max_candidates
        self.end_bonus = end_bonus
        self.rhyme_bonus = rhyme_bonus
        self.whole_bonus = whole_bonus
        self.syllables = syllables or None
        self.strict_lexicon = strict_lexicon

        self.eos_token_id = tokenizer.eos_token_id
        self.newline_ids = self._collect_newline_ids(tokenizer)
        if not self.newline_ids:
            raise ValueError("Không xác định được token xuống dòng của bộ tách từ")
        self.nl_token_id = self.newline_ids[0]

        (allowed, self.cont_prefixes, self.surfaces, self.starts_word,
         self.by_surface) = _vocab_tables(tokenizer)
        # bề mặt của những token nối tiếp, dùng để biết một tiếng đang dở có
        # thật sự viết tiếp được bằng token nào đó hay không
        self.cont_surfaces = set(self.by_surface)
        if self.syllables:
            (self.prefix_index, self.base_index, self.syllable_variants,
             self.reach) = get_indices(self.syllables, self.cont_surfaces,
                                       id(tokenizer))
        else:
            self.prefix_index = None
            self.base_index = None
            self.syllable_variants = None
            self.reach = None
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.special_ids.discard(None)
        self.allowed_ids = [t for t in allowed if t not in self.special_ids]
        # token nối tiếp, tức token không mở đầu một tiếng mới
        self.cont_ids = [t for t in self.allowed_ids if t not in self.starts_word]
        self._allowed_index = None
        self._cont_index = None
        self._lex_cache = {}
        self._prefix_cache = {}
        self._token_cache = {}
        self._prev_ids = None
        self._prev_text = ""
        self._incremental = self._probe_incremental()

    @staticmethod
    def _collect_newline_ids(tokenizer):
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
        s = self.surfaces.get(token_id)
        if s is not None:
            return s
        s = self._token_cache.get(token_id)
        if s is None:
            s = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self._token_cache[token_id] = s
        return s

    def _probe_incremental(self):
        try:
            enc = self.tokenizer.encode("hoa mùa xuân\nem còn đợi",
                                        add_special_tokens=False)
            joined = "".join(self._tok_str(t) for t in enc)
            full = self.tokenizer.decode(enc, skip_special_tokens=True)
            return joined == full
        except Exception:
            return False

    def _decode(self, input_ids):
        row = input_ids[0]
        ids = row.tolist() if hasattr(row, "tolist") else list(row)
        if self._incremental and self._prev_ids is not None:
            k = len(self._prev_ids)
            if len(ids) == k + 1 and ids[:k] == self._prev_ids:
                text = self._prev_text + self._tok_str(ids[-1])
                self._prev_ids, self._prev_text = ids, text
                return text
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        self._prev_ids, self._prev_text = ids, text
        return text

    def _in_lexicon(self, word):
        if word in self.syllables:
            return True
        return (remove_tones(word), get_tone_char(word)) in self.syllable_variants

    def _lex_tier(self, word):
        # L16 và L17: trả về None nghĩa là loại thẳng
        cached = self._lex_cache.get(word, _MISS)
        if cached is not _MISS:
            return cached
        w = normalize(word)
        if not is_valid_syllable(w):
            result = None
        elif self.syllables is not None:
            if self._in_lexicon(w):
                result = TIER_FULL
            elif self.strict_lexicon:
                result = None
            else:
                result = TIER_LEX if has_viet_structure(w) else None
        else:
            # L17: kiểm tra cấu trúc âm tiết nếu không có danh sách
            result = TIER_FULL if has_viet_structure(w) else None
        self._lex_cache[word] = result
        return result

    def _prefix_ok(self, word):
        # L17: tiếng đang dở còn khả năng trở thành một tiếng hợp lệ
        cached = self._prefix_cache.get(word, _MISS)
        if cached is not _MISS:
            return cached
        w = normalize(word)
        if not is_valid_syllable(w):
            ok = False
        elif self._lex_tier(w) is not None:
            # đã là một tiếng trọn vẹn thì đương nhiên hợp lệ
            ok = True
        elif self.prefix_index:
            if w in self.prefix_index:
                # còn tiếng nào bắt đầu bằng chuỗi này và còn viết tiếp được
                ok = self._advanceable(w)
            else:
                # dạng bỏ dấu chỉ được chấp nhận khi còn tiếng dài hơn để
                # viết tiếp, nhờ vậy không sinh ra tiếng thiếu dấu bế tắc
                b = remove_tones(w)
                longest = self.base_index.get(b)
                ok = longest is not None and longest > len(b)
            if not ok and not self.strict_lexicon:
                ok = is_structure_prefix(w)
        else:
            ok = is_structure_prefix(w)
        self._prefix_cache[word] = ok
        return ok

    def _advanceable(self, word):
        """L17: tiếng đang dở phải viết tiếp được bằng một token có thật.

        Nếu không có token nào mang phần còn thiếu thì tiếng đó sẽ mắc kẹt
        giữa chừng và dòng thơ buộc phải kết thúc sớm. Chặn ngay từ đầu vẫn
        hơn là chữa cháy về sau.
        """
        return word in self.reach

    def _extendable(self, word):
        # L02: tiếng đã hoàn chỉnh nhưng vẫn còn là tiền tố của tiếng dài hơn
        if not self.prefix_index:
            return False
        w = normalize(word)
        return w in self.prefix_index and not self._in_lexicon(w)

    def _can_close(self, word):
        # L02: chỉ chốt dòng khi tiếng cuối là một tiếng tiếng Việt trọn vẹn
        tier = self._lex_tier(word)
        if tier is None:
            return False
        if tier == TIER_FULL:
            return True
        return not self._extendable(word)

    # ------------------------------------------------------------------
    # Luật niêm luật
    # ------------------------------------------------------------------
    def _can_reach(self, word, target_rhyme):
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
        if pos != 6:
            return None
        return ctx["luc_rhyme"] if ctx["target_len"] == 8 else ctx["bat_rhyme"]

    def _target_word(self, pos, ctx):
        if pos != 6:
            return None
        return ctx["luc_word"] if ctx["target_len"] == 8 else ctx["bat_word"]

    def _finalize_tier(self, pos, word, ctx):
        # L16, L17: chính tả tiếng Việt, sai là loại thẳng
        tier = self._lex_tier(word)
        if tier is None:
            return None

        # L06 đến L09
        if pos in TARGET_TONE and get_tone(word) != TARGET_TONE[pos]:
            tier = max(tier, TIER_TONE)

        # L10 và L12
        target = self._target_rhyme(pos, ctx)
        if target and get_rhyme_part(word) != target:
            tier = max(tier, TIER_RHYME)

        # L15
        same = self._target_word(pos, ctx)
        if same and word.lower() == same.lower():
            tier = max(tier, TIER_SOFT)

        # L13
        if pos == 8 and ctx["word6"]:
            if get_tone_mark(word) == get_tone_mark(ctx["word6"]):
                tier = max(tier, TIER_SOFT)
        return tier

    def _partial_tier(self, pos, word, ctx):
        # L17: chuỗi đang dở phải còn khả năng thành một tiếng tiếng Việt
        if not self._prefix_ok(word):
            return None

        tone = get_tone(word)
        # L11: tiếng chưa có nguyên âm thì bỏ qua kiểm tra thanh điệu
        if tone is None:
            return TIER_FULL

        tier = TIER_FULL
        # L06 đến L09
        if pos in TARGET_TONE and tone != TARGET_TONE[pos]:
            tier = max(tier, TIER_TONE)

        # L10 và L12
        target = self._target_rhyme(pos, ctx)
        if target and self._can_reach(word, target) is None:
            tier = max(tier, TIER_RHYME)

        # L15
        same = self._target_word(pos, ctx)
        if same and word.lower() == same.lower():
            tier = max(tier, TIER_SOFT)

        # L13
        if pos == 8 and ctx["word6"] and tone == "B":
            if get_tone_mark(word) == get_tone_mark(ctx["word6"]):
                tier = max(tier, TIER_SOFT)
        return tier

    def _is_exact_rhyme(self, pos, word, ctx):
        target = self._target_rhyme(pos, ctx)
        return bool(target) and get_rhyme_part(word) == target

    # ------------------------------------------------------------------
    # Điều hướng
    # ------------------------------------------------------------------
    def _rescue_complete(self, row, mask, words, scores): # Đ03 viết nốt tiếng đang dở trước khi bỏ cuộc
        if not words:
            return None
        last = normalize(words[-1])
        if self._lex_tier(last) is not None:
            return None

        cands = []
        if self.prefix_index:
            n = len(last)
            for syl in self.prefix_index.get(last, ())[:64]:
                if len(syl) <= n:
                    continue
                suffix = syl[n:]
                # nhận cả token viết được trọn phần còn thiếu lẫn token chỉ
                # viết được một phần, miễn là tiếng vẫn còn đường hoàn tất
                for k in range(1, len(suffix) + 1):
                    tid = self.by_surface.get(suffix[:k])
                    if tid is None:
                        continue
                    step = last + suffix[:k]
                    if step == syl or step in self.reach:
                        cands.append(tid)

        if cands:
            idx = torch.as_tensor(sorted(set(cands)), device=row.device)
            best = int(idx[int(torch.argmax(row[idx]))])
        elif self.syllables is not None and self.strict_lexicon:
            # có danh sách tiếng thì không bịa thêm tiếng ngoài danh sách
            return None
        else:
            # quét rộng, chỉ dùng khi không tra được theo chỉ mục
            if self._cont_index is None:
                self._cont_index = torch.as_tensor(self.cont_ids,
                                                   device=row.device)
            values = row[self._cont_index].tolist()
            best, best_score = None, None
            for tid, score in zip(self.cont_ids, values):
                s = self.surfaces.get(tid)
                if not s or " " in s:
                    continue
                cand = last + s
                if len(cand) > MAX_SYLLABLE_LEN:
                    continue
                if not has_viet_structure(cand):
                    continue
                if best_score is None or score > best_score:
                    best, best_score = tid, score
            if best is None:
                return None

        mask[best] = row[best] + self.end_bonus
        scores[0] = mask
        return scores

    def _close_line(self, row, mask, terminators, scores):
        # Đ01, Đ02: chốt dòng bằng token xuống dòng hoặc token kết thúc
        if not terminators:
            terminators = self.newline_ids
        best = max(terminators, key=lambda t: row[t].item())
        mask[best] = row[best] + self.end_bonus
        scores[0] = mask
        return scores

    def __call__(self, input_ids, scores):
        text = self._decode(input_ids)
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

        # L02, Đ01, Đ02: chỉ kết thúc dòng khi tiếng cuối đã hoàn chỉnh
        if n == target_len and words and has_vowel(words[-1]) \
                and self._can_close(words[-1]):
            return self._close_line(row, mask, terminators, scores)

        line_open = bool(current_line) and not current_line[-1].isspace()
        last_word = words[-1] if n else ""

        if n >= 1:
            prev_tier = self._finalize_tier(n, last_word, ctx)
        else:
            prev_tier = TIER_FULL
        prev_lower = last_word.lower()

        if self._allowed_index is None:
            self._allowed_index = torch.as_tensor(self.allowed_ids,
                                                  device=row.device)
        pool = row[self._allowed_index]
        k = min(self.top_k, pool.shape[-1])
        values, positions = torch.topk(pool, k)
        cand_ids = self._allowed_index[positions].tolist()

        buckets = {t: [] for t in TIER_ORDER}
        full = buckets[TIER_FULL]
        surfaces = self.surfaces
        starts_word = self.starts_word

        for token_id in cand_ids:
            s = surfaces[token_id]

            # L05: số tiếng sau khi ghép không vượt quá độ dài mục tiêu
            if not line_open or token_id in starts_word:
                m = n + 1
                if m > target_len:
                    continue
                word = s.strip()
                tier = prev_tier
                if tier is None:
                    continue
                # L14: tiếng mới không được trùng tiếng liền trước
                if n >= 1 and word.lower() == prev_lower:
                    tier = TIER_SOFT if tier < TIER_SOFT else tier
            else:
                m = n
                word = last_word + s
                if len(word) > MAX_SYLLABLE_LEN:
                    continue
                tier = TIER_FULL

            # L17: tiếng đang hình thành phải có khả năng thành tiếng hợp lệ
            t = self._partial_tier(m, word, ctx)
            if t is None:
                continue
            if t > tier:
                tier = t

            bonus = self.rhyme_bonus if self._is_exact_rhyme(
                m, word, ctx) else 0.0
            # Đ01, Đ02 mở rộng: cộng thưởng cho token khép lại một tiếng trọn vẹn
            if self._lex_tier(word) is not None:
                bonus += self.whole_bonus
            buckets[tier].append((token_id, bonus))

            if len(full) >= self.max_candidates:
                break

        # Đ03 mở rộng: nới lỏng theo tầng ưu tiên, không nới luật chính tả
        chosen = []
        for t in TIER_ORDER:
            if buckets[t]:
                chosen = buckets[t]
                break

        if not chosen:
            out = self._rescue_complete(row, mask, words, scores)
            if out is not None:
                return out
            return self._close_line(row, mask, terminators, scores)

        idx = torch.as_tensor([c[0] for c in chosen], device=row.device)
        bonuses = torch.as_tensor([c[1] for c in chosen], device=row.device,
                                  dtype=row.dtype)
        picked = row[idx] + bonuses
        if not bool(torch.isfinite(picked).any()):
            # Đ03: khôi phục điểm cho token xuống dòng hoặc kết thúc
            return self._close_line(row, mask, terminators, scores)
        mask[idx] = picked

        scores[0] = mask
        return scores