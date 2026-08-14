import unicodedata

import torch
from transformers import LogitsProcessor

HUYEN, SAC, NGA, HOI, NANG = "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"
TONE_MARKS = {HUYEN, SAC, NGA, HOI, NANG}
TRAC_MARKS = {SAC, NGA, HOI, NANG}

# dấu phụ tạo chữ cái tiếng Việt: trăng khuyết, dấu mũ, dấu móc
BREVE, CIRCUMFLEX, HORN = "\u0306", "\u0302", "\u031b"
MOD_MARKS = {BREVE, CIRCUMFLEX, HORN}

VOWELS = set("aeiouyăâêôơư")

# bảng chữ cái tiếng Việt, không có f, j, w, z
VIET_BASE = set("abcdeghiklmnopqrstuvxy") | {"đ"}

CONSONANTS = ["ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
              "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r",
              "s", "t", "v", "x"]

# phần vần hợp lệ khi đã bỏ dấu thanh và bỏ phụ âm đầu
NUCLEI = {
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "ia", "iê", "iu",
    "oa", "oă", "oe", "oi", "oo", "ôi", "ơi",
    "ua", "uâ", "uă", "ue", "uê", "ui", "uô", "uơ", "uy",
    "ưa", "ưi", "ươ", "ưu", "ya", "yê",
    "iêu", "yêu", "uôi", "ươi", "ươu", "uya", "uyê", "oai", "oay", "oao",
    "oeo", "uây", "uyu", "uai", "uao", "uau", "uay", "uôu",
}
FINALS = {"", "c", "ch", "m", "n", "ng", "nh", "p", "t"}

TARGET_TONE = {2: "B", 4: "T", 6: "B", 8: "B"}

WORD_START_MARKS = ("\u0120", "\u2581", " ")

MAX_SYLLABLE_LEN = 7
_LATIN_CACHE = {}
_VIET_CHAR_CACHE = {}

# Luật thơ
# L01 đến L05   luật độ dài
# L06 đến L09   luật thanh điệu tại vị trí chẵn
# L10, L12      luật gieo vần lưng và vần chân
# L11           miễn trừ khi tiếng chưa có nguyên âm
# L13           tiếng thứ 6 và thứ 8 của câu bát khác dấu thanh
# L14           không lặp tiếng liền kề
# L15           tiếng gieo vần không trùng nguyên tiếng vần đích
# L16           tiếng phải nằm trong danh sách tiếng hợp lệ, luật cứng
# L17           tiếng chỉ gồm chữ cái tiếng Việt và không dài quá bảy ký tự
# L18           token ứng viên chỉ được chứa chữ cái tiếng Việt và dấu cách
# L19           tiếng phải đúng cấu trúc âm đầu, phần vần, âm cuối
# L20           tiếng đang dở phải còn khả năng trở thành một tiếng hợp lệ
# L21           chỉ được kết thúc dòng khi tiếng cuối đã hoàn chỉnh
#
# Luật điều hướng và chống bế tắc
# Đ01, Đ02      cộng thưởng cho token xuống dòng và token kết thúc
# Đ03           khôi phục token kết thúc khi không còn lựa chọn
# Đ04           mở rộng dần nhóm token ứng viên trước khi nới lỏng luật
# Đ05           nới lỏng theo tầng, luật chính tả tiếng Việt không bao giờ nới
# Đ06           quét toàn bộ từ vựng để viết nốt tiếng đang dở trước khi bỏ cuộc
# Đ07           cộng thưởng cho token khép lại một tiếng trọn vẹn

TIER_FULL = 0    # thỏa mọi luật
TIER_SOFT = 1    # vi phạm luật phụ L13, L14, L15
TIER_LEX = 2     # đúng cấu trúc tiếng Việt nhưng không có trong danh sách L16
TIER_RHYME = 3   # sai vần L10, L12
TIER_TONE = 4    # sai thanh điệu L06 đến L09
TIER_ORDER = (TIER_FULL, TIER_SOFT, TIER_LEX, TIER_RHYME, TIER_TONE)


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


def get_tone_char(word):
    # Phục vụ L16: dấu thanh của một tiếng, chuỗi rỗng là thanh ngang
    for c in _decompose(word):
        if c in TONE_MARKS:
            return c
    return ""


def remove_tones(word):
    # Phục vụ L10, L12 và L15: chuẩn hóa tiếng trước khi so vần
    stripped = "".join(c for c in _decompose(word) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def normalize(word):
    # Hàm nền: đưa tiếng về dạng dựng sẵn và chữ thường trước mọi phép so sánh
    return unicodedata.normalize("NFC", word.strip().lower())


def _is_latin(ch):
    # Hàm nền cũ, giữ lại cho khả năng tương thích
    ok = _LATIN_CACHE.get(ch)
    if ok is None:
        ok = unicodedata.name(ch, "").startswith("LATIN")
        _LATIN_CACHE[ch] = ok
    return ok


def is_viet_char(ch):
    # Phục vụ L17 và L18: ký tự phải thuộc bảng chữ cái tiếng Việt
    ok = _VIET_CHAR_CACHE.get(ch)
    if ok is None:
        d = unicodedata.normalize("NFD", ch.lower())
        ok = bool(d) and d[0] in VIET_BASE and all(
            m in TONE_MARKS or m in MOD_MARKS for m in d[1:])
        _VIET_CHAR_CACHE[ch] = ok
    return ok


def is_viet_surface(text):
    # Phục vụ L18: bề mặt của token chỉ gồm chữ cái tiếng Việt và dấu cách
    return bool(text) and all(c == " " or is_viet_char(c) for c in text)


def is_valid_syllable(word):
    # Phục vụ L17: một tiếng chỉ gồm chữ cái tiếng Việt và không quá dài
    if not word or len(word) > MAX_SYLLABLE_LEN:
        return False
    return all(is_viet_char(c) for c in word)


def split_syllable(word):
    # Phục vụ L19 và L20: tách tiếng thành âm đầu, phần vần, âm cuối
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


def has_viet_structure(word):
    # Phục vụ L19: kiểm tra cấu trúc âm tiết tiếng Việt
    if not is_valid_syllable(word):
        return False
    _, nucleus, coda = split_syllable(word)
    return bool(nucleus) and nucleus in NUCLEI and coda in FINALS


def is_structure_prefix(word):
    # Phục vụ L20: tiếng đang dở còn có thể hoàn tất thành tiếng hợp lệ
    if not is_valid_syllable(word):
        return False
    onset, nucleus, coda = split_syllable(word)
    if not nucleus:
        head = onset + coda
        return any(c.startswith(head) for c in CONSONANTS)
    if not any(n.startswith(nucleus) for n in NUCLEI):
        return False
    return any(f.startswith(coda) for f in FINALS)


def get_rhyme_part(word):
    # Phục vụ L10, L12 và L15: trích phần vần của một tiếng
    w = remove_tones(word.strip())
    for c in CONSONANTS:
        if w.startswith(c):
            return w[len(c):]
    return w


def build_prefix_set(syllables):
    """Tập tiền tố sinh ra từ danh sách tiếng hợp lệ, phục vụ L20.

    Mỗi tiếng đóng góp cả dạng có dấu lẫn dạng đã bỏ dấu thanh, nhờ vậy hai
    lối đặt dấu khác nhau như hòa và hoà đều được chấp nhận khi tiếng còn
    đang hình thành.
    """
    prefixes = set()
    for s in syllables:
        for form in (s, remove_tones(s)):
            for i in range(1, len(form) + 1):
                prefixes.add(form[:i])
    return prefixes


def build_variant_index(syllables):
    """Chỉ mục tiếng theo cặp dạng bỏ dấu và dấu thanh, phục vụ L16.

    Chỉ mục này giúp nhận ra một tiếng viết đúng nhưng đặt dấu khác với kho
    ngữ liệu, chẳng hạn hoà so với hòa, thay vì loại nhầm tiếng đó.
    """
    return {(remove_tones(s), get_tone_char(s)) for s in syllables}


def _bytes_to_unicode():
    # Hàm nền: giải mã từ vựng của bộ tách từ byte level BPE
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
    # Hàm nền: đưa token thô về chuỗi ký tự thật
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
    """Quét một lần toàn bộ từ vựng để dựng ba bảng tra cứu.

    Bảng thứ nhất phục vụ L18, gồm những token mà bề mặt chỉ chứa chữ cái
    tiếng Việt và dấu cách. Mọi token còn lại bị loại ngay từ vòng lọc ứng
    viên, nhờ vậy chữ Hán, chữ số, dấu câu và ký tự hỏng không thể lọt vào
    bài thơ. Bảng thứ hai phục vụ L10 và L12, gồm tiền tố của những token
    nối tiếp, dùng để biết một tiếng đang dở còn khả năng đi tới vần đích.
    Bảng thứ ba lưu bề mặt của từng token hợp lệ, phục vụ Đ06.

    Kết quả được gắn vào chính bộ tách từ nên chỉ tốn một lần quét cho mỗi
    lần nạp mô hình.
    """
    cached = getattr(tokenizer, "_lucbat_vocab_tables", None)
    if cached is not None:
        return cached

    allowed = set()
    prefixes = set()
    surface_map = {}
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
            # L18: bề mặt token chỉ được gồm chữ cái tiếng Việt và dấu cách
            if not s or not s.strip():
                continue
            if not is_viet_surface(s):
                continue
            # mỗi token chỉ được mang tối đa một tiếng, nhờ ràng buộc này mọi
            # tiếng mới sinh ra đều đi qua đủ các bước kiểm tra
            pieces = s.split()
            if len(pieces) != 1 or len(pieces[0]) > MAX_SYLLABLE_LEN:
                continue
            allowed.add(tid)
            surface_map[tid] = s
            if s[0] in WORD_START_MARKS:
                continue
            t = remove_tones(s)
            for k in (1, 2, 3):
                if len(t) >= k:
                    prefixes.add(t[:k])

    tables = (allowed, prefixes, surface_map)
    try:
        setattr(tokenizer, "_lucbat_vocab_tables", tables)
    except Exception:
        pass
    return tables


def load_syllables(path, min_count=1):
    """Dựng danh sách tiếng hợp lệ từ kho ngữ liệu huấn luyện, phục vụ L16.

    Kho ngữ liệu vẫn còn dấu câu và một số chuỗi rác, vì vậy mỗi tiếng được
    chuẩn hóa rồi lọc lại theo bảng chữ cái và cấu trúc âm tiết tiếng Việt
    trước khi đưa vào danh sách.
    """
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
                 strict_lexicon=True):
        # Chuẩn bị tài nguyên dùng chung cho toàn bộ tập luật
        self.tokenizer = tokenizer
        self.max_total_lines = max_total_lines
        self.candidate_pool = candidate_pool
        self.end_bonus = end_bonus
        self.rhyme_bonus = rhyme_bonus
        self.whole_bonus = whole_bonus
        self.syllables = syllables or None
        self.strict_lexicon = strict_lexicon
        if self.syllables:
            self.syllable_prefixes = build_prefix_set(self.syllables)
            self.syllable_variants = build_variant_index(self.syllables)
        else:
            self.syllable_prefixes = None
            self.syllable_variants = None

        self.eos_token_id = tokenizer.eos_token_id
        self.newline_ids = self._collect_newline_ids(tokenizer)
        if not self.newline_ids:
            raise ValueError("Không xác định được token xuống dòng của bộ tách từ")
        self.nl_token_id = self.newline_ids[0]

        self.allowed_ids, self.cont_prefixes, self.surfaces = \
            _vocab_tables(tokenizer)
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.special_ids.discard(None)
        self.allowed_ids = self.allowed_ids - self.special_ids
        # token nối tiếp, tức token không mở đầu một tiếng mới
        self.cont_ids = [tid for tid in self.allowed_ids
                         if self.surfaces.get(tid, " ")[0] not in WORD_START_MARKS]
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

    # ------------------------------------------------------------------
    # Luật chính tả tiếng Việt, không bao giờ được nới lỏng
    # ------------------------------------------------------------------
    def _in_lexicon(self, word):
        # L16: tiếng có mặt trong danh sách, chấp nhận cả biến thể đặt dấu
        if word in self.syllables:
            return True
        key = (remove_tones(word), get_tone_char(word))
        return key in self.syllable_variants

    def _lex_tier(self, word):
        """Luật chính tả tiếng Việt, trả về None nghĩa là loại thẳng.

        Đây là chốt chặn cuối cùng khiến những chuỗi không phải tiếng Việt
        không bao giờ lọt vào bài thơ. Khi có danh sách tiếng dựng từ kho ngữ
        liệu thì danh sách này là căn cứ duy nhất, vì nó chính xác hơn mọi
        quy tắc cấu trúc. Khi không đọc được danh sách thì hệ thống lùi về
        kiểm tra cấu trúc âm đầu, phần vần và âm cuối.
        """
        w = normalize(word)
        # L17: bảng chữ cái tiếng Việt và độ dài tối đa của một tiếng
        if not is_valid_syllable(w):
            return None
        if self.syllables is not None:
            # L16: căn cứ theo danh sách tiếng
            if self._in_lexicon(w):
                return TIER_FULL
            if self.strict_lexicon:
                return None
            return TIER_LEX if has_viet_structure(w) else None
        # L19: không có danh sách thì căn cứ theo cấu trúc âm tiết
        return TIER_FULL if has_viet_structure(w) else None

    def _prefix_ok(self, word):
        # L20: tiếng đang dở còn khả năng trở thành một tiếng hợp lệ
        w = normalize(word)
        if not is_valid_syllable(w):
            return False
        if self.syllable_prefixes:
            if w in self.syllable_prefixes or remove_tones(w) in self.syllable_prefixes:
                return True
            if self.strict_lexicon:
                return False
        return is_structure_prefix(w)

    def _extendable(self, word):
        # L21: tiếng đã hoàn chỉnh nhưng vẫn còn là tiền tố của tiếng dài hơn
        w = normalize(word)
        if not self.syllable_prefixes:
            return False
        return w in self.syllable_prefixes and not self._in_lexicon(w)

    def _can_close(self, word):
        # L21: chỉ chốt dòng khi tiếng cuối là một tiếng tiếng Việt trọn vẹn
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
        # Kiểm tra tiếng vừa được chốt: L06 đến L10, L12, L13, L15, L16, L19
        # L16, L17 và L19: chính tả tiếng Việt, sai là loại thẳng
        tier = self._lex_tier(word)
        if tier is None:
            return None

        # L06 đến L09: thanh điệu tại vị trí chẵn
        if pos in TARGET_TONE and get_tone(word) != TARGET_TONE[pos]:
            tier = max(tier, TIER_TONE)

        # L10 và L12: gieo vần lưng và vần chân
        target = self._target_rhyme(pos, ctx)
        if target and get_rhyme_part(word) != target:
            tier = max(tier, TIER_RHYME)

        # L15: không trùng nguyên tiếng vần đích
        same = self._target_word(pos, ctx)
        if same and word.lower() == same.lower():
            tier = max(tier, TIER_SOFT)

        # L13: tiếng thứ 6 và thứ 8 của câu bát khác dấu thanh
        if pos == 8 and ctx["word6"]:
            if get_tone_mark(word) == get_tone_mark(ctx["word6"]):
                tier = max(tier, TIER_SOFT)
        return tier

    def _partial_tier(self, pos, word, ctx):
        # Kiểm tra tiếng đang hình thành: L06 đến L13, L15, L20
        # L20: chuỗi đang dở phải còn khả năng thành một tiếng tiếng Việt
        if not self._prefix_ok(word):
            return None

        tone = get_tone(word)
        # L11: tiếng chưa có nguyên âm thì bỏ qua kiểm tra thanh điệu
        if tone is None:
            return TIER_FULL

        tier = TIER_FULL
        # L06 đến L09: thanh điệu tại vị trí chẵn
        if pos in TARGET_TONE and tone != TARGET_TONE[pos]:
            tier = max(tier, TIER_TONE)

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

    def _rescue_complete(self, row, mask, words, scores):
        """Đ06: quét toàn bộ từ vựng để viết nốt tiếng đang dở.

        Vòng lọc chính chỉ xét vài nghìn token có điểm cao nhất, nên đôi khi
        token duy nhất viết nốt được tiếng đang dở lại nằm ngoài nhóm đó và
        hệ thống rơi vào bế tắc. Trước khi chấp nhận chốt dòng sớm, hàm này
        tìm trong toàn bộ từ vựng những token nối tiếp biến tiếng đang dở
        thành một tiếng tiếng Việt trọn vẹn.
        """
        if not words:
            return None
        last = words[-1]
        if self._lex_tier(last) is not None:
            return None
        # lượt một căn cứ theo danh sách tiếng, lượt hai chỉ cần đúng cấu
        # trúc âm tiết, cốt sao tiếng cuối vẫn là một tiếng tiếng Việt
        best = None
        for strict in (True, False):
            best, best_score = None, None
            for tid in self.cont_ids:
                s = self.surfaces.get(tid)
                if not s or " " in s:
                    continue
                cand = last + s
                if len(cand) > MAX_SYLLABLE_LEN:
                    continue
                if strict:
                    if self._lex_tier(cand) is None:
                        continue
                elif not has_viet_structure(cand):
                    continue
                score = row[tid].item()
                if best_score is None or score > best_score:
                    best, best_score = tid, score
            if best is not None:
                break
        if best is None:
            return None
        mask[best] = row[best] + self.end_bonus
        scores[0] = mask
        return scores

    def _close_line(self, row, mask, terminators, scores):
        # Đ01 đến Đ03: chốt dòng bằng token xuống dòng hoặc token kết thúc
        if not terminators:
            terminators = self.newline_ids
        best = max(terminators, key=lambda t: row[t].item())
        mask[best] = row[best] + self.end_bonus
        scores[0] = mask
        return scores

    def __call__(self, input_ids, scores):
        # Điều phối toàn bộ tập luật: L01 đến L21 và Đ01 đến Đ07
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

        # L01 đến L04, L21, Đ01, Đ02: dòng đủ tiếng và tiếng cuối đã trọn vẹn
        # thì chỉ cho kết thúc. Nếu tiếng cuối còn dở thì không chốt dòng mà
        # đi tiếp xuống vòng lọc để viết nốt tiếng đó.
        if n == target_len and words and has_vowel(words[-1]) \
                and self._can_close(words[-1]):
            return self._close_line(row, mask, terminators, scores)

        vocab = row.shape[-1]
        buckets = {t: [] for t in TIER_ORDER}
        seen = set()

        # Đ04: mở rộng dần nhóm ứng viên
        for k in self.candidate_pool:
            _, indices = torch.topk(row, min(k, vocab))
            for token_id in indices.tolist():
                if token_id in seen:
                    continue
                seen.add(token_id)

                # L01, L04 và L18: chỉ nhận token thuần chữ cái tiếng Việt,
                # token đặc biệt và token kết thúc không đi qua vòng này
                if token_id not in self.allowed_ids:
                    continue
                s = self._tok_str(token_id)
                if not is_viet_surface(s) or not s.strip():
                    continue

                # L05: không cho vượt quá số tiếng mục tiêu
                cand_words = (current_line + s).split()
                m = len(cand_words)
                if m == 0 or m > target_len:
                    continue

                tier = TIER_FULL
                if m > n and n >= 1:
                    # tiếng liền trước vừa được chốt lại
                    t = self._finalize_tier(n, words[-1], ctx)
                    if t is None:
                        continue
                    tier = max(tier, t)
                    # L14: không lặp tiếng liền kề
                    if m >= 2 and cand_words[-1].lower() == cand_words[-2].lower():
                        tier = max(tier, TIER_SOFT)

                # L17, L19 và L20: tiếng đang hình thành phải hợp lệ
                t = self._partial_tier(m, cand_words[-1], ctx)
                if t is None:
                    continue
                tier = max(tier, t)

                bonus = self.rhyme_bonus if self._is_exact_rhyme(
                    m, cand_words[-1], ctx) else 0.0
                # Đ07: ưu tiên nhẹ những token khép lại một tiếng trọn vẹn,
                # nhờ vậy dòng thơ ít khi dừng giữa chừng một tiếng dở dang
                if self._lex_tier(cand_words[-1]) is not None:
                    bonus += self.whole_bonus
                buckets[tier].append((token_id, bonus))

            if buckets[TIER_FULL]:
                break

        # Đ05: chọn theo tầng ưu tiên, đầy đủ trước rồi mới nới lỏng dần.
        # Luật chính tả tiếng Việt không nằm trong các tầng nới lỏng nên một
        # chuỗi không phải tiếng Việt không bao giờ được chọn.
        chosen = []
        for t in TIER_ORDER:
            if buckets[t]:
                chosen = buckets[t]
                break

        if not chosen:
            # Đ06 rồi mới tới Đ03: cố viết nốt tiếng đang dở, nếu vẫn không
            # được thì chốt dòng sớm, tuyệt đối không sinh ký tự lạ
            out = self._rescue_complete(row, mask, words, scores)
            if out is not None:
                return out
            return self._close_line(row, mask, terminators, scores)

        for token_id, bonus in chosen:
            mask[token_id] = row[token_id] + bonus

        # lưới an toàn cuối cùng, không bao giờ trả về hàng điểm toàn âm vô cực
        if not bool(torch.isfinite(mask).any()):
            return self._close_line(row, mask, terminators, scores)

        scores[0] = mask
        return scores