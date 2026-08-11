import gc
import unicodedata

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, StoppingCriteria,
                          StoppingCriteriaList, set_seed)

from lucbat_processor import StrictLucBatProcessor, load_syllables

MODELS_CONFIG = {
    "Qwen": "./qwen-lucbat-model",
    "LLaMA": "./llama-lucbat-model",
    "Gemma": "./gemma-lucbat-model",
}

KEYWORDS = [
    "Chiều buông",
    "Mưa nguồn",
    "Đường quê",
    "Trăng lên",
    "Người đi",
]

TARGET_LINES = 4

SYLLABLE_FILE = "train.txt"

SEED = 42

GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.85,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
)

MAX_NEW_TOKENS = 40 * TARGET_LINES

assert TARGET_LINES % 2 == 0, "TARGET_LINES phải là số chẵn"

SYLLABLES = None


HUYEN, SAC, NGA, HOI, NANG = "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"
TONE_MARKS = {HUYEN, SAC, NGA, HOI, NANG}
TRAC_MARKS = {SAC, NGA, HOI, NANG}
VOWELS = set("aeiouyăâêôơư")
CONSONANTS = ["ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
              "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r",
              "s", "t", "v", "x"]


def _decompose(word):
    return unicodedata.normalize("NFD", word.lower())


def get_tone(word):
    d = _decompose(word)
    if not any(c in VOWELS for c in d if c not in TONE_MARKS):
        return None
    return "T" if any(c in TRAC_MARKS for c in d) else "B"


def remove_tones(word):
    stripped = "".join(c for c in _decompose(word) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def get_rhyme_part(word):
    w = remove_tones(word.strip())
    for c in CONSONANTS:
        if w.startswith(c):
            return w[len(c):]
    return w


def check_keyword(keyword):
    words = keyword.split()
    if not 1 <= len(words) <= 5:
        return "từ khóa phải có từ một đến năm tiếng"
    for pos, target in ((2, "B"), (4, "T")):
        if len(words) >= pos and get_tone(words[pos - 1]) != target:
            return ("tiếng thứ %d là \"%s\" mang thanh %s, cần thanh %s"
                    % (pos, words[pos - 1], get_tone(words[pos - 1]), target))
    return None


def check_poem(poem):
    lines = [ln for ln in poem.split("\n") if ln.strip()]
    report = []
    for i, line in enumerate(lines):
        words = line.split()
        expected = 6 if i % 2 == 0 else 8
        positions = [2, 4, 6] if expected == 6 else [2, 4, 6, 8]
        targets = ["B", "T", "B"] if expected == 6 else ["B", "T", "B", "B"]

        tone_ok = 0
        for pos, tgt in zip(positions, targets):
            if len(words) >= pos and get_tone(words[pos - 1]) == tgt:
                tone_ok += 1

        rhyme = None
        if i > 0:
            prev = lines[i - 1].split()
            if expected == 8 and len(prev) >= 6 and len(words) >= 6:
                rhyme = get_rhyme_part(prev[5]) == get_rhyme_part(words[5])
            elif expected == 6 and len(prev) >= 8 and len(words) >= 6:
                rhyme = get_rhyme_part(prev[7]) == get_rhyme_part(words[5])

        report.append(dict(line=line, n_words=len(words), expected=expected,
                           ok_len=len(words) == expected, tone_ok=tone_ok,
                           tone_total=len(positions), rhyme=rhyme))
    return report


def score_poem(poem, target_lines):
    rows = check_poem(poem)
    if not rows:
        return dict(r_lines=0.0, r_len=0.0, r_tone=0.0, r_rhyme=0.0, score=0.0)

    r_lines = 1.0 if len(rows) == target_lines else 0.0
    r_len = sum(r["ok_len"] for r in rows) / len(rows)
    tone_ok = sum(r["tone_ok"] for r in rows)
    tone_total = sum(r["tone_total"] for r in rows) or 1
    r_tone = tone_ok / tone_total
    rhymes = [r["rhyme"] for r in rows if r["rhyme"] is not None]
    r_rhyme = (sum(rhymes) / len(rhymes)) if rhymes else 0.0

    score = 10 * r_len + 30 * r_tone + 60 * r_rhyme
    return dict(r_lines=r_lines, r_len=r_len, r_tone=r_tone, r_rhyme=r_rhyme,
                score=score)


def print_report(poem):
    if not poem.strip():
        print("      (không có kết quả)")
        return
    for r in check_poem(poem):
        info = ("số tiếng %d/%d %s | thanh điệu %d/%d"
                % (r["n_words"], r["expected"],
                   "đúng" if r["ok_len"] else "SAI",
                   r["tone_ok"], r["tone_total"]))
        if r["rhyme"] is not None:
            info += " | gieo vần %s" % ("đúng" if r["rhyme"] else "SAI")
        print("      %-42s %s" % (r["line"], info))


class LineCountStoppingCriteria(StoppingCriteria):

    def __init__(self, tokenizer, max_total_lines):
        self.tokenizer = tokenizer
        self.max_total_lines = max_total_lines

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        lines = [ln for ln in text.split("\n") if ln.strip()]
        return len(lines) > self.max_total_lines or (
            len(lines) == self.max_total_lines and text.rstrip("\n") != text)


def generate_poem(model, tokenizer, keyword, target_lines, apply_rules, seed):
    set_seed(seed)
    inputs = tokenizer(keyword, return_tensors="pt").to(model.device)

    processors = LogitsProcessorList()
    stoppers = StoppingCriteriaList()

    if apply_rules:
        processors.append(
            StrictLucBatProcessor(tokenizer, max_total_lines=target_lines,
                                  syllables=SYLLABLES))
    else:
        stoppers.append(LineCountStoppingCriteria(tokenizer, target_lines))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            logits_processor=processors,
            stopping_criteria=stoppers,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            **GEN_KWARGS,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalize_poem(text, target_lines)


def normalize_poem(text, target_lines):
    lines = [" ".join(ln.split()) for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[:target_lines])


def pick_dtype():
    if torch.cuda.is_available():
        return torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def average(rows, key):
    return (sum(r[key] for r in rows) / len(rows)) if rows else 0.0


def main():
    global SYLLABLES
    if SYLLABLE_FILE:
        try:
            SYLLABLES = load_syllables(SYLLABLE_FILE)
            print("Danh sách tiếng hợp lệ: %d tiếng" % len(SYLLABLES))
        except OSError:
            print("Không đọc được %s nên bỏ qua luật L17" % SYLLABLE_FILE)

    keywords = []
    for kw in KEYWORDS:
        err = check_keyword(kw)
        if err:
            print("[!] Bỏ qua từ khóa \"%s\": %s" % (kw, err))
        else:
            keywords.append(kw)
    if not keywords:
        print("Không còn từ khóa hợp lệ")
        return

    print("Số từ khóa      : %d" % len(keywords))
    print("Số dòng quy định: %d" % TARGET_LINES)
    print("Mô hình phải tự viết tiếp câu lục từ từ khóa\n")

    results = {}
    for name, path in MODELS_CONFIG.items():
        print("-" * 66)
        print("Đang tải %s từ %s" % (name, path))
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=pick_dtype(),
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            model.eval()

            samples = []
            for idx, kw in enumerate(keywords):
                print("[%s] Từ khóa \"%s\"" % (name, kw))
                seed = SEED + idx
                no_rules = generate_poem(model, tokenizer, kw, TARGET_LINES,
                                         False, seed)
                with_rules = generate_poem(model, tokenizer, kw, TARGET_LINES,
                                           True, seed)
                samples.append(dict(keyword=kw, no_rules=no_rules,
                                    with_rules=with_rules))
            results[name] = samples

            del model, tokenizer
            free_memory()
            print("Đã giải phóng bộ nhớ của %s" % name)

        except Exception as exc:
            print("[!] Lỗi khi chạy mô hình %s: %s" % (name, exc))
            results[name] = []

    print("\n" + "=" * 66)
    print("KẾT QUẢ CHI TIẾT (%d DÒNG)" % TARGET_LINES)
    print("=" * 66)
    for name, samples in results.items():
        print("\n>>> MÔ HÌNH: %s <<<" % name.upper())
        for s in samples:
            print("\n  Từ khóa: %s" % s["keyword"])
            print("    [1] KHÔNG CÓ TẬP LUẬT")
            print_report(s["no_rules"])
            print("    [2] CÓ TẬP LUẬT")
            print_report(s["with_rules"])
        print("-" * 66)

    print("\n" + "=" * 66)
    print("BẢNG TỔNG HỢP TRÊN %d TỪ KHÓA" % len(keywords))
    print("=" * 66)
    header = ("%-8s %-14s %8s %8s %9s %8s %8s"
              % ("Mô hình", "Chế độ", "Đủ dòng", "Độ dài", "Thanh điệu",
                 "Gieo vần", "Điểm"))
    print(header)
    print("-" * len(header))
    for name, samples in results.items():
        for key, label in (("no_rules", "Không luật"), ("with_rules", "Có luật")):
            rows = [score_poem(s[key], TARGET_LINES) for s in samples]
            if not rows:
                continue
            print("%-8s %-14s %7.1f%% %7.1f%% %8.1f%% %7.1f%% %8.1f"
                  % (name, label,
                     average(rows, "r_lines") * 100,
                     average(rows, "r_len") * 100,
                     average(rows, "r_tone") * 100,
                     average(rows, "r_rhyme") * 100,
                     average(rows, "score")))


if __name__ == "__main__":
    main()