# -*- coding: utf-8 -*-
import gc
import time
import unicodedata

import torch
from bert_score import BERTScorer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, set_seed)

from lucbat_processor import StrictLucBatProcessor, load_syllables

MODEL_NAME = "Qwen 3.5 0.8B"
MODEL_PATH = "./qwen-lucbat-model"
DATASET_FILE = "test.txt"
SYLLABLE_FILE = "train.txt"

SAMPLE_RATIO = 0.1
SEED = 42
MAX_NEW_TOKENS = 25
BATCH_SIZE = 64
MODES = ("no_rules", "with_rules")

GEN_KWARGS = dict(do_sample=True, temperature=0.8, top_p=0.9)

TRAC_MARKS = {"\u0301", "\u0303", "\u0309", "\u0323"}
HUYEN = "\u0300"
TONE_MARKS = TRAC_MARKS | {HUYEN}
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


def get_tone_mark(word):
    return "huyen" if HUYEN in _decompose(word) else "ngang"


def remove_tones(word):
    stripped = "".join(c for c in _decompose(word) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def get_rhyme_part(word):
    w = remove_tones(word.strip())
    for c in CONSONANTS:
        if w.startswith(c):
            return w[len(c):]
    return w


def is_rhyme(w1, w2):
    if not w1 or not w2:
        return False
    return get_rhyme_part(w1) == get_rhyme_part(w2)


def evaluate_luc_bat(cau_luc, cau_bat):
    luc = cau_luc.strip().split()
    bat = cau_bat.strip().split()

    # L05
    score_len = 10.0 if len(bat) == 8 else 0.0

    # L06 đến L09
    score_tone = 0.0
    if len(bat) >= 8:
        step = 30.0 / 4.0
        for pos, target in ((2, "B"), (4, "T"), (6, "B"), (8, "B")):
            if get_tone(bat[pos - 1]) == target:
                score_tone += step

    # L10
    score_rhyme = 0.0
    if len(luc) >= 6 and len(bat) >= 6 and is_rhyme(luc[5], bat[5]):
        score_rhyme = 60.0

    # L13
    mark_ok = 0.0
    if len(bat) >= 8 and get_tone_mark(bat[5]) != get_tone_mark(bat[7]):
        mark_ok = 1.0

    # L15
    dup_rhyme = 0.0
    if len(luc) >= 6 and len(bat) >= 6 and luc[5].lower() == bat[5].lower():
        dup_rhyme = 1.0

    return dict(score=score_len + score_tone + score_rhyme,
                length=score_len, tone=score_tone, rhyme=score_rhyme,
                mark=mark_ok, dup=dup_rhyme)


def load_test_pairs(path, ratio):
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    pairs = []
    i = 0
    while i < len(lines) - 1:
        if len(lines[i].split()) == 6 and len(lines[i + 1].split()) == 8:
            pairs.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    if ratio < 1.0:
        pairs = pairs[:max(1, int(len(pairs) * ratio))]
    return pairs


def pick_dtype():
    if torch.cuda.is_available():
        return torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def generate_cau_bat(model, tokenizer, cau_luc, processor, seed):
    set_seed(seed)
    prompt = cau_luc + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    processors = LogitsProcessorList()
    if processor is not None:
        processors.append(processor)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            logits_processor=processors,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            **GEN_KWARGS,
        )

    new_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    for line in text.split("\n"):
        if line.strip():
            return " ".join(line.split())
    return ""


def phobert_f1(scorer, generated, reference):
    scores = []
    with torch.no_grad():
        for idx in range(0, len(generated), BATCH_SIZE):
            g = generated[idx: idx + BATCH_SIZE]
            r = reference[idx: idx + BATCH_SIZE]
            P, R, F1 = scorer.score(g, r)
            scores.extend(F1.cpu().tolist())
            del P, R, F1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    return (sum(scores) / len(scores) * 100) if scores else 0.0


def main():
    syllables = None
    if SYLLABLE_FILE:
        try:
            syllables = load_syllables(SYLLABLE_FILE)
            print("Danh sách tiếng hợp lệ: %d tiếng" % len(syllables))
        except OSError:
            print("Không đọc được %s nên bỏ qua luật L16" % SYLLABLE_FILE)

    test_data = load_test_pairs(DATASET_FILE, SAMPLE_RATIO)
    print("Số cặp câu lục và câu bát: %d" % len(test_data))

    print("Đang tải mô hình %s" % MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=pick_dtype(),
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    print("Đang khởi tạo BERTScorer")
    scorer = BERTScorer(model_type="vinai/phobert-base", num_layers=9,
                        rescale_with_baseline=False)

    results = {}
    for mode in MODES:
        apply_rules = mode == "with_rules"
        label = "CÓ TẬP LUẬT" if apply_rules else "KHÔNG CÓ TẬP LUẬT"
        print("\n" + "=" * 62)
        print("ĐÁNH GIÁ %s TRÊN %d MẪU" % (label, len(test_data)))
        print("=" * 62)

        processor = None
        if apply_rules:
            processor = StrictLucBatProcessor(tokenizer, max_total_lines=2,
                                              syllables=syllables)

        totals = dict(score=0.0, length=0.0, tone=0.0, rhyme=0.0,
                      mark=0.0, dup=0.0)
        generated, reference = [], []
        start = time.time()

        for i, (cau_luc, cau_bat_ref) in enumerate(test_data):
            cau_bat = generate_cau_bat(model, tokenizer, cau_luc, processor,
                                       SEED + i)
            row = evaluate_luc_bat(cau_luc, cau_bat)
            for k in totals:
                totals[k] += row[k]
            generated.append(cau_bat)
            reference.append(cau_bat_ref)

            if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
                print("Đã xử lý [%d/%d] mẫu" % (i + 1, len(test_data)))

        gen_time = time.time() - start
        print("Đang tính PhoBERTScore")
        f1 = phobert_f1(scorer, generated, reference)

        n = len(test_data)
        results[mode] = dict(
            score=totals["score"] / n,
            length=totals["length"] / n / 10 * 100,
            tone=totals["tone"] / n / 30 * 100,
            rhyme=totals["rhyme"] / n / 60 * 100,
            mark=totals["mark"] / n * 100,
            dup=totals["dup"] / n * 100,
            phobert=f1,
            seconds=gen_time,
            samples=generated[:5],
        )

    print("\n" + "=" * 62)
    print("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH %s" % MODEL_NAME.upper())
    print("=" * 62)
    print("Tổng số mẫu: %d" % len(test_data))
    header = ("%-18s %9s %10s %10s %9s %8s %9s"
              % ("Chế độ", "Độ dài", "Thanh điệu", "Gieo vần", "Điểm luật",
                 "PhoBERT", "Thời gian"))
    print(header)
    print("-" * len(header))
    for mode in MODES:
        r = results[mode]
        label = "Có tập luật" if mode == "with_rules" else "Không tập luật"
        print("%-18s %8.1f%% %9.1f%% %9.1f%% %9.1f %7.1f%% %8.1f phút"
              % (label, r["length"], r["tone"], r["rhyme"], r["score"],
                 r["phobert"], r["seconds"] / 60))

    if len(MODES) == 2:
        a, b = results[MODES[0]], results[MODES[1]]
        print("-" * len(header))
        print("%-18s %8.1f%% %9.1f%% %9.1f%% %9.1f %7.1f%%"
              % ("Chênh lệch",
                 b["length"] - a["length"], b["tone"] - a["tone"],
                 b["rhyme"] - a["rhyme"], b["score"] - a["score"],
                 b["phobert"] - a["phobert"]))

    print("\nCác chỉ số bổ sung")
    for mode in MODES:
        r = results[mode]
        label = "Có tập luật" if mode == "with_rules" else "Không tập luật"
        print("%-18s khác dấu thanh %5.1f%% | trùng tiếng vần %5.1f%%"
              % (label, r["mark"], r["dup"]))

    print("\nMột số câu bát sinh ra")
    for i in range(min(5, len(test_data))):
        print("  Câu lục: %s" % test_data[i][0])
        for mode in MODES:
            label = "không luật" if mode == "no_rules" else "có luật"
            print("    %-12s %s" % (label, results[mode]["samples"][i]))


if __name__ == "__main__":
    main()