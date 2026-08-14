
import gc
import json
import os
import shutil
import tempfile
import time
import unicodedata

import streamlit as st
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList, StoppingCriteria,
                          StoppingCriteriaList)

from lucbat_processor import StrictLucBatProcessor, load_syllables

MODELS_CONFIG = {
    "Qwen 3.5 0.8B": "./qwen-lucbat-model",
    "Llama 3.2 3B": "./llama-lucbat-model",
    "Gemma 2 2B": "./gemma-lucbat-model",
}

SYLLABLE_FILE = "train.txt"

HUYEN, SAC, NGA, HOI, NANG = "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"
TONE_MARKS = {HUYEN, SAC, NGA, HOI, NANG}
TRAC_MARKS = {SAC, NGA, HOI, NANG}
VOWELS = set("aeiouyăâêôơư")
CONSONANTS = ["ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
              "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r",
              "s", "t", "v", "x"]


# ==========================================================================
# Xử lý ngữ âm và chấm điểm
# ==========================================================================
def _decompose(word):
    return unicodedata.normalize("NFD", word.lower())


def get_tone(word):
    d = _decompose(word)
    if not any(c in VOWELS for c in d if c not in TONE_MARKS):
        return None
    return "T" if any(c in TRAC_MARKS for c in d) else "B"


def get_tone_mark(word):
    return "huyền" if HUYEN in _decompose(word) else "ngang"


def remove_tones(word):
    stripped = "".join(c for c in _decompose(word) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def get_rhyme_part(word):
    w = remove_tones(word.strip())
    for c in CONSONANTS:
        if w.startswith(c):
            return w[len(c):]
    return w


def check_prompt(text):
    """Kiểm tra lời nhắc có thể mở đầu một bài lục bát hay không."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "chưa nhập nội dung"

    last = lines[-1].split()
    if len(last) > 6:
        return ("dòng cuối đang có %d tiếng, vượt quá 6 tiếng của câu lục"
                % len(last))
    for pos, target in ((2, "B"), (4, "T"), (6, "B")):
        if len(last) >= pos and get_tone(last[pos - 1]) != target:
            return ("tiếng thứ %d của dòng cuối là %s mang thanh %s nhưng câu "
                    "lục cần thanh %s"
                    % (pos, last[pos - 1],
                       "trắc" if get_tone(last[pos - 1]) == "T" else "bằng",
                       "bằng" if target == "B" else "trắc"))
    return None


def check_poem(poem):
    lines = [ln for ln in poem.split("\n") if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        words = line.split()
        expected = 6 if i % 2 == 0 else 8
        positions = [2, 4, 6] if expected == 6 else [2, 4, 6, 8]
        targets = ["B", "T", "B"] if expected == 6 else ["B", "T", "B", "B"]

        tone_bad = []
        for pos, tgt in zip(positions, targets):
            # dòng thiếu tiếng thì vị trí tương ứng cũng tính là vi phạm
            if len(words) < pos or get_tone(words[pos - 1]) != tgt:
                tone_bad.append(pos)
        tone_ok = len(positions) - len(tone_bad)

        rhyme = None
        if i > 0:
            prev = lines[i - 1].split()
            prev_expected = 8 if expected == 6 else 6
            # chỉ xác định được vị trí gieo vần khi cả hai dòng đủ tiếng
            if len(prev) != prev_expected or len(words) != expected:
                rhyme = False
            elif expected == 8:
                rhyme = get_rhyme_part(prev[5]) == get_rhyme_part(words[5])
            else:
                rhyme = get_rhyme_part(prev[7]) == get_rhyme_part(words[5])

        mark = None
        if expected == 8 and len(words) >= 8:
            mark = get_tone_mark(words[5]) != get_tone_mark(words[7])

        rows.append(dict(index=i + 1, line=line, n_words=len(words),
                         expected=expected, ok_len=len(words) == expected,
                         tone_ok=tone_ok, tone_total=len(positions),
                         tone_bad=tone_bad, rhyme=rhyme, mark=mark))
    return rows


MAX_LEN, MAX_TONE, MAX_RHYME = 10.0, 30.0, 60.0


def score_poem(poem, target_lines):
    """Chấm điểm niêm luật theo thang 100.

    Độ dài tối đa 10 điểm, thanh điệu tối đa 30 điểm, gieo vần tối đa 60 điểm.
    Mỗi thành phần được quy đổi theo tỷ lệ đạt được rồi nhân với điểm tối đa.
    """
    rows = check_poem(poem)
    if not rows:
        return dict(lines_ok=False, n_lines=0, r_len=0.0, r_tone=0.0,
                    r_rhyme=0.0, p_len=0.0, p_tone=0.0, p_rhyme=0.0,
                    score=0.0, n_len_ok=0, n_len_total=0, n_tone_ok=0,
                    n_tone_total=0, n_rhyme_ok=0, n_rhyme_total=0)

    n_len_ok = sum(r["ok_len"] for r in rows)
    n_len_total = len(rows)
    n_tone_ok = sum(r["tone_ok"] for r in rows)
    n_tone_total = sum(r["tone_total"] for r in rows)
    rhymes = [r["rhyme"] for r in rows if r["rhyme"] is not None]
    n_rhyme_ok = sum(rhymes)
    n_rhyme_total = len(rhymes)

    r_len = n_len_ok / n_len_total
    r_tone = n_tone_ok / n_tone_total if n_tone_total else 0.0
    r_rhyme = n_rhyme_ok / n_rhyme_total if n_rhyme_total else 0.0

    p_len = MAX_LEN * r_len
    p_tone = MAX_TONE * r_tone
    p_rhyme = MAX_RHYME * r_rhyme

    return dict(lines_ok=len(rows) == target_lines, n_lines=len(rows),
                r_len=r_len, r_tone=r_tone, r_rhyme=r_rhyme,
                p_len=p_len, p_tone=p_tone, p_rhyme=p_rhyme,
                score=p_len + p_tone + p_rhyme,
                n_len_ok=n_len_ok, n_len_total=n_len_total,
                n_tone_ok=n_tone_ok, n_tone_total=n_tone_total,
                n_rhyme_ok=n_rhyme_ok, n_rhyme_total=n_rhyme_total)


# ==========================================================================
# Nạp mô hình
# ==========================================================================
def _downgrade_tokenizer_json(data):
    changed = False
    model = data.get("model") or {}
    merges = model.get("merges")
    if isinstance(merges, list) and merges and isinstance(merges[0], (list, tuple)):
        model["merges"] = [" ".join(pair) for pair in merges]
        changed = True
    if "ignore_merges" in model:
        model.pop("ignore_merges")
        changed = True
    return changed


def _copy_tokenizer_files(path, target):
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full) and not name.endswith(".safetensors"):
            shutil.copy(full, os.path.join(target, name))


def _repaired_tokenizer_dir(path):
    src = os.path.join(path, "tokenizer.json")
    if not os.path.exists(src):
        return None
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)
    if not _downgrade_tokenizer_json(data):
        return None
    tmp = tempfile.mkdtemp(prefix="lucbat_fix_")
    _copy_tokenizer_files(path, tmp)
    with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return tmp


def _patched_config_dir(path):
    src = os.path.join(path, "tokenizer_config.json")
    if not os.path.exists(src):
        return None
    with open(src, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "tokenizer_class" not in cfg:
        return None
    cfg.pop("tokenizer_class", None)
    tmp = tempfile.mkdtemp(prefix="lucbat_tok_")
    _copy_tokenizer_files(path, tmp)
    with open(os.path.join(tmp, "tokenizer_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False)
    return tmp


def load_tokenizer(path):
    """Nạp bộ tách từ từ chính thư mục mô hình, không tải gì từ mạng."""
    attempts = [
        ("thư mục mô hình",
         lambda: AutoTokenizer.from_pretrained(path, trust_remote_code=True)),
        ("thư mục mô hình, bộ tách từ chậm",
         lambda: AutoTokenizer.from_pretrained(path, use_fast=False,
                                               trust_remote_code=True)),
    ]

    repaired = patched = None
    try:
        repaired = _repaired_tokenizer_dir(path)
    except Exception:
        repaired = None
    if repaired:
        attempts.append(("sau khi hạ định dạng trường merges",
                         lambda: AutoTokenizer.from_pretrained(
                             repaired, trust_remote_code=True)))
    try:
        patched = _patched_config_dir(path)
    except Exception:
        patched = None
    if patched:
        attempts.append(("sau khi gỡ trường tokenizer_class",
                         lambda: AutoTokenizer.from_pretrained(
                             patched, trust_remote_code=True)))
    if repaired:
        try:
            both = _patched_config_dir(repaired)
        except Exception:
            both = None
        if both:
            attempts.append(("sau khi sửa cả hai lỗi định dạng",
                             lambda: AutoTokenizer.from_pretrained(
                                 both, trust_remote_code=True)))

    errors = []
    for label, fn in attempts:
        try:
            return fn(), label
        except Exception as exc:
            errors.append("%s: %s" % (label, str(exc).split("\n")[0][:150]))
    raise RuntimeError(
        "Không nạp được bộ tách từ từ %s\nTệp hiện có: %s\n%s"
        % (path, ", ".join(sorted(os.listdir(path))), "\n".join(errors)))


def pick_dtype():
    if torch.cuda.is_available():
        return torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_model_weights(path):
    common = dict(dtype=pick_dtype(), trust_remote_code=True,
                  low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, attn_implementation="sdpa", **common)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(path, **common)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        model = model.to("mps")
    elif torch.cuda.is_available():
        model = model.to("cuda")
        
    return model


@st.cache_resource(show_spinner=False)
def load_model(name):
    path = MODELS_CONFIG[name]
    if not os.path.isdir(path):
        raise FileNotFoundError("Không tìm thấy thư mục mô hình %s" % path)
    tokenizer, source = load_tokenizer(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_weights(path)
    model.eval()
    return model, tokenizer, source


@st.cache_resource(show_spinner=False)
def load_syllable_set():
    try:
        return load_syllables(SYLLABLE_FILE)
    except OSError:
        return None


# ==========================================================================
# Sinh thơ
# ==========================================================================
def _eos_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return []
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def _count_lines(tokenizer, input_ids):
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return [ln for ln in text.split("\n") if ln.strip()], text


class MinLineLogitsProcessor(LogitsProcessor):
    """Chặn token kết thúc cho tới khi bài thơ đủ số dòng yêu cầu.

    Đây là cơ chế kiểm soát độ dài, không phải luật niêm luật. Nếu thiếu cơ
    chế này, mô hình ở chế độ không có tập luật thường tự dừng sau vài dòng,
    khiến hai chế độ cho ra số dòng khác nhau và không so sánh được với nhau.
    """

    def __init__(self, tokenizer, max_total_lines):
        self.tokenizer = tokenizer
        self.max_total_lines = max_total_lines
        self.eos_ids = _eos_ids(tokenizer)

    def __call__(self, input_ids, scores):
        if not self.eos_ids:
            return scores
        lines, _ = _count_lines(self.tokenizer, input_ids)
        if len(lines) < self.max_total_lines:
            for eid in self.eos_ids:
                scores[0, eid] = -float("inf")
        return scores


class LineCountStoppingCriteria(StoppingCriteria):
    """Dừng khi văn bản đã đạt đủ số dòng quy định."""

    def __init__(self, tokenizer, max_total_lines):
        self.tokenizer = tokenizer
        self.max_total_lines = max_total_lines

    def __call__(self, input_ids, scores, **kwargs):
        lines, text = _count_lines(self.tokenizer, input_ids)
        return len(lines) > self.max_total_lines or (
            len(lines) == self.max_total_lines and text.rstrip("\n") != text)


def normalize_poem(text, target_lines):
    lines = [" ".join(ln.split()) for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[:target_lines])


def estimate_max_tokens(tokenizer, prompt, target_lines):
    """Ước lượng số token cần sinh dựa trên số token trung bình của một tiếng."""
    try:
        n_tok = len(tokenizer(prompt)["input_ids"])
    except Exception:
        n_tok = len(prompt.split())
    n_syl = max(1, len(prompt.split()))
    per_syllable = max(1.0, n_tok / n_syl)
    per_line = 9 * per_syllable + 2
    return int(per_line * target_lines) + 12


def _run_once(model, tokenizer, text, target_lines, apply_rules, syllables,
              gen_kwargs, max_new_tokens):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    processors = LogitsProcessorList()
    stoppers = StoppingCriteriaList()
    if apply_rules:
        processors.append(
            StrictLucBatProcessor(tokenizer, max_total_lines=target_lines,
                                  syllables=syllables))
    else:
        # cùng một quy định về số dòng cho cả hai chế độ
        processors.append(MinLineLogitsProcessor(tokenizer, target_lines))
    stoppers.append(LineCountStoppingCriteria(tokenizer, target_lines))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=processors,
            stopping_criteria=stoppers,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def count_lines(text):
    return len([ln for ln in text.split("\n") if ln.strip()])


def generate(model, tokenizer, prompt, target_lines, apply_rules,
             syllables, gen_kwargs, max_new_tokens, max_rounds=5):
    """Sinh thơ cho tới khi đủ số dòng yêu cầu.

    Mô hình có thể dừng sớm dù token kết thúc đã bị chặn, chẳng hạn khi chạm
    giới hạn số token. Khi đó hàm này nối thêm ký tự xuống dòng rồi sinh tiếp,
    nhờ vậy hai chế độ luôn cho ra cùng một số dòng.

    Quá trình sinh không cố định hạt giống ngẫu nhiên, do đó mỗi lần bấm sinh
    thơ sẽ cho ra một phương án khác nhau.
    """
    text = prompt
    per_line = max(12, max_new_tokens // max(1, target_lines))
    for _ in range(max_rounds):
        done = count_lines(text)
        budget = max_new_tokens if done == 0 else max(
            per_line * (target_lines - done) + 8, 24)
        text = _run_once(model, tokenizer, text, target_lines, apply_rules,
                         syllables, gen_kwargs, budget)
        if count_lines(text) >= target_lines:
            break
        text = "\n".join(
            [ln for ln in text.split("\n") if ln.strip()]) + "\n"
    return normalize_poem(text, target_lines)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================================================
# Giao diện
# ==========================================================================
def render_evaluation(poem, target_lines):
    rows = check_poem(poem)
    if not rows:
        st.warning("Không có nội dung để đánh giá.")
        return

    result = score_poem(poem, target_lines)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng điểm", "%.1f / 100" % result["score"])
    c2.metric("Số tiếng", "%.1f / %.0f" % (result["p_len"], MAX_LEN),
              "%d/%d dòng đúng" % (result["n_len_ok"], result["n_len_total"]),
              delta_color="off")
    c3.metric("Thanh điệu", "%.1f / %.0f" % (result["p_tone"], MAX_TONE),
              "%d/%d vị trí đúng" % (result["n_tone_ok"],
                                     result["n_tone_total"]),
              delta_color="off")
    c4.metric("Gieo vần", "%.1f / %.0f" % (result["p_rhyme"], MAX_RHYME),
              "%d/%d cặp đúng" % (result["n_rhyme_ok"],
                                  result["n_rhyme_total"]),
              delta_color="off")

    if result["lines_ok"]:
        st.success("Bài thơ có đúng %d dòng theo yêu cầu." % target_lines)
    else:
        st.warning("Bài thơ có %d dòng, khác với %d dòng được yêu cầu. Thử "
                   "bấm sinh lại hoặc tăng temperature để mô hình viết dài hơn."
                   % (result["n_lines"], target_lines))

    st.markdown("**Chi tiết từng dòng**")
    table = []
    for r in rows:
        loai = "Câu lục" if r["expected"] == 6 else "Câu bát"
        do_dai = "%d/%d" % (r["n_words"], r["expected"])
        if not r["ok_len"]:
            do_dai += " (sai)"
        thanh = "%d/%d" % (r["tone_ok"], r["tone_total"])
        if r["tone_bad"]:
            thanh += " (sai ở tiếng %s)" % ", ".join(
                str(p) for p in r["tone_bad"])
        van = "không xét"
        if r["rhyme"] is not None:
            van = "đúng" if r["rhyme"] else "sai"
        dau = "không xét"
        if r["mark"] is not None:
            dau = "khác dấu" if r["mark"] else "trùng dấu"
        table.append({"Dòng": r["index"], "Loại": loai, "Câu thơ": r["line"],
                      "Số tiếng": do_dai, "Thanh điệu": thanh,
                      "Gieo vần": van, "Tiếng 6 và 8": dau})
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Cách tính điểm"):
        st.markdown(
            "Tổng điểm theo thang 100 gồm ba thành phần với điểm tối đa quy "
            "định sẵn.")
        st.dataframe([
            {"Thành phần": "Số tiếng", "Điểm tối đa": "10",
             "Cách tính": "Tỷ lệ dòng có đúng số tiếng, câu lục 6 tiếng và "
                          "câu bát 8 tiếng",
             "Đạt được": "%.1f điểm" % result["p_len"]},
            {"Thành phần": "Thanh điệu", "Điểm tối đa": "30",
             "Cách tính": "Tỷ lệ vị trí chẵn đúng luật bằng trắc, gồm tiếng "
                          "2, 4, 6 của câu lục và tiếng 2, 4, 6, 8 của câu bát",
             "Đạt được": "%.1f điểm" % result["p_tone"]},
            {"Thành phần": "Gieo vần", "Điểm tối đa": "60",
             "Cách tính": "Tỷ lệ cặp dòng hiệp vần đúng, gồm vần lưng và "
                          "vần chân",
             "Đạt được": "%.1f điểm" % result["p_rhyme"]},
        ], use_container_width=True, hide_index=True)
        st.markdown(
            "Cột Tiếng 6 và 8 trong bảng chi tiết kiểm tra riêng quy tắc hai "
            "tiếng này của câu bát phải khác dấu thanh. Quy tắc này không tính "
            "vào tổng điểm.")


def main():
    st.set_page_config(page_title="Sinh thơ lục bát", page_icon="📜",
                       layout="wide")
    st.title("Hệ thống sinh thơ lục bát tiếng Việt")
    st.caption("Ứng dụng mô hình ngôn ngữ lớn kết hợp kỹ thuật giải mã có "
               "ràng buộc")

    syllables = load_syllable_set()

    with st.sidebar:
        st.header("Cấu hình")
        model_name = st.selectbox("Mô hình", list(MODELS_CONFIG.keys()))
        target_lines = st.slider("Số câu", min_value=2, max_value=12, value=4,
                                 step=2)
        mode = st.radio("Chế độ sinh",
                        ["Có tập luật", "Không có tập luật", "So sánh cả hai"],
                        help="Số dòng được kiểm soát như nhau ở cả hai chế độ. "
                             "Khác biệt giữa hai chế độ chỉ nằm ở tập luật "
                             "niêm luật gồm số tiếng, thanh điệu và gieo vần.")
        st.divider()
        st.subheader("Tham số sinh")
        temperature = st.slider(
            "temperature", 0.1, 1.5, 0.80, 0.05,
            help="Độ ngẫu nhiên khi chọn từ. Giá trị thấp cho câu thơ an "
                 "toàn, dùng từ quen thuộc và ít sai luật nhưng dễ nhàm. Giá "
                 "trị cao cho câu thơ giàu bất ngờ nhưng dễ lạc ý và tăng "
                 "nguy cơ mô hình bị bộ ràng buộc chặn.")
        top_k = st.slider(
            "top_k", 1, 100, 100, 1,
            help="Số từ có xác suất cao nhất được đưa vào vòng chọn. Giá trị "
                 "nhỏ khiến thơ đơn điệu và lặp ý. Giá trị lớn mở rộng vốn từ "
                 "nhưng có thể chọn phải từ ít phù hợp ngữ cảnh.")
        top_p = st.slider(
            "top_p", 0.1, 1.0, 0.9, 0.05,
            help="Ngưỡng xác suất tích lũy. Chỉ giữ nhóm từ nhỏ nhất có tổng "
                 "xác suất vượt ngưỡng này. Giá trị thấp thu hẹp lựa chọn "
                 "quanh những từ chắc chắn, giá trị gần 1 gần như không lọc.")
        rep_penalty = st.slider(
            "repetition_penalty", 1.0, 1.5, 1.0, 0.05,
            help="Mức phạt những từ đã xuất hiện. Bằng 1 là không phạt nên "
                 "thơ dễ lặp từ. Giá trị lớn hạn chế lặp nhưng nếu quá cao sẽ "
                 "cản cả những từ cần nhắc lại như điệp ngữ.")
        st.divider()
        if syllables:
            st.caption("Danh sách tiếng hợp lệ: %d tiếng" % len(syllables))
        else:
            st.caption("Không đọc được %s nên bỏ qua luật danh sách tiếng"
                       % SYLLABLE_FILE)

    st.subheader("Lời nhắc đầu vào")
    prompt = st.text_area(
        "Nhập một từ khóa, một câu lục hoặc một đoạn thơ mồi",
        value="Thanh xuân như một tách trà",
        height=120,
        help="Có thể nhập vài tiếng làm gợi ý, một câu lục đủ 6 tiếng, hoặc "
             "nhiều dòng thơ. Mô hình sẽ viết tiếp cho đủ số câu yêu cầu.")

    prompt = prompt.strip()
    if prompt:
        warning = check_prompt(prompt)
        if warning:
            st.warning("Lời nhắc chưa đúng luật câu lục: %s. Mô hình vẫn sinh "
                       "được nhưng phần lời nhắc sẽ bị tính là sai luật."
                       % warning)

    run = st.button("Sinh thơ", type="primary", use_container_width=True)

    if not run:
        st.info("Chọn mô hình và chế độ ở thanh bên, nhập lời nhắc rồi bấm "
                "Sinh thơ.")
        return
    if not prompt:
        st.error("Vui lòng nhập lời nhắc đầu vào.")
        return

    try:
        with st.spinner("Đang nạp mô hình %s" % model_name):
            model, tokenizer, source = load_model(model_name)
        st.caption("Bộ tách từ nạp từ: %s" % source)
    except Exception as exc:
        st.error("Không nạp được mô hình %s" % model_name)
        st.code(str(exc))
        return

    gen_kwargs = dict(do_sample=True, temperature=float(temperature),
                      top_k=int(top_k), top_p=float(top_p),
                      repetition_penalty=float(rep_penalty), use_cache=True)
    max_tokens = estimate_max_tokens(tokenizer, prompt, target_lines)

    if mode == "So sánh cả hai":
        modes = [("Không có tập luật", False), ("Có tập luật", True)]
    elif mode == "Có tập luật":
        modes = [("Có tập luật", True)]
    else:
        modes = [("Không có tập luật", False)]

    outputs = []
    for label, apply_rules in modes:
        with st.spinner("Đang sáng tác ở chế độ %s" % label.lower()):
            start = time.time()
            try:
                poem = generate(model, tokenizer, prompt, target_lines,
                                apply_rules, syllables, gen_kwargs,
                                max_tokens)
            except Exception as exc:
                st.error("Lỗi khi sinh thơ ở chế độ %s" % label.lower())
                st.code(str(exc))
                continue
            elapsed = time.time() - start
        outputs.append((label, poem, elapsed))

    free_memory()
    if not outputs:
        return

    st.divider()
    if len(outputs) == 1:
        label, poem, elapsed = outputs[0]
        st.subheader("Bài thơ (%s)" % label.lower())
        st.text(poem)
        st.caption("Thời gian sinh: %.1f giây" % elapsed)
        st.download_button("Tải bài thơ", poem, file_name="bai_tho.txt")
        st.divider()
        st.subheader("Đánh giá bài thơ")
        render_evaluation(poem, target_lines)
        return

    cols = st.columns(2)
    for col, (label, poem, elapsed) in zip(cols, outputs):
        with col:
            st.subheader(label)
            st.text(poem)
            st.caption("Thời gian sinh: %.1f giây" % elapsed)

    st.divider()
    st.subheader("Đánh giá bài thơ")
    tabs = st.tabs([label for label, _, _ in outputs])
    for tab, (label, poem, _) in zip(tabs, outputs):
        with tab:
            render_evaluation(poem, target_lines)

    st.markdown("**Đối chiếu hai chế độ**")
    compare = []
    for label, poem, elapsed in outputs:
        r = score_poem(poem, target_lines)
        compare.append({
            "Chế độ": label,
            "Số tiếng": "%.1f / %.0f" % (r["p_len"], MAX_LEN),
            "Thanh điệu": "%.1f / %.0f" % (r["p_tone"], MAX_TONE),
            "Gieo vần": "%.1f / %.0f" % (r["p_rhyme"], MAX_RHYME),
            "Tổng điểm": "%.1f / 100" % r["score"],
            "Số dòng": "%d / %d" % (r["n_lines"], target_lines),
            "Thời gian": "%.1f giây" % elapsed,
        })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    if len(compare) == 2:
        a = score_poem(outputs[0][1], target_lines)
        b = score_poem(outputs[1][1], target_lines)
        st.caption(
            "Chênh lệch khi bật tập luật: số tiếng %+.1f điểm, thanh điệu "
            "%+.1f điểm, gieo vần %+.1f điểm, tổng %+.1f điểm."
            % (b["p_len"] - a["p_len"], b["p_tone"] - a["p_tone"],
               b["p_rhyme"] - a["p_rhyme"], b["score"] - a["score"]))


if __name__ == "__main__":
    main()