# Indonesian TTS Text Normalization for YouTube subtitles
# Requirements: pip install num2words
import re
from num2words import num2words

# --- small slang map (expandable) ---
SLANG_MAP = {
    "gpp": "nggak apa-apa",
    "gak": "nggak", "ga": "nggak", "gk": "nggak",
    "sy": "saya", "sya": "saya",
    "km": "kamu",
    "tp": "tapi", "tpi": "tapi",
    "jd": "jadi",
    "bgt": "banget",
    "blm": "belum",
    "trs": "terus",
    "sm": "sama",
    "wkwk": "wkwk",  # keep as-is (laugh token) or strip later
    "wkwkwk": "wkwk"
}

# emoji pattern: removes most emoji blocks
EMOJI_PATTERN = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

# units map
UNITS = {
    "kg": "kilogram","g": "gram","km": "kilometer",
    "m": "meter","cm": "sentimeter","mm": "milimeter",
    "l": "liter"
}

# helper: safe num2words for Indonesian
def num_to_words_ind(num_str):
    """Convert numeric string to Indonesian words.
       - Handles integers and simple decimals like '1.5' (reads digits after decimal).
       - Removes grouping dots in Indonesian numbers (e.g. '10.000').
    """
    num_str = num_str.strip()
    # remove thousand separators commonly used in Indonesian (dot)
    # but if decimal point (like '1,5' or '1.5'), assume '.' is decimal point (we expect '.' used)
    # We'll treat commas as thousand separators too if no decimal comma present.
    if re.match(r'^\d+[.,]\d+$', num_str):
        # decimal number: normalize to use '.' then split
        s = num_str.replace(',', '.')
        left, right = s.split('.', 1)
        try:
            left_w = num2words(int(left), lang='id')
        except:
            left_w = left
        # read each decimal digit separately
        right_w = " ".join(num2words(int(d), lang='id') for d in right if d.isdigit())
        return f"{left_w} koma {right_w}"
    else:
        # remove non-digit separators like dots or commas used as thousand separators
        cleaned = re.sub(r'[.,]', '', num_str)
        try:
            return num2words(int(cleaned), lang='id')
        except:
            return num_str

# helper: per-digit reader for phone numbers (default)
def read_digits_per_digit(number_str, prefix_plus=False):
    digits = re.findall(r'\d', number_str)
    words = " ".join(num2words(int(d), lang='id') for d in digits)
    if prefix_plus:
        return "plus " + words
    return words

# noise removal rule for tokens like 'yyy6yy' or other long mixed garbage:
def is_noise_token(tok):
    # remove tokens that:
    # - length >=4 and contain at least one digit and at least one letter (typical ASR/keyboard noise)
    # - or tokens of a single repeated char length >=4 (e.g., 'aaaa', '!!!!!!' but punctuation handled earlier)
    if len(tok) < 4:
        return False
    if re.search(r'[A-Za-z]', tok) and re.search(r'\d', tok):
        return True
    if re.fullmatch(r'(.)\1{3,}', tok):  # same char repeated >=4
        return True
    return False

# --- 新增：标点规范化函数 ---
def punctuation_normalize(text):
    """
    - 替换除 . , ! ? 之外的所有标点为逗号
    - 统一多重逗号为单逗号
    - 去掉开头多余逗号、省略号
    - 逗号后空格规范化
    """
    # 替换括号、引号、冒号、分号、破折号、省略号等为逗号
    text = re.sub(r'[:;()\[\]{}"“”«»…—–/\\]', ',', text)
    # 多个逗号替换成一个
    text = re.sub(r',+', ',', text)
    # 开头去掉逗号和省略号
    text = re.sub(r'^(,|\.\.\.|…)+\s*', '', text)
    # 逗号后空格规范
    text = re.sub(r'\s*,\s*', ', ', text)
    # 多余空白合并
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_id_tts(text):
    """
    Main normalization pipeline tailored for:
    - Indonesian YouTube subtitles (mostly ASR/MT)
    - TTS frontend requirements:
      * Remove emojis
      * Keep . , ! ? as sentence/phrase delimiters
      * Replace other punctuation with comma
      * Expand numbers, percents, currency, units, times, dates
      * Remove keyboard noise like 'yyy6yy'
      * Keep English words as-is
      * Keep repeated words (do not collapse)
    """
    if not text:
        return text

    # 1) Normalize whitespace and trim
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 2) Remove emojis
    text = EMOJI_PATTERN.sub('', text)

    # 3) 标点规范化（替代原有 PUNCT_TO_COMMA 替换）
    text = punctuation_normalize(text)

    # 保护时间和日期的代码（防止被逗号破坏）
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"__TIME_{m.group(1)}_{m.group(2)}__", text)
    text = re.sub(r'(\d{1,4})[\/-](\d{1,2})[\/-](\d{1,4})', lambda m: f"__DATE_{m.group(1)}_{m.group(2)}_{m.group(3)}__", text)

    # 恢复时间日期标记
    text = re.sub(r'__TIME_(\d{1,2})_(\d{2})__', lambda m: f"{m.group(1)}:{m.group(2)}", text)
    text = re.sub(r'__DATE_(\d{1,4})_(\d{1,2})_(\d{1,4})__', lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}", text)

    # 4) Tokenize loosely by spaces and punctuation
    tokens = re.split(r'(\s+|[,.!?])', text)  # keep delimiters

    out_tokens = []
    for tok in tokens:
        if not tok or tok.isspace():
            out_tokens.append(tok)
            continue

        # keep punctuation .,!? as-is
        if tok in ['.', ',', '!', '?']:
            out_tokens.append(tok)
            continue

        # remove any remaining emojis or control chars
        if EMOJI_PATTERN.search(tok):
            continue

        # slang normalization
        lower_tok = tok.lower()
        if lower_tok in SLANG_MAP:
            out_tokens.append(SLANG_MAP[lower_tok])
            continue

        # remove noise tokens
        if is_noise_token(tok):
            continue

        # currency: Rp 10.000 or rp10.000
        m = re.match(r'^(Rp|rp)\s*([0-9\.,]+)$', tok)
        if m:
            num = m.group(2)
            cleaned = re.sub(r'[.,]', '', num)
            out_tokens.append(f"{num_to_words_ind(cleaned)} rupiah")
            continue

        # percent like 30%
        m = re.match(r'^(\d+)%$', tok)
        if m:
            out_tokens.append(f"{num_to_words_ind(m.group(1))} persen")
            continue

        # phone numbers +62..., 0812...
        m = re.match(r'^\+?\d[\d\-\s]{6,}\d$', tok)
        if m:
            prefix_plus = tok.startswith('+')
            out_tokens.append(read_digits_per_digit(tok, prefix_plus=prefix_plus))
            continue

        # time hh:mm
        m = re.match(r'^(\d{1,2}):(\d{2})$', tok)
        if m:
            h, mi = m.group(1), m.group(2)
            h_w = num_to_words_ind(h.lstrip('0') or '0')
            mi_w = num_to_words_ind(mi.lstrip('0') or '0')
            out_tokens.append(f"pukul {h_w} lewat {mi_w} menit")
            continue

        # date yyyy/mm/dd or dd/mm/yyyy
        m = re.match(r'^(\d{1,4})\/(\d{1,2})\/(\d{1,4})$', tok)
        if m:
            a,b,c = m.group(1), m.group(2).zfill(2), m.group(3)
            if len(a) == 4:
                year, month, day = a, b, c
            elif len(c) == 4:
                day, month, year = a, b, c
            else:
                day, month, year = a, b, c
            MONTHS = {
                "01": "Januari","02": "Februari","03": "Maret","04": "April",
                "05": "Mei","06": "Juni","07": "Juli","08": "Agustus",
                "09": "September","10": "Oktober","11": "November","12": "Desember"
            }
            day_w = num_to_words_ind(day.lstrip('0') or '0')
            year_w = num_to_words_ind(year)
            month_name = MONTHS.get(month, month)
            out_tokens.append(f"{day_w} {month_name} {year_w}")
            continue

        # units like 30kg
        m = re.match(r'^(\d+)\s*(kg|g|km|m|cm|mm|l)$', tok, flags=re.I)
        if m:
            num, unit = m.group(1), m.group(2).lower()
            unit_word = UNITS.get(unit, unit)
            out_tokens.append(f"{num_to_words_ind(num)} {unit_word}")
            continue

        # plain integers
        if re.fullmatch(r'\d+', tok):
            out_tokens.append(num_to_words_ind(tok))
            continue

        # numbers with separators
        if re.fullmatch(r'[\d\.,]+', tok) and re.search(r'[.,]', tok):
            out_tokens.append(num_to_words_ind(tok))
            continue

        # keep English/as-is tokens
        out_tokens.append(tok)

    normalized = "".join(out_tokens)

    # final cleanup: spacing around punctuation
    normalized = re.sub(r'\s+,', ',', normalized)
    normalized = re.sub(r',\s*', ', ', normalized)
    normalized = re.sub(r'\s+\.', '.', normalized)
    normalized = re.sub(r'\s+!', '!', normalized)
    normalized = re.sub(r'\s+\?', '?', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # 如果你不想全部小写，注释掉下面这行
    normalized = normalized.lower()

    return normalized

# -------------------------
# Example usage and tests
# -------------------------
if __name__ == "__main__":
    examples = [
        "kita cek Project nadi PHP pemberi harapan palsu tuh yyy6yy 46 ini ini usernya ini di bagian user",
        "Harga Rp 10.000, diskon 30%! Buka jam 09:30 (hari 2025/11/28).",
        "Call +62 812-3456-7890 sekarang!",
        "angka kecil 3.14 dan 1,234 serta 1000",
        "[musik]",
        "... atau mungkin juga jumlah anggota keluarga mereka."
    ]
    for ex in examples:
        print("IN: ", ex)
        print("OUT:", normalize_id_tts(ex))
        print("-"*60)
