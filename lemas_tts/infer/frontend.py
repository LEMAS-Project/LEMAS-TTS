import os, re, regex
import langid
import uroman as ur
import jieba, zhconv
from num2words import num2words

jieba.set_dictionary(dictionary_path=os.path.join(os.path.dirname(__file__) + "/../infer/text_norm/jieba_dict.txt"))
# from pypinyin.core import Pinyin
from pypinyin import pinyin, lazy_pinyin, Style

from .text_norm.txt2pinyin import _PAUSE_SYMBOL, get_phoneme_from_char_and_pinyin
from .text_norm.cn_tn import NSWNormalizer
from .text_norm.tokenizer import TextTokenizer, txt2phone
from pypinyin.contrib.tone_convert import to_initials, to_finals_tone3
from pypinyin_dict.phrase_pinyin_data import large_pinyin  # large_pinyin  #  cc_cedict
large_pinyin.load()

class TextNorm():
    def __init__(self, dtype="phone"):
        # my_pinyin = Pinyin(MyConverter())
        # self.pinyin_parser = my_pinyin.pinyin
        cmn_lexicon = open(os.path.join(os.path.dirname(__file__)+'/../infer/text_norm/pinyin-lexicon-r.txt'),'r', encoding="utf-8").readlines()
        cmn_lexicon = [x.strip().split() for x in cmn_lexicon]
        self.cmn_dict = {x[0]:x[1:] for x in cmn_lexicon}
        langid.set_languages(['es','pt','zh','en','de','fr','it','ru', 'vi','id','th','ja','ko','ar'])
        langs = {"en":"en-us", "it":"it", "es":"es", "pt":"pt-br", "fr":"fr-fr", "de":"de", "ru":"ru", "vi":"vi", "id":"id", "th":"th", "ja":"ja", "ko":"ko"} # "zh":"cmn", "cmn":"cmn", "ar":"ar-sa"}
        text_tokenizer = {}
        for k,v in langs.items():
            tokenizer = TextTokenizer(language=v, backend="espeak")
            lang = "zh" if k == "cmn" else k
            text_tokenizer[k] = (lang, tokenizer)
        self.text_tokenizer = text_tokenizer
        self.cn_tn = NSWNormalizer()
        self.dtype = dtype

    def detect_lang(self, text):
        lang, _ = langid.classify(text)[0]
        return lang

    def sil_type(self, time_s):
        if round(time_s) < 0.4:
            return ""
        elif round(time_s) >= 0.4 and round(time_s) < 0.8:
            return "#1"
        elif round(time_s) >= 0.8 and round(time_s) < 1.5:
            return "#2"
        elif round(time_s) >= 1.5 and round(time_s) < 3.0:
            return "#3"
        elif round(time_s) >= 3.0:
            return "#4"


    def add_sil_raw(self, sub_list, start_time, end_time, target_transcript):
        txt = []
        txt_list = [x["word"] for x in sub_list]
        sil = self.sil_type(sub_list[0]["start"])
        if len(sil) > 0:
            txt.append(sil)
        txt.append(txt_list[0])
        for i in range(1, len(sub_list)):
            if sub_list[i]["start"] >= start_time and sub_list[i]["end"] <= end_time:
                txt.append(target_transcript)
                target_transcript = ""
            else:
                sil = self.sil_type(sub_list[i]["start"] - sub_list[i-1]["end"])
                if len(sil) > 0:
                    txt.append(sil)
                txt.append(txt_list[i])
        return ' '.join(txt)

    def add_sil(self, sub_list, start_time, end_time, target_transcript, src_lang, tar_lang):
        txts = []
        txt_list = [x["word"] for x in sub_list]
        sil = self.sil_type(sub_list[0]["start"])
        if len(sil) > 0:
            txts.append([src_lang, sil])

        if sub_list[0]["start"] < start_time:
            txts.append([src_lang, txt_list[0]])
        for i in range(1, len(sub_list)):
            if sub_list[i]["start"] >= start_time and sub_list[i]["end"] <= end_time:
                txts.append([tar_lang, target_transcript])
                target_transcript = ""
            else:
                sil = self.sil_type(sub_list[i]["start"] - sub_list[i-1]["end"])
                if len(sil) > 0:
                    txts.append([src_lang, sil])
                txts.append([src_lang, txt_list[i]])
                
        target_txt = [txts[0]]
        for txt in txts[1:]:
            if txt[1] == "":
                continue
            if txt[0] != target_txt[-1][0]:
                target_txt.append([txt[0], ""])
            target_txt[-1][-1] += " " + txt[1]
        
        return target_txt

    def replace_numbers_with_words(self, sentence, lang="en"):
        sentence = re.sub(r'(\d+)', r' \1 ', sentence) # add spaces around numbers
        
        def replace_with_words(match):
            num = match.group(0)
            try:
                return num2words(num, lang=lang) # Convert numbers to words
            except:
                return num # In case num2words fails (unlikely with digits but just to be safe)
        return re.sub(r'\b\d+\b', replace_with_words, sentence) # Regular expression that matches numbers


    def get_prompt(self, sub_list, start_time, end_time, src_lang):
        txts = []
        txt_list = [x["word"] for x in sub_list]

        if start_time <= sub_list[0]["start"]:
            sil = self.sil_type(sub_list[0]["start"])
            if len(sil) > 0:
                txts.append([src_lang, sil])
            txts.append([src_lang, txt_list[0]])
        
        for i in range(1, len(sub_list)):
            # if sub_list[i]["start"] <= start_time and sub_list[i]["end"] <= end_time:
            #     txts.append([tar_lang, target_transcript])
            #     target_transcript = ""
            if sub_list[i]["start"] >= start_time and sub_list[i]["end"] <= end_time:
                sil = self.sil_type(sub_list[i]["start"] - sub_list[i-1]["end"])
                if len(sil) > 0:
                    txts.append([src_lang, sil])
                txts.append([src_lang, txt_list[i]])

        target_txt = [txts[0]]
        for txt in txts[1:]:
            if txt[1] == "":
                continue
            if txt[0] != target_txt[-1][0]:
                target_txt.append([txt[0], ""])
            target_txt[-1][-1] += " " + txt[1]
        return target_txt


    def txt2pinyin(self, text):
        txts, phonemes = [], []
        texts = re.split(r"(#\d)", text)
        print("before norm: ", texts)
        for text in texts:
            if text in {'#1', '#2', '#3', '#4'}:
                txts.append(text)
                phonemes.append(text)
                continue
            text = self.cn_tn.normalize(text.strip())
            
            text_list = list(jieba.cut(text))
            print("jieba cut: ", text, text_list)
            for words in text_list:
                if words in _PAUSE_SYMBOL:
                    # phonemes[-1] += _PAUSE_SYMBOL[words]
                    phonemes.append(_PAUSE_SYMBOL[words])
                    # phonemes.append('#1')
                    txts[-1] += words
                elif re.search("[\u4e00-\u9fa5]+", words):
                    # pinyin = self.pinyin_parser(words, style=Style.TONE3, errors="ignore")
                    pinyin = lazy_pinyin(words, style=Style.TONE3, tone_sandhi=True, neutral_tone_with_five=True)
                    new_pinyin = []
                    for x in pinyin:
                        x = "".join(x)
                        if "#" not in x:
                            new_pinyin.append(x)
                        else:
                            phonemes.append(words)
                            continue
                    # new_pinyin = change_tone_in_bu_or_yi(words, new_pinyin) if len(words)>1 and words[-1] not in {"一","不"} else new_pinyin
                    phoneme = get_phoneme_from_char_and_pinyin(words, new_pinyin)
                    phonemes += phoneme
                    txts += list(words)
                elif re.search(r"[a-zA-Z]", words) or re.search(r"#[1-4]", words):
                    phonemes.append(words.upper())
                    txts.append(words.upper())
                    # phonemes.append("#1")
        # phones = " ".join(phonemes)
        return txts, phonemes


    def txt2pin_phns(self, text):
        text = re.sub(r'(?<! )(' + r'[^\w\s]' + r')', r' \1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # print(text.split(" "))
        res_list = []
        for txt in text.split(" "):
            if txt in self.cmn_dict:
                # res_list +=  ["(zh)" + x for x in self.cmn_dict[txt]]
                res_list.append("(zh)")
                res_list.append(to_initials(txt, strict=False))
                res_list.append(to_finals_tone3(txt, neutral_tone_with_five=True))
            elif txt == '':
                continue
            elif txt[0] in {"#1", "#2", "#3", "#4"} or not bool(regex.search(r'\p{L}', txt[0][0])): 
                if len(res_list) > 0 and res_list[-1] == "_":
                    res_list.pop()
                res_list += [txt]
                continue
            else:
                if len(res_list) > 0 and res_list[-1] == "_":
                    res_list.pop()
                lang = langid.classify(txt)[0]
                lang = lang if lang in self.text_tokenizer else "en"
                tokenizer = self.text_tokenizer[lang][1]
                ipa = tokenizer.backend.phonemize([txt], separator=tokenizer.separator, strip=True, njobs=1)
                phns = ipa[0] if ipa[0][0] == "(" else f"({lang})_" + ipa[0]
                res_list += phns.replace("_", "|_|").split("|")

                # lang = phns.split(")")[0][1:]
                # phns = phns[len(lang)+3:].replace("_", "|_|")
                # phns = phns.split("|")
                # for i in range(len(phns)):
                #     if phns[i] not in {"#1", "#2", "#3", "#4", "_", ",", ".", "?", "!"}: 
                #         phns[i] = f"({lang})" + phns[i]
                # res_list += phns
            res_list.append("_")
        res = "|".join(res_list)
        res = re.sub(r'(\|_)+', '|_', res)
        return res


    def text2phn(self, sentence, lang=None):
        if not lang:
            lang = langid.classify(sentence)[0]
        if re.search("[\u4e00-\u9fa5]+", sentence):
            txts, phones = self.txt2pinyin(sentence)
            transcript_norm = " ".join(phones)
            phones = self.txt2pin_phns(transcript_norm) # IPA mix Pinyin
        else:
            transcript = self.replace_numbers_with_words(sentence, lang=lang).split(' ')
            transcript_norm = sentence
            # All IPA
            phones = txt2phone(self.text_tokenizer[lang][1], transcript_norm.strip().replace(".", ",").replace("。", ","))
            phones = f"({lang})|" + phones if phones[0] != "(" else phones
        return phones


    def text2norm(self, sentence, lang=None):
        if not lang:
            lang = langid.classify(sentence)[0]
        if re.search("[\u4e00-\u9fa5]+", sentence):
            txts, phones = self.txt2pinyin(sentence)
            transcript_norm = " ".join(phones)
        else:
            transcript = self.replace_numbers_with_words(sentence, lang=lang).split(' ')
            transcript_norm = sentence
        return (lang, transcript_norm)
