import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import argparse
import os, sys, re
from random import shuffle
from tqdm import tqdm
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
import jieba
jieba.set_dictionary(dictionary_path=os.path.join(os.path.dirname(__file__)+'/jieba_dict.txt'))

from .symbols import pinyin_dict
from .cn_tn import NSWNormalizer


zh_pattern = re.compile("[\u4e00-\u9fa5]")
alpha_pattern = re.compile(r"[a-zA-Z]")

def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None

def is_alpha(word):
    global alpha_pattern
    match = alpha_pattern.search(word)
    return match is not None

def get_phoneme_from_char_and_pinyin(chn_char, pinyin):
    # we do not need #4, use sil to replace it
    chn_char = chn_char.replace("#4", "")
    char_len = len(chn_char)
    i, j = 0, 0
    result = []
    # print(pinyin)
    while i < char_len:
        cur_char = chn_char[i]
        if is_zh(cur_char):
            if pinyin[j][:-1] == 'n':  # 处理特殊“嗯” 特殊拼音
                pinyin[j] = 'en' + pinyin[j][-1]
            if i < len(chn_char)-2 and is_zh(chn_char[i:i+3]) and pinyin[j][-1] == pinyin[j+1][-1] == pinyin[j+2][-1] == '3':  # 处理连续三个三声变调
                pinyin[j+1] = pinyin[j+1][:-1] + '2'
                # print(chn_char[i:i+3], pinyin[j:j+3])
            if i < len(chn_char)-1 and pinyin[j][:-1] in pinyin_dict and is_zh(chn_char[i]) and is_zh(chn_char[i+1]) and pinyin[j][-1] == pinyin[j+1][-1] == '3':  # 处理连续两个三声变调
                pinyin[j] = pinyin[j][:-1] + '2'
                # print('change tone ', chn_char[i:i+2], pinyin[j:j + 2])
            if pinyin[j][:-1] not in pinyin_dict:  # 处理儿化音
                assert chn_char[i + 1] == "儿", f"current_char : {cur_char}, next_char: {chn_char[i+1]}, cur_pinyin: {pinyin[j]}"
                assert pinyin[j][-2] == "r"
                tone = pinyin[j][-1]
                a = pinyin[j][:-2]
                # a1, a2 = pinyin_dict[a]
                # result += [a1, a2 + tone, "er5"]
                result += [a + tone, er5]
                if i + 2 < char_len and chn_char[i + 2] != "#":
                    result.append("#0")
                i += 2
                j += 1
            else:
                tone = pinyin[j][-1]
                a = pinyin[j][:-1]
                a1, a2 = pinyin_dict[a] # a="wen" a1="^", a2="en"
                # result += [a1, a2 + tone]  # result = [zh, ong1, ^,en2]
                result.append(a+tone)
                # if i + 1 < char_len and chn_char[i + 1] != "#":  # 每个字后面接一个#0
                    # result.append("#0")

                i += 1
                j += 1

        # TODO support English alpha
        # elif is_alpha(cur_char):
        #     result += ALPHA_PHONE_DICT[cur_char.upper()]
        #     if i + 1 < char_len and chn_char[i + 1] not in "#、，。！？：" :  # 每个字后面接一个#0
        #         result.append("#0")
        #     i += 1
        #     j += 1  # baker alpha dataset "ABC" in pinyin
        elif cur_char == "#":
            result.append(chn_char[i : i + 2])
            i += 2
        elif cur_char in _PAUSE_SYMBOL:  # 遇到标点符号，添加停顿
            result.pop()  # 去掉#0
            result.append("#3")
            i += 1
        else:
            # ignore the unknown char 
            # result.append(chn_char[i])
            i += 1
    if result[-1] == "#0":  # 去掉最后的#0，改为sil
        result = result[:-1]
    # if result[-1] != "sil":
    #     result.append("sil")
    assert j == len(pinyin)
    return result

# _PAUSE_SYMBOL = {'、', '，', '。', ',', '！', '!', '？', '：', ':', '《', '》', '·', '（', '）', '(', ')'}
_PAUSE_SYMBOL = {'.':'.', '、':',', '，':',', '。':'.', ',':',', '！':'!', '!':'!', '？':'?', '?':'?', '：':',', ':':',', '——':','}

class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def checkErHuaYin(text, GT_pinyin):
    new_pinyin = []
    check_pattern = re.compile("[\\t\.\!\?\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）“”：；]+")
    check_text = check_pattern.sub('', text)
    if len(check_text) > len(GT_pinyin) and '儿' in check_text:
        # print('Size mismatch: ', check_text, len(check_text), '\n', GT_pinyin, len(GT_pinyin))
        for i in range(len(GT_pinyin)):
            if GT_pinyin[i][-2] == 'r' and GT_pinyin[i][:2] != 'er' and check_text[i + 1] == '儿':
                new_pinyin.append(GT_pinyin[i][:-2] + GT_pinyin[i][-1])
                new_pinyin.append('er5')
                replace_word = check_text[i:i + 2]
                replace_pattern = re.compile(replace_word)
                # text = replace_pattern.sub(replace_word[:-1], text)
                check_text = replace_pattern.sub(replace_word[:-1], check_text, count=1)
            else:
                new_pinyin.append(GT_pinyin[i])
        GT_pinyin = new_pinyin
    return GT_pinyin


def change_tone_in_bu_or_yi(chars, pinyin_list):
    location_yi = [m.start() for m in re.finditer(r'一', chars)]
    location_bu = [m.start() for m in re.finditer(r'不', chars)]
    # print('data: ', chars, pinyin_list, location_yi, location_bu)
    for l in location_yi:
        if l > 0 and l<len(chars) and chars[l-1]==chars[l+1]:
            pinyin_list[l] = 'yi5'
        elif l<len(chars) and pinyin_list[l+1][-1] == '4':
                pinyin_list[l] = 'yi2'
    for l in location_bu:
        if l<len(chars) and pinyin_list[l+1][-1] == '4':
            pinyin_list[l] = 'bu2'
    return pinyin_list


def txt2pinyin(text, pinyin_parser):
    phonemes = []
    text = NSWNormalizer(text.strip()).normalize().upper()
    texts = text.split(' ')
    for text in texts:
        text_list = list(jieba.cut(text))
        for words in text_list:
            # print('words: ', words)
            if words in _PAUSE_SYMBOL:
                # phonemes.append('#2')
                phonemes[-1] += _PAUSE_SYMBOL[words]
            elif re.search("[\u4e00-\u9fa5]+", words):
                pinyin = pinyin_parser(words, style=Style.TONE3, errors="ignore")
                new_pinyin = []
                for x in pinyin:
                    x = "".join(x)
                    if "#" not in x:
                        new_pinyin.append(x)
                new_pinyin = change_tone_in_bu_or_yi(words, new_pinyin) if len(words)>1 and words[-1] not in {"一","不"} else new_pinyin
                phoneme = get_phoneme_from_char_and_pinyin(words, new_pinyin) # phoneme seq: [sil c e4 #0 sh iii4 #0 ^ uen2 #0 b en3 sil]  string 的list
                phonemes += phoneme
            elif re.search(r"[a-zA-Z]", words):
                phonemes.append(words.upper())
                # phonemes.append("#1")
    phones = " ".join(phonemes)
    return phones



def process_batch(text_list, save_dir):
    my_pinyin = Pinyin(MyConverter())
    pinyin_parser = my_pinyin.pinyin

    for text_info in tqdm(text_list):
        try:
            name, text = text_info
            save_path = os.path.join(save_dir, name+".txt")
            phones = txt2pinyin(text, pinyin_parser)
            open(save_path, 'w', encoding='utf-8').write(phones)
        except Exception as e:
            print(text_info, e)

def parallel_process(filenames, num_processes, save_dir):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, chunk, save_dir))

        for task in tqdm(tasks):
            task.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_file", type=str, default="", help="path to input text file")
    parser.add_argument(
        "--save_dir", type=str, default="", help="path to output text file")
    parser.add_argument( 
        '--workers', type=int, default=4, help='You are advised to set the number of processes to the same as the number of CPU cores')
    args = parser.parse_args()

    sampling_rate = 16000

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = open(args.text_file, 'r', encoding='utf-8').readlines()
    filenames = [x.strip().split('\t') for x in tqdm(filenames)]
    filenames = [[x[0], x[-1]] for x in tqdm(filenames)]
    # shuffle(filenames)
    print(len(filenames))
    multiprocessing.set_start_method("spawn", force=True)

    if args.workers == 0:
        args.workers = os.cpu_count()
    
    parallel_process(filenames, args.workers, args.save_dir)


#################################################################################



