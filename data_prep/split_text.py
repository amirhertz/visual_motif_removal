import random
from utils.image_utils import NEWLINE_REPLACEMENT_STRING, SPACE_REPLACEMENT_STRING


MAX_LENGTH = 8
UNWANTED_SIGNS = {'\n', ',', '.'}

input_file = 'raw text file'
output_file = ''


def fix_word(word):
    if len(word) == 0:
        return None
    while word[-1] in UNWANTED_SIGNS:
        word = word[: -1]
        if len(word) == 0:
            return None
    return word


def update_group(words_set, line):
    cur_word = ''
    cur_len = 0
    for idx, word in enumerate(line):
        word_len = len(word)
        if idx != 0:
            if cur_len + word_len < MAX_LENGTH and random.random() < 0.7:
                if random.random() < 1:
                    cur_word += NEWLINE_REPLACEMENT_STRING
                else:
                    cur_word += SPACE_REPLACEMENT_STRING
            else:
                words_set.add(cur_word)
                cur_word = ''
                cur_len = 0
        cur_word += word
        cur_len += word_len
    words_set.add(cur_word)


def fill_words():
    words_set = set()
    file = open(input_file, 'r')
    for line in file:
        splitted_line = line.split(' ')
        # splitted_line = [i for i in line]
        fix_line = []
        for idx, word in enumerate(splitted_line):
            fixed_word = fix_word(word)
            if fixed_word is not None:
                fix_line.append(fixed_word)
        update_group(words_set, fix_line)
    file.close()
    return words_set


def write_words(words_set):
    power = len(words_set) - 1
    file = open(output_file, 'w')
    for idx, word in enumerate(words_set):
        file.write(word)
        if idx < power:
            file.write(' ')
    file.close()


def split_book():
    words_set = fill_words()
    write_words(words_set)


if __name__ == '__main__':
    split_book()
