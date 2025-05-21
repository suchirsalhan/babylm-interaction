# File: baby_llama_clean.py
# -------------------------
# Data cleaning script, taken from the BabyLlama repository (https://github.com/timinar/BabyLlama)
# of Timirsayov and Tastet, 2023.

import re

# START_TOKEN = '<s>'
# END_TOKEN = '</s>'
# PADDING_TOKEN = '<pad>'

START_TOKEN = ''
END_TOKEN = ''
PADDING_TOKEN = ''

def _make_padding_sequence(seq_length):
    return ''.join([END_TOKEN] + seq_length * [PADDING_TOKEN])

def cleanup_simple_wikipedia(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, text) + pad_seq
    return text

def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text

def cleanup_bnc_spoken(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = cleanup_extra_spaces(text)
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, text) + pad_seq
    return text

def cleanup_aochildes(text, seq_length):
    text = cleanup_extra_spaces(text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_gutenberg(text, seq_length):
    # Overall, the text is clean, however some entries don’t seem
    # very useful, e.g. figure captions preceded by a number.
    # Not sure if we should remove them, because that would also
    # remove bullet lists which are otherwise consistent with the
    # surrounding text.
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [PADDING_TOKEN])

def cleanup_open_subtitles(text, seq_length):
    # The text is mostly clean, apart from some subtitle credits
    # such as "Subtitles by ...".
    subtitle_credit_ex = re.compile(r'^.*subtitle.*$\n', re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub('', text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_switchboard(text, seq_length):
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [PADDING_TOKEN])
