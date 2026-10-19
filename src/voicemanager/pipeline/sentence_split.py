"""Sentence splitter for TTS text, copied from OpenVoice.

Copied from openvoice/utils.py in https://github.com/myshell-ai/OpenVoice (commit
41aaecdc) so the default audiocpp engine can split tts_stream text without OpenVoice
installed. The behavior is unchanged; only the CJK punctuation is written as escapes.

Copyright 2024 MyShell.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import re


def split_sentence(text: str, min_len: int = 10, language_str: str = "[EN]") -> list[str]:
    if language_str in ["EN"]:
        return split_sentences_latin(text, min_len=min_len)
    return split_sentences_zh(text, min_len=min_len)


def split_sentences_latin(text: str, min_len: int = 10) -> list[str]:
    """Split long text into short sentences of more than min_len words."""
    # CJK full stop, exclamation, question mark and semicolon; CJK comma; curly quotes
    text = re.sub("[。！？；]", ".", text)
    text = re.sub("[，]", ",", text)
    text = re.sub("[“”]", '"', text)
    text = re.sub("[‘’]", "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"«»]+", "", text)
    text = re.sub("[\n\t ]+", " ", text)
    text = re.sub("([,.!?;])", r"\1 $#!", text)
    sentences = [s.strip() for s in text.split("$#!")]
    if len(sentences[-1]) == 0:
        del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent.split(" "))
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(" ".join(new_sent))
            new_sent = []
    return merge_short_sentences_latin(new_sentences)


def merge_short_sentences_latin(sens: list[str]) -> list[str]:
    """Merge a sentence of two words or fewer into its neighbour."""
    sens_out = []
    for s in sens:
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except IndexError:
        pass
    return sens_out


def split_sentences_zh(text: str, min_len: int = 10) -> list[str]:
    """Split long text into short sentences of more than min_len characters."""
    text = re.sub("[。！？；]", ".", text)
    text = re.sub("[，]", ",", text)
    text = re.sub("[\n\t ]+", " ", text)
    text = re.sub("([,.!?;])", r"\1 $#!", text)
    sentences = [s.strip() for s in text.split("$#!")]
    if len(sentences[-1]) == 0:
        del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(" ".join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def merge_short_sentences_zh(sens: list[str]) -> list[str]:
    """Merge a sentence of two characters or fewer into its neighbour."""
    sens_out = []
    for s in sens:
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except IndexError:
        pass
    return sens_out
