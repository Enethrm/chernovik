from config import cfg


def get_text_from_config(text, block):
    text_ans = cfg[block][text]
    return text_ans