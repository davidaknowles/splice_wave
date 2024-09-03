import unicodedata
import re
import transcript_data
from datasets import load_dataset
import string
import numpy as np
import torch
import one_hot

def strip_accents_and_selected_punctuation(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r"[^\w\s.,'!?]", '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def create_char_to_int_dict():
    chars = string.ascii_lowercase + string.digits + ".,'!? "
    return {char: idx for idx, char in enumerate(chars)}

char_to_int = create_char_to_int_dict()

def wiki_one_hot_slow(seq, dtype=np.float32):
    seq_len = len(seq)
    arr_rep = np.zeros((seq_len, len(char_to_int)), dtype=dtype) 
    for i in range(seq_len):
        if seq[i] in char_to_int:
            arr_rep[i,char_to_int[seq[i]]] = 1 
    return arr_rep


class WikiDataset(torch.utils.data.Dataset):

    def __init__(self, data, width=1000, mask = False):

        super().__init__()
        self.width = width
        assert width % 2 == 0
        self.data = data
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        cleaned = strip_accents_and_selected_punctuation(self.data[i]["text"])

        l = len(cleaned)
        if l > self.width: 
            start_index = np.random.randint(0, l - self.width + 1)
            cleaned = cleaned[start_index:start_index + self.width]

        oh = one_hot.wiki_one_hot(cleaned) 
        
        if l < self.width: 
            oh = np.pad(oh, ((0, self.width - l), (0,0)))
        
        if self.mask: 
            one_hot_masked = np.copy(oh)
            mask = transcript_data.get_mask_np_efficient(one_hot_masked) 
            return 0, 0, 0, oh, one_hot_masked, mask
        else: 
            return 0, 0, 0, oh

