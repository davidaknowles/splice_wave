# cython: language_level=3

import numpy as np
np.get_include() # do we need this on colab?
cimport cython
#cimport numpy as np
cimport numpy as np
cimport numpy as cnp

import string

cdef dict bases={ 'A':<int>0, 'C':<int>1, 'G':<int>2, 'T':<int>3 }

def create_char_to_int_dict():
    cdef str chars = string.ascii_lowercase + string.digits + ".,'!? "
    cdef dict char_to_int = {char: idx for idx, char in enumerate(chars)}
    return char_to_int

cdef dict char_to_int = create_char_to_int_dict()

@cython.boundscheck(False)
def one_hot_transpose( str string ):
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros( (4,len(string)), dtype=np.float32 )
    cdef int j
    for j in range(len(string)):
        if string[j] in bases: # bases can be 'N' signifying missing: this corresponds to all 0 in the encoding
            res[ bases[ string[j] ], j ]=float(1.0)
    return(res)

@cython.boundscheck(False)
def one_hot( str string ):
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros( (len(string),4), dtype=np.float32 )
    cdef int j
    for j in range(len(string)):
        if string[j] in bases: # bases can be 'N' signifying missing: this corresponds to all 0 in the encoding
            res[ j, bases[ string[j] ] ]=float(1.0)
    return(res)

def wiki_one_hot(seq: str, dtype=np.float32):
    cdef int seq_len = len(seq)
    cdef int char_set_size = len(char_to_int)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_rep = np.zeros((seq_len, char_set_size), dtype=dtype)

    cdef int i, idx
    cdef str ch
    for i in range(seq_len):
        ch = seq[i]
        if ch in char_to_int:
            idx = char_to_int[ch]
            arr_rep[i, idx] = 1.

    return arr_rep

