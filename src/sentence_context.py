# -*- coding: utf-8 -*-
"""sentence_context.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qkTlpCZ-cNE8mcy03_9wNvhhbAmwtD9H

# Sentences and Contexts
## Given two Entity Mentions, Extract Relevant Contexts
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plot
import json
import re
import math
import seaborn as sns
# %matplotlib inline

# For NYT dataset
# Extract the span between the two nearest entity mentions in the sentence

def spanEMs(sent, em1, em2):
    """
        input: sent: a cleaned sentence text
               em1, em2: two given entity mentions
        return: the span and the start and end poistions and indicators of the mentions in the sentence
    """
    
    em1starts = []
    # Find all positions where the exact mention 1 word appears
    for m in re.finditer("\\b"+em1+"\\b", sent, re.UNICODE):
        em1starts.append(m.start())
    
    # Find all positions where the exact mention 2 word appears
    em2starts = []
    for m in re.finditer("\\b"+em2+"\\b", sent, re.UNICODE):
        em2starts.append(m.start())
    
    # if an entity doesn't exist in the sentence, return an empty string
    if len(em1starts) == 0 or len(em2starts) == 0:
        return '', -1, -1, -1, -1

    loc1 = em1starts[0]
    loc2 = em2starts[0]
    dist = len(sent)
    for s1 in em1starts:
        for s2 in em2starts:
            if s1 != s2:
                if abs(s1-s2) < dist:
                    dist = abs(s1-s2)
                    loc1 = s1
                    loc2 = s2
    #print(loc1, loc2)
    if loc1 < loc2:
        span = sent[loc1:loc2+len(em2)]
        # return the span and the start and end poistions of the span in the sentence
        return span, loc1, loc2+len(em2),1 ,2
    else:
        span = sent[loc2:loc1+len(em1)]
        # return the span and the start and end poistions of the span in the sentence
        return span, loc2, loc1+len(em1), 2, 1

# For NYT dataset
# Extract span with before and after n=3 words

def baN(sent, em1, em2, n):
    """
        input: sent: a cleaned sentence text
               em1: the first entity mention
               em2: the second entity mention
               n: number of words before and after
        return: the subsentence with span between the entity mentions and 
                n words before first mention and n words after second mention
    """
    span, start, end, _, _ = spanEMs(sent, em1, em2)
    if len(span) == 0:
        return ''
    if n==0:
        return span
    else:
        before = sent[0:start]
        bN = " ".join(before.strip().split()[-n:])
        after = sent[end:]
        aN = " ".join(after.strip().split()[:n])
    
        return bN.strip() + " " + span.strip() + " " + aN

# For NYT dataset
# Extract the 2n surrounding words for each mention,
# n words before and n words after the mention,
# and if the length of span is longer than 2n.

def surrounding2N(sent, em1, em2, n):
    """
        input: sent: a cleaned sentence text
               em1: the first entity mention
               em2: the second entity mention
               n: number of words before and after each mentions (2n surrounding words)
        return: the context consiting of 2n words surrounding em1 and 2n words surrounding em2,
                if the span between the two mentions has more than 2n words;
                otherwise, None
    """

    # for n=0, return the two mentions
    if n==0:
        return em1 + " " + em2

    else:
        span, start, end, _, _ = spanEMs(sent, em1, em2)

        if len(span) == 0:
            return None

        span_words = span.strip().split()

        # If the span between the two mentions doesn't have more than 2n 
        # words, 2n case is covered by the case of span with before and after n words,
        # so return None.
        if len(span_words) - 2 <= 2*n:
            return None
        
        # There are enough words for extracting context with 2n words
        else: 

            beforeFirstEm = sent[0:start]
            bNFirstEm = " ".join(beforeFirstEm.strip().split()[-n:])
            
            # get the n+1 words including and after the first mention
            aNFirstEm = " ".join(span_words[:n+1])

            bNSecondEm = " ".join(span_words[-(n+1):])
            
            afterSecondEm = sent[end:]
            aNSecondEm = " ".join(afterSecondEm.strip().split()[:n])


            return bNFirstEm.strip() + " " + aNFirstEm.strip() + " " + bNSecondEm.strip() + " " + aNSecondEm.strip()