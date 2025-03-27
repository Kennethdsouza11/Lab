# WRITE A PROGRAM TO SPLIT A WORD INTO PAIR’S AT ALL POSSIBLE POSITIONS. FOR EXAMPLE, CARRIED WILL BE SPLIT INTO {C, ARRIED, CA ,RRIED, CAR, RIED, CARR, IED, CARRI, ED, CARRI, D}.


def split_words(word):
    pairs = [(word[:i], word[i:]) for i in range(1, len(word))]
    return pairs


word = "CARRIED"
result = split_words(word)
print(result)
