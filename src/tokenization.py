init_token = '<sos>'
eos_token='<eos>'

def tokenize_seq(text):
    tokens = []
    for token in text.split():
        if token in [init_token, eos_token]:
            tokens.append(token)
        else:
            if len(token) == 3:
                tokens.append(token)
            else:
                tokens.extend(list(token))
    return tokens

def tokenize_aa(protein):
    return list(protein)


def numericalize(text, vocab, sos_token=init_token, eos_token=eos_token):
    tokens = [sos_token] + tokenize_seq(text) + [eos_token]
    return [vocab[token] if token in vocab else vocab.get('<unk>', 0) for token in tokens]