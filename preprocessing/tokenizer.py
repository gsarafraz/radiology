def tokenizer_radiology(sample):
    
    my_tagger = POSTagger(tagging_model="wapiti")  # tagging_model = "wapiti" or "stanford". "wapiti" is faster than "stanford"
    tags = my_tagger.parse(my_tokenizer.tokenize_words(sample.translate(str.maketrans('', '', string.punctuation))))
    func = lambda x: x[0] if (re.match(pattern,x[1])) else None

    verbs = np.array([func(tag) for tag in tags])
    verbs = set(verbs[verbs != None])

    for verb in verbs:
        sample = sample.replace(verb, verb + '.')
    sample = sample.replace('..','.')
    sample = sample.replace(':','.')
    
    my_tokenizer = Tokenizer()
    sentences = my_tokenizer.tokenize_sentences(sample)

    return sentences