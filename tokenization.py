import re


class BaseTokenizer(object):
    def __init__(self, do_lower_case=False, preserved_pattern=None):
        self.do_lower_case = do_lower_case
        self.preserved_pattern = preserved_pattern

    def tokenize_words(self, text):
        raise NotImplementedError

    def tokenize(self, text):
        if self.preserved_pattern is not None:
            tokens = []
            split_texts = self.preserved_pattern.split(text)
            matched_texts = \
                [m.group(0) for m in self.preserved_pattern.finditer(text)] + [None]
            assert len(split_texts) == len(matched_texts)
            for (split_text, matched_text) in zip(split_texts, matched_texts):
                if self.do_lower_case:
                    tokens += [t.lower() for t in self.tokenize_words(split_text)]
                else:
                    tokens += self.tokenize_words(split_text)

                if matched_text is not None:
                    tokens += [matched_text]
        else:
            if self.do_lower_case:
                tokens = [t.lower() for t in self.tokenize_words(text)]
            else:
                tokens = self.tokenize_words(text)

        return tokens


class RegExpTokenizer(BaseTokenizer):
    def __init__(self, pattern=r'\w+|\S', do_lower_case=False, preserved_pattern=None):
        super(RegExpTokenizer, self).__init__(do_lower_case, preserved_pattern)
        self.pattern = re.compile(pattern)

    def tokenize_words(self, text):
        tokens = [t.strip() for t in self.pattern.findall(text) if t.strip()]
        return tokens


class NLTKTokenizer(BaseTokenizer):
    def __init__(self, do_lower_case=False, preserved_pattern=None):
        super(NLTKTokenizer, self).__init__(do_lower_case, preserved_pattern)
        from nltk import word_tokenize
        self.nltk_tokenize = word_tokenize

    def tokenize_words(self, text):
        tokens = [t.strip() for t in self.nltk_tokenize(text) if t.strip()]
        return tokens


class MeCabTokenizer(BaseTokenizer):
    def __init__(self, mecab_option='', do_lower_case=False, preserved_pattern=None):
        super(MeCabTokenizer, self).__init__(do_lower_case, preserved_pattern)
        import MeCab
        self.mecab_option = mecab_option
        self.mecab = MeCab.Tagger(self.mecab_option)

    def tokenize_words(self, text):
        tokens = []
        for line in self.mecab.parse(text).split('\n'):
            if line == 'EOS':
                break

            token = line.split('\t')[0].strip()
            if not token:
                continue

            tokens.append(token)

        return tokens
