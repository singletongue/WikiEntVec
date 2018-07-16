import re


regex_entity = re.compile(r'<ENT>(.+?)</ENT>')

class Tokenizer(object):
    def __init__(self):
        pass

    def _tokenize(self, text):
        raise NotImplementedError

    def tokenize(self, text, preserving_pattern=None):
        if preserving_pattern is None:
            tokens = self._tokenize(text)
        else:
            tokens = []
            matches = [match.group(0) for match
                       in preserving_pattern.finditer(text)]
            for (snippet, match) in zip(
                    preserving_pattern.split(text), matches + [None]):
                tokens += self._tokenize(snippet)
                if match is not None:
                    tokens.append(match)

        return tokens


class RegExpTokenizer(Tokenizer):
    def __init__(self, pattern=r'\w+|\S'):
        super(RegExpTokenizer, self).__init__()
        self._regex = re.compile(pattern)

    def _tokenize(self, text):
        tokens = self._regex.findall(text)

        return tokens


class NLTKTokenizer(Tokenizer):
    def __init__(self):
        super(NLTKTokenizer, self).__init__()
        from nltk import word_tokenize
        self.word_tokenize = word_tokenize

    def _tokenize(self, text):
        return self.word_tokenize(text)


class MeCabTokenizer(Tokenizer):
    def __init__(self, dic=None, udic=None):
        super(MeCabTokenizer, self).__init__()
        import MeCab
        mecab_options = ['-O wakati']
        if dic:
            mecab_options.append(f'-d {dic}')
        if udic:
            mecab_options.append(f'-u {udic}')

        self._mt = MeCab.Tagger(' '.join(mecab_options))

    def _tokenize(self, text):
        return self._mt.parse(text).strip().split()
