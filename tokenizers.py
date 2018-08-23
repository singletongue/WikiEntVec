import re


regex_entity = re.compile(r'<ENT>(.+?)</ENT>')

class Tokenizer(object):
    def __init__(self, lower):
        self._lower = lower

    def _tokenize(self, text):
        raise NotImplementedError

    def tokenize(self, text, preserving_pattern=None, lower=False):
        if preserving_pattern is None:
            if self._lower:
                tokens = [t.lower() for t in self._tokenize(text)]
            else:
                tokens = self._tokenize(text)
        else:
            tokens = []
            matched_strings = [match.group(0) \
                for match in preserving_pattern.finditer(text)]

            for (snippet, matched_string) in zip(
                    preserving_pattern.split(text), matched_strings + [None]):
                if self._lower:
                    tokens += [t.lower() for t in self._tokenize(snippet)]
                else:
                    tokens += self._tokenize(snippet)

                if matched_string is not None:
                    tokens.append(matched_string)

        return tokens


class RegExpTokenizer(Tokenizer):
    def __init__(self, pattern=r'\w+|\S', lower=False):
        super(RegExpTokenizer, self).__init__(lower)
        self._regex = re.compile(pattern)

    def _tokenize(self, text):
        tokens = self._regex.findall(text)

        return tokens


class NLTKTokenizer(Tokenizer):
    def __init__(self, lower=False):
        super(NLTKTokenizer, self).__init__(lower)
        from nltk import word_tokenize
        self.word_tokenize = word_tokenize

    def _tokenize(self, text):
        return self.word_tokenize(text)


class MeCabTokenizer(Tokenizer):
    def __init__(self, dic=None, udic=None, lower=False):
        super(MeCabTokenizer, self).__init__(lower)
        import MeCab
        mecab_options = ['-O wakati']
        if dic:
            mecab_options.append(f'-d {dic}')
        if udic:
            mecab_options.append(f'-u {udic}')

        self._mt = MeCab.Tagger(' '.join(mecab_options))

    def _tokenize(self, text):
        return self._mt.parse(text).strip().split()
