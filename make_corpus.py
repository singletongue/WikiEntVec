import re
import json
import gzip
import argparse
from collections import OrderedDict

from logzero import logger

from tokenization import RegExpTokenizer, NLTKTokenizer, MeCabTokenizer


regex_spaces = re.compile(r'\s+')
regex_title_paren = re.compile(r' \([^()].+?\)$')
regex_hyperlink = re.compile(r'\[\[(.+?)\]\]')
regex_entity = re.compile(r'##[^#]+?##')


def main(args):
    logger.info('initializing a tokenizer')
    if args.tokenizer == 'regexp':
        tokenizer = RegExpTokenizer(do_lower_case=args.do_lower_case,
                                    preserved_pattern=regex_entity)
    elif args.tokenizer == 'nltk':
        tokenizer = NLTKTokenizer(do_lower_case=args.do_lower_case,
                                  preserved_pattern=regex_entity)
    elif args.tokenizer == 'mecab':
        tokenizer = MeCabTokenizer(mecab_option=args.tokenizer_option,
                                   do_lower_case=args.do_lower_case,
                                   preserved_pattern=regex_entity)
    else:
        raise RuntimeError(f'Invalid tokenizer: {args.tokenizer}')


    redirects = dict()
    if args.do_resolve_redirects:
        logger.info('loading redirect information')
        with gzip.open(args.cirrus_file, 'rt') as fi:
            for line in fi:
                json_item = json.loads(line)
                if 'title' not in json_item:
                    continue

                if 'redirect' not in json_item:
                    continue

                dst_title = json_item['title']
                redirects[dst_title] = dst_title
                for redirect_item in json_item['redirect']:
                    if redirect_item['namespace'] == 0:
                        src_title = redirect_item['title']
                        redirects.setdefault(src_title, dst_title)

    logger.info('generating corpus for training')
    n_processed = 0
    with gzip.open(args.cirrus_file, 'rt') as fi, \
         open(args.output_file, 'wt') as fo:
        for line in fi:
            json_item = json.loads(line)
            if 'title' not in json_item:
                continue

            title = json_item['title']
            text = regex_spaces.sub(' ', json_item['text'])

            hyperlinks = dict()
            title_without_paren = regex_title_paren.sub('', title)
            hyperlinks.setdefault(title_without_paren, title)
            for match in regex_hyperlink.finditer(json_item['source_text']):
                if '|' in match.group(1):
                    (entity, anchor) = match.group(1).split('|', maxsplit=1)
                else:
                    entity = anchor = match.group(1)

                if '#' in entity:
                    entity = entity[:entity.find('#')]

                anchor = anchor.strip()
                entity = entity.strip()

                if args.do_resolve_redirects:
                    entity = redirects.get(entity, '')

                if len(anchor) > 0 and len(entity) > 0:
                    hyperlinks.setdefault(anchor, entity)

            hyperlinks_sorted = OrderedDict(sorted(
                hyperlinks.items(), key=lambda t: len(t[0]), reverse=True))

            replacement_flags = [0] * len(text)
            for (anchor, entity) in hyperlinks_sorted.items():
                cursor = 0
                while cursor < len(text) and anchor in text[cursor:]:
                    start = text.index(anchor, cursor)
                    end = start + len(anchor)
                    if not any(replacement_flags[start:end]):
                        entity_token = f'##{entity}##'.replace(' ', '_')
                        text = text[:start] + entity_token + text[end:]
                        replacement_flags = replacement_flags[:start] \
                            + [1] * len(entity_token) + replacement_flags[end:]
                        assert len(text) == len(replacement_flags)
                        cursor = start + len(entity_token)
                    else:
                        cursor = end

            text = ' '.join(tokenizer.tokenize(text))

            print(text, file=fo)
            n_processed += 1

            if n_processed <= 10:
                logger.info('*** Example ***')
                example_text = text[:400] + '...' if len(text) > 400 else text
                logger.info(example_text)

            if n_processed % 10000 == 0:
                logger.info(f'processed: {n_processed}')

    if n_processed % 10000 != 0:
        logger.info(f'processed: {n_processed}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cirrus_file', type=str, required=True,
        help='Wikipedia Cirrussearch content dump file (.json.gz)')
    parser.add_argument('--output_file', type=str, required=True,
        help='output corpus file (.txt)')
    parser.add_argument('--tokenizer', default='regexp',
        help='tokenizer type [regexp]')
    parser.add_argument('--do_lower_case', action='store_true',
        help='lowercase words (not applied to NEs)')
    parser.add_argument('--do_resolve_redirects', action='store_true',
        help='resolve redirects of entity names')
    parser.add_argument('--tokenizer_option', type=str, default='',
        help='option string passed to the tokenizer')
    args = parser.parse_args()
    main(args)
