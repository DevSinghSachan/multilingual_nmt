import itertools
import os
import csv
import sys
from collections import defaultdict
from six.moves import zip


class MultiLingualAlignedCorpusReader(object):
    """Handles the case of reading TED talk files
    """
    def __init__(self, corpus_path, vocab=None, delimiter='\t',
                 target_token=True, bilingual=True, corpus_type='file',
                 lang_dict={'source': ['fr'], 'target': ['en']},
                 zero_shot=False, eval_lang_dict=None):
        self.empty_line_flag = 'NULL'
        self.corpus_path = corpus_path
        self.delimiter = delimiter
        self.bilingual = bilingual
        self.lang_dict = lang_dict
        self.lang_set = set()
        self.target_token = target_token
        self.zero_shot = zero_shot
        self.eval_lang_dict = eval_lang_dict
        self.corpus_type = corpus_type

        for list_ in self.lang_dict.values():
            for lang in list_:
                self.lang_set.add(lang)

        self.data = dict()
        self.data['train'] = self.read_aligned_corpus(split_type='train')
        self.data['test'] = self.read_aligned_corpus(split_type='test')
        self.data['dev'] = self.read_aligned_corpus(split_type='dev')

    def read_data(self, file_loc_):
        data_list = list()
        with open(file_loc_) as fp:
            for line in fp:
                try:
                    text = line.strip()
                except IndexError:
                    text = self.empty_line_flag
                data_list.append(text)
        return data_list

    def filter_text(self, dict_):
        if self.target_token:
            field_index = 1
        else:
            field_index = 0
        data_dict = defaultdict(list)
        list1 = dict_['source']
        list2 = dict_['target']
        for sent1, sent2 in zip(list1, list2):
            try:
                src_sent = ' '.join(sent1.split()[field_index: ])
            except IndexError:
                src_sent = 'NULL'

            if src_sent.find(self.empty_line_flag) != -1 or len(src_sent) == 0:
                continue

            elif sent2.find(self.empty_line_flag) != -1 or len(sent2) == 0:
                continue

            else:
                data_dict['source'].append(sent1)
                data_dict['target'].append(sent2)
        return data_dict

    def read_file(self, split_type, data_type):
        return self.data[split_type][data_type]

    def save_file(self, path_, split_type, data_type):
        with open(path_, 'w') as fp:
            for line in self.data[split_type][data_type]:
                fp.write(line + '\n')

    def add_target_token(self, list_, lang_id):
        new_list = list()
        token = '__' + lang_id + '__'
        for sent in list_:
            new_list.append(token + ' ' + sent)
        return new_list

    def read_from_single_file(self, path_, s_lang, t_lang):
        data_dict = defaultdict(list)
        with open(path_, 'r') as fp:
            reader = csv.DictReader(fp, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                data_dict['source'].append(row[s_lang])
                data_dict['target'].append(row[t_lang])

        if self.target_token:
            text = self.add_target_token(data_dict['source'], t_lang)
            data_dict['source'] = text

        return data_dict['source'], data_dict['target']

    def read_from_directory(self, path_, s_lang, t_lang):
        data_dict = defaultdict(list)

        for talk_dir in os.listdir(path_):
            dir_path = os.path.join(path_, talk_dir)

            talk_lang_set = set([l.split('.')[0] for l in os.listdir(dir_path)])

            if s_lang not in talk_lang_set or t_lang not in talk_lang_set:
                continue

            for infile in os.listdir(dir_path):
                lang = os.path.splitext(infile)[0]

                if lang in self.lang_set:
                    file_path = os.path.join(dir_path, infile)
                    text = self.read_data(file_path)

                    if lang == s_lang:
                        if self.target_token:
                            text = self.add_target_token(text, t_lang)
                            data_dict['source'] += text
                        else:
                            data_dict['source'] += text

                    elif lang == t_lang:
                        data_dict['target'] += text

        return data_dict['source'], data_dict['target']

    def read_aligned_corpus(self, split_type='train'):
        data_dict = defaultdict(list)
        iterable = []
        s_list = []
        t_list = []

        if self.zero_shot:
            if split_type == "train":
                iterable = zip(self.lang_dict['source'], self.lang_dict['target'])
            else:
                iterable = zip(self.eval_lang_dict['source'], self.eval_lang_dict['target'])

        elif self.bilingual:
            iterable = itertools.product(self.lang_dict['source'], self.lang_dict['target'])

        for s_lang, t_lang in iterable:
            if s_lang == t_lang:
                continue
            if self.corpus_type == 'directory':
                split_type_path = os.path.join(self.corpus_path, split_type)
                s_list, t_list = self.read_from_directory(split_type_path, s_lang=s_lang, t_lang=t_lang)

            elif self.corpus_type == 'file':
                split_type_file_path = os.path.join(self.corpus_path, "all_talks_{}.tsv".format(split_type))
                s_list, t_list = self.read_from_single_file(split_type_file_path, s_lang=s_lang, t_lang=t_lang)

            data_dict['source'] += s_list
            data_dict['target'] += t_list

        new_data_dict = self.filter_text(data_dict)
        return new_data_dict


if __name__ == "__main__":

    # Testing the code
    data_path = 'data'
    src_langs = ['ja', 'en']
    tgt_langs = ['ja', 'en']
    DATA_DIR = 'data/{}_{}'.format('+'.join(src_langs), '+'.join(tgt_langs))

    zs_train_lang_dict={'source': src_langs, 'target': tgt_langs}
    zs_eval_lang_dict = {'source': src_langs, 'target': tgt_langs}

    obj = MultiLingualAlignedCorpusReader(corpus_path=data_path,
                                          lang_dict=zs_train_lang_dict,
                                          target_token=True,
                                          corpus_type='file',
                                          eval_lang_dict=zs_eval_lang_dict,
                                          zero_shot=False,
                                          bilingual=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    obj.save_file(os.path.join(DATA_DIR, 'train.src'),
                  split_type='train', data_type='source')
    obj.save_file(os.path.join(DATA_DIR, 'train.tgt'),
                  split_type='train', data_type='target')

    obj.save_file(os.path.join(DATA_DIR, 'test.src'),
                  split_type='test', data_type='source')
    obj.save_file(os.path.join(DATA_DIR, 'test.tgt'),
                  split_type='test', data_type='target')

    obj.save_file(os.path.join(DATA_DIR, 'dev.src'),
                  split_type='dev', data_type='source')
    obj.save_file(os.path.join(DATA_DIR, 'dev.tgt'),
                  split_type='dev', data_type='target')
