from data_utils import read_dictionary, Dataset, vocab_tags
from general_utils import get_logger
from model import Model
from config import Config
import os
import sys


if not os.path.exists(Config.output_path):
    os.makedirs(Config.output_path)

# vocab_words = load_vocab(Config.words_vocab)
# vocab_tags = load_vocab(Config.tags_vocab)
vocab_words = read_dictionary(Config.words_vocab)

# print(vocab_words)
# print(vocab_tags)
# sys.exit(0)

test = Dataset(Config.test_path, Config.test_tgt_path, Config.max_iter)
train = Dataset(Config.source_path, Config.source_tgt_path, Config.max_iter)

logger = get_logger(Config.log_path)

model = Model(Config, ntags=len(vocab_tags), n_words=len(vocab_words), logger=logger)

model.build()

model.train(train, test, vocab_tags, vocab_words)