from config import Config
from data_utils import Dataset, vocab_build

def build_data(Config):
    """
    Procedure to build data
    Args:
        Config: defines attributes needed in the function
    Returns:
        creates vocab files from the datasets
    """
    # Generators
    train = Dataset(words_filename=Config.source_path,
    tags_filename=Config.source_tgt_path)
    # test = Dataset(words_filename=Config.test_path,
    # tags_filename=Config.test_tgt_path)

    # Build Word and Tag vocab
    # vocab_words, vocab_tags = get_vocabs([train, test])

    # vocab_words.add(UNK)

    # Save vocab
    # write_vocab(vocab_words, Config.words_vocab)
    # write_vocab(vocab_tags, Config.tags_vocab)
    vocab_build(train, Config.min_count, Config.words_vocab)
    

if __name__=="__main__":
    build_data(Config)