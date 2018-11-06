import logging

def print_sentence(logger, data):
    """
    Args:
        logger: logger instance
        data: dict
    """
    spacings = [max([len(seq[i]) for seq in data.itervalues()]) for i in range(len(data[data.keys()[0]]))]

    # Compute the word spacing
    for key, seq in data.iteritems():
        to_print=""
        for token, spacing in zip(seq, spacings):
            to_print += token + " " *(spacing-len(token)+1)
        logger.info(to_print)

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    logging.getLogger().addHandler(handler)
    return logger