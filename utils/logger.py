import os
import os.path as osp
import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

logger_init = False

def init_logger(_log_file, dir='log/'):
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H.%M.%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)

    if _log_file is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        log_file = osp.join(dir, _log_file + '.log')
        fhlr = logging.FileHandler(log_file)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    global logger_init
    logger_init = True