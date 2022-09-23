import logging

ROOT_LOGGER_PREFIX = 'qube'

__root_logger = None
__DEFAULT_FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def getLogger(name, level=None):
    root_logger = __get_root_logger()

    root_logger_name = root_logger.name
    if not name.startswith(root_logger_name):
        name = root_logger_name + '.' + name

    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level)

    return logger


def addFileHandler(logger, log_file_path: str, formatter=__DEFAULT_FORMATTER):
    if log_file_path:
        # add fileHandler
        fileHandler = logging.FileHandler(log_file_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)


def addStreamHandler(logger, stream, formatter=__DEFAULT_FORMATTER):
    if stream:
        # add streamHandler
        streamHandler = logging.StreamHandler(stream)
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)


def __get_root_logger():
    global __root_logger
    if not __root_logger:
        __root_logger = logging.getLogger(ROOT_LOGGER_PREFIX)
        __root_logger.setLevel(logging.INFO)  # we must set default root log level!

        # add consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(__DEFAULT_FORMATTER)
        __root_logger.addHandler(consoleHandler)

    return __root_logger
