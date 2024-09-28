import logging
from datetime import datetime, timezone

# Создание лог-файла и начало логирования
def create_logger(project_name, console_log=True):
    # Создание логгера
    logger = logging.getLogger(f'{project_name}')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Вывод лога в файл
    current_time = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    file_handler = logging.FileHandler(f'./{project_name}/log/{current_time}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Вывод лога в консоль
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Закрытие лог-файла и завершение логирования
def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)