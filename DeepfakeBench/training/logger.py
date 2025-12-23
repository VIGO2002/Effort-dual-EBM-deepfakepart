import os
import logging

# --- 补全开始：添加缺失的 RankFilter 类 ---
class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0
# --- 补全结束 ---

def create_logger(log_path):
    # Create log path
    if os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)  # Set logging level for stream handler
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger