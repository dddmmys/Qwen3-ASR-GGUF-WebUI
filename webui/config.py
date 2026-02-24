import os
import sys

# 项目根目录（即 webui 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 上传文件存放目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

# 使用当前 Python 解释器路径，确保子进程使用相同环境
PYTHON_EXE = sys.executable

# 临时文件默认最大数量
DEFAULT_MAX_UPLOADS = 12
