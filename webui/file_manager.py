# -*- coding: utf-8 -*-
"""
文件管理模块
负责处理上传文件的保存、清理、信息获取以及最大文件数限制的设置。
"""

import os
import glob
from datetime import datetime
from config import UPLOAD_FOLDER, DEFAULT_MAX_UPLOADS

# 确保上传目录存在，如果不存在则创建
# 使用 exist_ok=True 避免目录已存在时报错
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 当前允许的最大文件数，初始化为默认值
MAX_UPLOADS = DEFAULT_MAX_UPLOADS


def save_uploaded_file(file):
    """
    保存上传的文件到上传目录，文件名使用时间戳命名以避免冲突。

    参数:
        file: Flask 的文件对象（来自 request.files）

    返回:
        保存后的完整文件路径（字符串）
    """
    # 获取当前时间，用于生成时间戳文件名
    now = datetime.now()
    # 格式化为：年-月-日-时-分-秒-微秒，确保唯一性
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    # 获取原始文件的扩展名，例如 .wav .mp3
    extension = os.path.splitext(file.filename)[1]
    # 组合文件名：时间戳 + 扩展名
    filename = timestamp + extension
    # 构建完整的文件保存路径
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    # 调用 Flask 文件对象的 save 方法，将文件保存到磁盘
    file.save(filepath)
    # 返回保存路径，供后续处理使用
    return filepath


def clean_old_files(limit=None):
    """
    清理超出数量限制的最旧文件，保留最新的 limit 个文件。

    参数:
        limit: 可选，指定要保留的文件数量。若为 None，则使用全局 MAX_UPLOADS。
    """
    # 如果没有传入 limit，则使用全局设置的最大文件数
    if limit is None:
        limit = MAX_UPLOADS

    # 获取上传目录下所有文件和文件夹的路径列表
    all_items = glob.glob(os.path.join(UPLOAD_FOLDER, "*"))

    # 按修改时间从新到旧排序
    # key 为 os.path.getmtime 获取修改时间戳，reverse=True 表示降序
    all_items.sort(key=os.path.getmtime, reverse=True)

    # 如果当前文件数超过了限制数量，则需要删除最旧的那些文件
    if len(all_items) > limit:
        # 获取需要删除的文件列表（索引 limit 及之后的）
        items_to_delete = all_items[limit:]

        # 遍历待删除的每个文件
        for file_path in items_to_delete:
            try:
                # 尝试删除文件
                os.remove(file_path)
                # 打印删除成功的日志（便于调试）
                print(f"[CLEAN] 删除旧文件: {file_path}")
            except Exception as error:
                # 如果删除失败，捕获异常并打印错误信息
                print(f"[ERROR] 删除文件失败 {file_path}: {error}")


def get_temp_files_info():
    """
    获取上传目录下所有临时文件/文件夹的详细信息。

    返回:
        一个列表，每个元素是一个字典，包含以下字段：
        - name: 文件名
        - size: 文件大小（字节）
        - mtime: 最后修改时间的 ISO 格式字符串
        - type: 类型（“文件夹”或文件扩展名，如“wav”）
    """
    # 获取所有文件/文件夹路径
    all_items = glob.glob(os.path.join(UPLOAD_FOLDER, "*"))

    # 按修改时间从新到旧排序
    all_items.sort(key=os.path.getmtime, reverse=True)

    # 用于存储结果信息的列表
    info_list = []

    # 遍历每个路径
    for item_path in all_items:
        # 获取文件状态信息（大小、修改时间等）
        stat_info = os.stat(item_path)

        # 判断是否是目录
        is_directory = os.path.isdir(item_path)

        # 提取基本文件名（不含路径）
        base_name = os.path.basename(item_path)

        # 确定类型字符串
        if is_directory:
            file_type = "文件夹"
        else:
            # 获取文件扩展名，转为小写，去掉开头的点
            extension = os.path.splitext(base_name)[1].lower().lstrip(".")
            # 如果扩展名为空，则类型为“文件”，否则为扩展名
            if extension:
                file_type = extension
            else:
                file_type = "文件"

        # 将修改时间从时间戳转换为 ISO 格式字符串
        modification_time = datetime.fromtimestamp(stat_info.st_mtime).isoformat()

        # 构建单个文件的信息字典
        file_info = {
            "name": base_name,
            "size": stat_info.st_size,
            "mtime": modification_time,
            "type": file_type,
        }

        # 添加到结果列表
        info_list.append(file_info)

    # 返回所有文件的信息
    return info_list


def set_max_uploads_limit(new_limit):
    """
    设置新的最大文件数量限制，并立即清理超出新限制的文件。

    参数:
        new_limit: 新的限制数量（整数，必须 >= 1）

    返回:
        bool: 设置成功返回 True，否则返回 False
    """
    # 声明我们要修改全局变量 MAX_UPLOADS
    global MAX_UPLOADS

    # 检查新限制是否有效（必须是大于等于 1 的整数）
    if new_limit >= 1:
        # 更新全局最大文件数
        MAX_UPLOADS = new_limit

        # 根据新限制立即清理超出文件
        clean_old_files()

        # 返回成功
        return True
    else:
        # 无效的限制值，返回失败
        return False
