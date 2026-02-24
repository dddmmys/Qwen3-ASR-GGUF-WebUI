# -*- coding: utf-8 -*-
"""
转写运行器模块
负责调用外部 transcribe.py 脚本处理音频文件，并返回识别结果。
该模块封装了命令行构建、子进程执行、结果文件读取等操作。
"""

import os
import subprocess
from config import BASE_DIR, PYTHON_EXE


def run_transcribe(audio_filepath, params=None):
    """
    调用 transcribe.py 处理音频文件，返回识别结果。

    该函数根据传入的参数动态构建命令行，执行子进程，然后读取生成的
    .txt、.srt、.json 文件，并将内容封装为字典返回。

    参数:
        audio_filepath (str): 音频文件的完整路径
        params (dict, optional): 包含所有转写参数的字典。
            例如:
            {
                "precision": "int4",
                "timestamp": True,
                "use_dml": True,
                "use_vulkan": True,
                "n_ctx": 2048,
                "language": "Chinese",
                "context": "some prompt",
                "temperature": 0.4,
                "seek_start": 0.0,
                "duration": null,
                "chunk_size": 40.0,
                "memory_num": 1,
                "verbose": True,
                "yes": False
            }
            如果某个参数未提供，则使用 transcribe.py 的默认值。

    返回:
        dict: 包含以下字段的字典：
            - "text": 识别出的文本内容（字符串）
            - "srt": 字幕内容（字符串，SRT格式）
            - "log": 命令行标准输出日志（字符串）
            - "json": （可选）时间戳数据（字典，如果存在）

    抛出:
        RuntimeError: 如果子进程返回非零退出码，或超时，或其他执行异常。
    """
    # 如果 params 为 None，初始化为空字典
    if params is None:
        params = {}

    # ==================== 构建命令行 ====================
    # 基础命令：Python解释器 + 脚本名
    cmd = [PYTHON_EXE, "transcribe.py"]

    # 1. 处理字符串类型的选项（如 --prec, --language, --context）
    str_options_mapping = {
        "precision": "--prec",
        "language": "--language",
        "context": "--context",
    }

    for option_key, command_flag in str_options_mapping.items():
        # 获取参数值
        value = params.get(option_key)

        # 只有当值存在且不为空字符串时才添加
        if value is not None and value != "":
            cmd.append(command_flag)
            cmd.append(str(value))

    # 2. 处理数值类型的选项（如 --n-ctx, --temperature 等）
    num_options_mapping = {
        "n_ctx": "--n-ctx",
        "temperature": "--temperature",
        "seek_start": "--seek-start",
        "duration": "--duration",
        "chunk_size": "--chunk-size",
        "memory_num": "--memory-num",
    }

    for option_key, command_flag in num_options_mapping.items():
        value = params.get(option_key)

        # 只有当值存在且不为空时才添加（数值可能为0，所以不能检查空字符串）
        if value is not None and value != "":
            cmd.append(command_flag)
            cmd.append(str(value))

    # 3. 处理布尔类型的选项（可能有 --no-* 形式）
    # 每个选项对应一个元组 (true_flag, false_flag)
    bool_options_mapping = {
        "timestamp": ("--timestamp", "--no-ts"),
        "use_dml": ("--dml", "--no-dml"),
        "use_vulkan": ("--vulkan", "--no-vulkan"),
        "verbose": ("--verbose", "--quiet"),
        "yes": ("--yes", ""),  # --yes 没有对应的 --no-yes
    }

    for option_key, (true_flag, false_flag) in bool_options_mapping.items():
        value = params.get(option_key)

        if value is True:
            # 如果参数为 True，添加对应的标志
            cmd.append(true_flag)
        elif value is False and false_flag:
            # 如果参数为 False，且有对应的 --no-* 标志，则添加
            cmd.append(false_flag)
        # 如果 value 为 None，则不添加任何标志（使用默认值）

    # 4. 最后添加音频文件路径
    cmd.append(audio_filepath)

    # 打印最终的命令行，便于调试
    command_string = " ".join(cmd)
    print(f"[DEBUG] 最终命令: {command_string}")

    # ==================== 设置子进程环境 ====================
    # 复制当前环境变量
    process_env = os.environ.copy()
    # 强制子进程使用 UTF-8 编码输出，避免 Windows 控制台编码问题
    process_env["PYTHONIOENCODING"] = "utf-8"

    # ==================== 执行子进程 ====================
    try:
        # 使用 subprocess.run 执行命令
        completed_process = subprocess.run(
            cmd,
            capture_output=True,  # 捕获 stdout 和 stderr
            encoding="utf-8",  # 使用 UTF-8 解码输出
            cwd=BASE_DIR,  # 设置工作目录为项目根目录
            timeout=300,  # 超时时间 300 秒
            env=process_env,  # 使用自定义环境变量
        )

        # 检查返回码
        if completed_process.returncode != 0:
            error_message = (
                f"转写脚本执行失败 (code {completed_process.returncode}): "
                f"{completed_process.stderr}"
            )
            raise RuntimeError(error_message)

        # ==================== 读取结果文件 ====================
        # 获取音频文件的基本名称（不含扩展名）
        base_name = os.path.splitext(audio_filepath)[0]

        # 构建可能的输出文件路径
        txt_filepath = base_name + ".txt"
        srt_filepath = base_name + ".srt"
        json_filepath = base_name + ".json"

        # 初始化响应字典
        response_data = {
            "text": "",
            "srt": "",
            "log": completed_process.stdout,
        }

        # 读取文本文件（如果存在）
        if os.path.exists(txt_filepath):
            with open(txt_filepath, "r", encoding="utf-8") as text_file:
                response_data["text"] = text_file.read()

        # 读取字幕文件（如果存在）
        if os.path.exists(srt_filepath):
            with open(srt_filepath, "r", encoding="utf-8") as srt_file:
                response_data["srt"] = srt_file.read()

        # 读取 JSON 文件（如果存在）
        if os.path.exists(json_filepath):
            import json  # 仅在需要时导入

            with open(json_filepath, "r", encoding="utf-8") as json_file:
                response_data["json"] = json.load(json_file)

        # 返回结果
        return response_data

    except subprocess.TimeoutExpired:
        # 超时异常
        raise RuntimeError("转写超时")
    except Exception as error:
        # 其他所有异常（包括文件读取错误等）
        raise RuntimeError(f"执行异常: {str(error)}")
