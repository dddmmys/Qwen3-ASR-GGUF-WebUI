# -*- coding: utf-8 -*-
"""
路由模块
定义所有 Flask 路由，处理前端请求并与各功能模块交互。
"""

from flask import request, jsonify, send_from_directory
import os
import json
import tempfile
import shutil

# 导入自定义模块
import file_manager
import transcribe_runner
from config import UPLOAD_FOLDER
from task_manager import task_manager
from task_manager import Task
from task_manager import PENDING, PROCESSING, COMPLETED, FAILED
from waveform_generator import generate_waveform


def register_routes(app):
    """
    将所有路由注册到 Flask 应用实例上。
    参数:
        app: Flask 应用实例
    """

    # ==================== 静态文件路由 ====================

    @app.route("/")
    def index():
        """
        提供前端主页面 index.html。
        返回:
            HTML 文件内容
        """
        # 获取当前文件所在目录的路径
        current_dir = os.path.dirname(__file__)
        # 从该目录发送 index.html 文件
        return send_from_directory(current_dir, "index.html")

    @app.route("/style.css")
    def serve_css():
        """
        提供前端样式文件 style.css。
        返回:
            CSS 文件内容
        """
        current_dir = os.path.dirname(__file__)
        return send_from_directory(current_dir, "style.css")

    @app.route("/script.js")
    def serve_js():
        """
        提供前端 JavaScript 文件 script.js。
        返回:
            JS 文件内容
        """
        current_dir = os.path.dirname(__file__)
        return send_from_directory(current_dir, "script.js")

    # ==================== 单文件转写路由 ====================

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        """
        处理单个音频文件转写请求。
        请求格式: multipart/form-data
            - audio: 音频文件
            - config: JSON 字符串，包含所有转写参数
        返回:
            JSON 包含识别结果文本、字幕和日志
        """
        # 1. 获取上传的音频文件
        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "No file uploaded"}), 400

        # 2. 解析配置参数
        config_json = request.form.get("config", "{}")
        try:
            params = json.loads(config_json)
        except json.JSONDecodeError as decode_error:
            error_message = f"Invalid config format: {decode_error}"
            return jsonify({"error": error_message}), 400

        # 3. 保存上传的文件
        try:
            saved_filepath = file_manager.save_uploaded_file(audio_file)
            print(f"[DEBUG] 音频文件已保存至: {saved_filepath}")
        except Exception as save_error:
            print(f"[ERROR] 保存文件失败: {save_error}")
            error_message = f"保存文件失败: {str(save_error)}"
            return jsonify({"error": error_message}), 500

        # 4. 清理旧文件（根据当前限制）
        file_manager.clean_old_files()

        # 5. 执行转写
        try:
            # 调用转写运行器，传入文件路径和参数
            result = transcribe_runner.run_transcribe(saved_filepath, params)
            return jsonify(result)
        except RuntimeError as runtime_error:
            # 转写脚本执行失败
            return jsonify({"error": str(runtime_error)}), 500
        except Exception as unknown_error:
            # 其他未知错误
            return jsonify({"error": f"未知错误: {str(unknown_error)}"}), 500

    # ==================== 批量任务路由 ====================

    @app.route("/tasks", methods=["POST"])
    def create_tasks():
        """
        提交批量任务（多个音频文件）。
        请求格式: multipart/form-data
            - files: 多个音频文件（字段名必须为 'files'）
            - config: JSON 字符串，包含所有转写参数
        返回:
            JSON 包含所有创建的任务 ID
        """
        # 1. 获取上传的文件列表
        uploaded_files = request.files.getlist("files")
        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400

        # 2. 解析配置参数
        config_json = request.form.get("config", "{}")
        try:
            params = json.loads(config_json)
        except json.JSONDecodeError as decode_error:
            error_message = f"Invalid config format: {decode_error}"
            return jsonify({"error": error_message}), 400

        # 3. 逐个保存文件并创建任务
        task_ids = []
        for current_file in uploaded_files:
            try:
                # 保存文件到上传目录
                saved_filepath = file_manager.save_uploaded_file(current_file)
                # 添加到任务管理器
                task_id = task_manager.add_task(
                    saved_filepath, current_file.filename, params
                )
                task_ids.append(task_id)
            except Exception as save_error:
                # 记录错误，但继续处理其他文件
                print(f"[ERROR] 保存文件失败: {save_error}")
                # 可以选择将错误信息返回，这里简单跳过

        # 4. 触发清理旧文件
        file_manager.clean_old_files()

        # 5. 返回任务 ID 列表，状态码 202 表示已接受但尚未处理完成
        response_data = {"task_ids": task_ids}
        return jsonify(response_data), 202

    @app.route("/tasks", methods=["GET"])
    def get_tasks():
        """
        获取所有任务的摘要信息。
        返回:
            JSON 数组，每个元素包含任务的基本信息（无结果详情）
        """
        # 获取所有任务对象
        all_tasks = task_manager.get_all_tasks()

        # 构建返回列表
        result_list = []
        for task in all_tasks:
            # 提取需要返回的字段
            task_info = {
                "id": task.id,
                "filename": task.original_filename,
                "status": task.status,
                "created_time": (
                    task.created_time.isoformat() if task.created_time else None
                ),
                "started_time": (
                    task.started_time.isoformat() if task.started_time else None
                ),
                "completed_time": (
                    task.completed_time.isoformat() if task.completed_time else None
                ),
                "error": task.error,
            }
            result_list.append(task_info)

        return jsonify(result_list)

    @app.route("/tasks/<task_id>", methods=["GET"])
    def get_task(task_id):
        """
        获取单个任务的详细信息（包含结果）。
        参数:
            task_id: 任务 ID
        返回:
            JSON 包含任务的所有信息，包括识别结果
        """
        # 从任务管理器获取任务
        task = task_manager.get_task(task_id)
        if task is None:
            return jsonify({"error": "Task not found"}), 404

        # 构建详细结果
        task_detail = {
            "id": task.id,
            "filename": task.original_filename,
            "status": task.status,
            "created_time": (
                task.created_time.isoformat() if task.created_time else None
            ),
            "started_time": (
                task.started_time.isoformat() if task.started_time else None
            ),
            "completed_time": (
                task.completed_time.isoformat() if task.completed_time else None
            ),
            "error": task.error,
            "result": task.result,  # 可能包含 text, srt, log 等
        }
        return jsonify(task_detail)

    @app.route("/tasks/<task_id>", methods=["DELETE"])
    def delete_task(task_id):
        """
        删除一个已完成或失败的任务（正在处理的任务不可删除）。
        参数:
            task_id: 任务 ID
        返回:
            JSON 表示操作结果
        """
        delete_success = task_manager.delete_task(task_id)
        if delete_success:
            return jsonify({"status": "deleted"})
        else:
            error_message = "Task cannot be deleted or not found"
            return jsonify({"error": error_message}), 400

    # ==================== 临时文件管理路由 ====================

    @app.route("/tempfiles", methods=["GET"])
    def tempfiles():
        """
        获取上传目录中的临时文件/文件夹列表以及当前文件数量限制。
        返回:
            JSON 包含 max_uploads 和 files 数组
        """
        # 获取当前限制
        current_limit = file_manager.MAX_UPLOADS
        # 获取文件列表信息
        files_info = file_manager.get_temp_files_info()

        response_data = {
            "max_uploads": current_limit,
            "files": files_info,
        }
        return jsonify(response_data)

    @app.route("/tempfiles/clean", methods=["POST"])
    def tempfiles_clean():
        """
        手动触发清理超出数量限制的旧文件。
        返回:
            JSON 包含清理后的状态和文件列表
        """
        # 执行清理
        file_manager.clean_old_files()
        # 获取清理后的文件列表
        updated_files = file_manager.get_temp_files_info()

        response_data = {
            "status": "cleaned",
            "files": updated_files,
        }
        return jsonify(response_data)

    @app.route("/tempfiles/set_limit", methods=["POST"])
    def tempfiles_set_limit():
        """
        设置新的文件数量限制，并立即清理超出文件。
        请求 JSON 格式:
            {
                "max_uploads": 20
            }
        返回:
            JSON 包含操作结果、新的限制和更新后的文件列表
        """
        # 解析 JSON 请求体
        request_data = request.get_json()
        if request_data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        # 获取新的限制值
        new_limit_value = request_data.get("max_uploads")
        if new_limit_value is None:
            return jsonify({"error": "Missing max_uploads field"}), 400

        try:
            new_limit = int(new_limit_value)
        except (ValueError, TypeError):
            return jsonify({"error": "max_uploads must be an integer"}), 400

        # 调用文件管理器设置新限制
        update_success = file_manager.set_max_uploads_limit(new_limit)
        if update_success:
            # 获取更新后的文件列表
            updated_files = file_manager.get_temp_files_info()
            response_data = {
                "status": "ok",
                "max_uploads": file_manager.MAX_UPLOADS,
                "files": updated_files,
            }
            return jsonify(response_data)
        else:
            return jsonify({"error": "Invalid limit"}), 400

    # ==================== 波形生成路由 ====================

    @app.route("/waveform", methods=["POST"])
    def waveform():
        """
        生成音频文件的波形数据（用于前端绘制波形图）。
        请求格式: multipart/form-data
            - audio: 音频文件
        返回:
            JSON 包含 waveform 数组和 duration（秒）
        """
        # 获取上传的音频文件
        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "No file uploaded"}), 400

        # 获取文件扩展名
        file_extension = os.path.splitext(audio_file.filename)[1]
        # 创建临时文件保存上传的音频
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # 调用波形生成函数
            waveform_data, audio_duration = generate_waveform(temp_file_path)
            response_data = {
                "waveform": waveform_data,
                "duration": audio_duration,
            }
            return jsonify(response_data)
        except Exception as generation_error:
            # 波形生成失败，返回错误信息
            return jsonify({"error": str(generation_error)}), 500
        finally:
            # 无论成功与否，删除临时文件
            os.unlink(temp_file_path)

    # ==================== 临时文件整理路由 ====================

    @app.route("/tempfiles/organize", methods=["POST"])
    def organize_tempfiles():
        """
        整理上传目录中的文件：
        为每个音频文件，将其对应的 .txt/.srt/.json 文件移动到同名文件夹内。
        返回:
            JSON 包含整理结果统计和错误信息
        """
        # 定义支持的音频扩展名集合
        AUDIO_EXTENSIONS = {
            ".mp3",
            ".wav",
            ".flac",
            ".m4a",
            ".mp4",
            ".aac",
            ".ogg",
            ".wma",
        }

        # 初始化计数器和错误列表
        organized_count = 0
        error_messages = []

        # 遍历上传目录中的所有条目
        for item_name in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item_name)

            # 跳过目录（只处理文件）
            if not os.path.isfile(item_path):
                continue

            # 获取文件扩展名（小写）
            file_ext = os.path.splitext(item_name)[1].lower()

            # 跳过本身就是生成文件（.txt/.srt/.json）的情况
            if file_ext in {".txt", ".srt", ".json"}:
                continue

            # 如果不是音频文件，也跳过
            if file_ext not in AUDIO_EXTENSIONS:
                continue

            # 获取不带扩展名的文件名（作为基础名）
            base_name = os.path.splitext(item_name)[0]

            # 收集存在的生成文件
            generated_files = []
            for gen_ext in [".txt", ".srt", ".json"]:
                gen_filename = base_name + gen_ext
                gen_filepath = os.path.join(UPLOAD_FOLDER, gen_filename)
                if os.path.exists(gen_filepath):
                    generated_files.append(gen_filepath)

            # 如果没有生成文件，则无需处理该音频
            if not generated_files:
                continue

            # 创建同名文件夹（如果不存在）
            target_directory = os.path.join(UPLOAD_FOLDER, base_name)
            os.makedirs(target_directory, exist_ok=True)

            # 将每个生成文件移动到该文件夹
            for source_path in generated_files:
                try:
                    shutil.move(source_path, target_directory)
                    organized_count += 1
                except Exception as move_error:
                    error_message = f"移动 {source_path} 失败: {str(move_error)}"
                    error_messages.append(error_message)

        # 返回整理结果
        result_data = {
            "status": "ok",
            "organized_count": organized_count,
            "errors": error_messages,
        }
        return jsonify(result_data)
