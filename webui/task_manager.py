# -*- coding: utf-8 -*-
"""
任务管理模块
负责批量任务的创建、排队、执行和状态跟踪。
使用单线程工作队列串行处理任务，避免并发冲突。
"""

import threading
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
import queue

# 导入转写运行器
from transcribe_runner import run_transcribe

# ==================== 任务状态常量 ====================
"""任务可能处于的生命周期状态"""
PENDING = "pending"  # 等待处理
PROCESSING = "processing"  # 正在处理中
COMPLETED = "completed"  # 已完成
FAILED = "failed"  # 失败


class Task:
    """
    单个任务的数据模型。
    存储任务的元数据、参数、结果和状态信息。
    """

    def __init__(self, task_id, file_path, original_filename, params):
        """
        初始化一个任务实例。

        参数:
            task_id (str): 任务的唯一标识符
            file_path (str): 音频文件在磁盘上的完整路径
            original_filename (str): 原始文件名（用于显示）
            params (dict): 转写参数配置
        """
        # 任务唯一标识
        self.id = task_id

        # 音频文件路径
        self.file_path = file_path

        # 原始文件名（不含路径）
        self.original_filename = original_filename

        # 转写参数（精度、语言等）
        self.params = params

        # 当前状态，初始为 PENDING
        self.status = PENDING

        # 处理进度（0-100），预留字段暂未使用
        self.progress = 0

        # 识别结果，包含 text, srt, log 等
        self.result = None

        # 错误信息（如果失败）
        self.error = None

        # 任务创建时间
        self.created_time = datetime.now()

        # 任务开始处理时间
        self.started_time = None

        # 任务完成时间（无论成功或失败）
        self.completed_time = None


class TaskManager:
    """
    任务管理器，负责任务的添加、排队、执行和查询。
    内部使用一个工作线程和一个队列实现串行处理。
    """

    def __init__(self, max_workers=1):
        """
        初始化任务管理器。

        参数:
            max_workers (int): 最大并发工作线程数（当前固定为1，串行处理）
        """
        # 存储所有任务对象的字典，键为任务ID
        self.tasks: Dict[str, Task] = {}

        # 任务队列，存放待处理的任务ID
        self.queue = queue.Queue()

        # 线程锁，用于保护共享数据（tasks 字典）
        self.lock = threading.Lock()

        # 创建工作线程（守护线程，主程序退出时自动结束）
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        # 最大工作线程数（当前未实现多线程，仅作标记）
        self.max_workers = max_workers

    def _worker(self):
        """
        工作线程的主循环。
        不断从队列中取出任务ID，执行对应的任务，并更新状态。
        """
        while True:
            # 从队列获取一个任务ID（阻塞直到有任务）
            task_id = self.queue.get()

            # 加锁保护，从字典中获取任务对象
            with self.lock:
                task = self.tasks.get(task_id)

                # 如果任务不存在或状态不是 PENDING，则跳过
                if task is None or task.status != PENDING:
                    # 标记任务完成（队列计数减一）
                    self.queue.task_done()
                    continue

                # 更新任务状态为 PROCESSING
                task.status = PROCESSING
                task.started_time = datetime.now()

            # 执行任务（不在锁内，避免阻塞其他操作）
            try:
                # 调用转写运行器执行实际转写
                result = run_transcribe(task.file_path, task.params)

                # 加锁更新任务结果和状态
                with self.lock:
                    task.result = result
                    task.status = COMPLETED

            except Exception as error:
                # 捕获任何异常，标记任务为失败
                with self.lock:
                    task.error = str(error)
                    task.status = FAILED

            finally:
                # 无论成功或失败，记录完成时间并标记队列任务完成
                with self.lock:
                    task.completed_time = datetime.now()

                # 标记队列任务完成
                self.queue.task_done()

    def add_task(self, file_path, original_filename, params) -> str:
        """
        添加一个新任务到队列中。

        参数:
            file_path (str): 音频文件的完整路径
            original_filename (str): 原始文件名
            params (dict): 转写参数

        返回:
            str: 新生成的任务ID
        """
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())

        # 创建任务对象
        task = Task(
            task_id=task_id,
            file_path=file_path,
            original_filename=original_filename,
            params=params,
        )

        # 加锁将任务加入字典
        with self.lock:
            self.tasks[task_id] = task

        # 将任务ID放入队列（非阻塞）
        self.queue.put(task_id)

        # 返回任务ID
        return task_id

    def get_task(self, task_id) -> Optional[Task]:
        """
        根据任务ID获取任务对象。

        参数:
            task_id (str): 任务ID

        返回:
            Optional[Task]: 如果存在则返回Task对象，否则返回None
        """
        with self.lock:
            task = self.tasks.get(task_id)
            # 直接返回任务对象（调用者不应修改）
            return task

    def get_all_tasks(self) -> List[Task]:
        """
        获取所有任务列表，按创建时间倒序排序。

        返回:
            List[Task]: 任务对象列表
        """
        with self.lock:
            # 获取所有任务值
            tasks_list = list(self.tasks.values())

            # 按创建时间从新到旧排序
            tasks_list.sort(key=lambda task: task.created_time, reverse=True)

            return tasks_list

    def delete_task(self, task_id) -> bool:
        """
        删除一个已完成或失败的任务（正在处理的任务不能删除）。

        参数:
            task_id (str): 任务ID

        返回:
            bool: 删除成功返回True，否则返回False
        """
        with self.lock:
            # 检查任务是否存在
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]

            # 只允许删除已完成或失败的任务
            # 正在处理或等待中的任务不允许删除
            if task.status in [COMPLETED, FAILED]:
                # 从字典中移除
                del self.tasks[task_id]
                return True
            else:
                # 任务状态为 PENDING 或 PROCESSING，不能删除
                return False


# ==================== 全局单例实例 ====================
# 创建一个全局的任务管理器实例，供整个应用使用
# 使用单例模式，确保所有任务都在同一个队列中处理
task_manager = TaskManager(max_workers=1)
