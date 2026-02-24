// ============================================================
// 本地音频转字幕图形化工具 - 主脚本
// ============================================================
// 包含所有前端交互逻辑：文件处理、转写请求、任务管理、波形绘制等
// ============================================================

// ==================== DOM 元素获取 ====================
// 所有需要操作的页面元素均在此处获取，便于后续使用

/** 文件拖拽上传区域 */
const dropArea = document.getElementById('dropArea');

/** 单文件上传 input 元素 */
const fileInput = document.getElementById('fileInput');

/** 显示已选择文件信息的区域 */
const fileInfo = document.getElementById('fileInfo');

/** 开始转写按钮 */
const uploadBtn = document.getElementById('uploadBtn');

/** 转写进度提示区域 */
const progress = document.getElementById('progress');

/** 转写结果整体区域（包含文本、字幕、日志等） */
const resultSection = document.getElementById('resultSection');

/** 显示识别文本的区域 */
const textResult = document.getElementById('textResult');

/** 显示 SRT 字幕预览的区域 */
const srtPreview = document.getElementById('srtPreview');

/** 显示命令行日志的区域 */
const logDetails = document.getElementById('logDetails');

/** 下载 SRT 字幕的按钮 */
const downloadSrt = document.getElementById('downloadSrt');

/** 复制识别文本的按钮 */
const copyTextBtn = document.getElementById('copyText');

/** 显示当前临时文件数量限制的 span */
const maxUploadsSpan = document.getElementById('maxUploadsSpan');

/** 设置新限制的输入框 */
const maxUploadsInput = document.getElementById('maxUploadsInput');

/** 应用新限制的按钮 */
const setLimitBtn = document.getElementById('setLimitBtn');

/** 手动清理旧文件的按钮 */
const cleanBtn = document.getElementById('cleanBtn');

/** 刷新临时文件列表的按钮 */
const refreshTempBtn = document.getElementById('refreshTempBtn');

/** 临时文件列表的表格主体 */
const tempFilesBody = document.getElementById('tempFilesBody');

/** 高级设置面板开关（齿轮图标） */
const settingsToggle = document.getElementById('settingsToggle');

/** 高级设置面板容器 */
const settingsPanel = document.getElementById('settingsPanel');

/** 高级设置面板内的关闭按钮 */
const closeSettings = document.getElementById('closeSettings');

/** 批量文件选择 input（隐藏） */
const batchFileInput = document.getElementById('batchFileInput');

/** 批量添加到队列的按钮 */
const batchUploadBtn = document.getElementById('batchUploadBtn');

/** 显示已选批量文件数量的 span */
const batchFileCount = document.getElementById('batchFileCount');

/** 显示已选批量文件列表的区域 */
const batchFileList = document.getElementById('batchFileList');

/** 任务列表的表格主体 */
const taskTableBody = document.getElementById('taskTableBody');

// ==================== 全局状态变量 ====================

/** 当前正在处理的单个文件对象（用于单文件转写） */
let currentFile = null;

/** 当前单文件转写生成的 SRT 内容（用于下载） */
let currentSrtContent = '';

/** 用户选择的批量文件列表 */
let selectedFiles = [];

/** 当前音频文件的 Object URL（用于释放内存） */
let currentAudioUrl = null;

// ==================== 初始化操作 ====================

// 页面加载完成后立即获取临时文件列表
fetchTempFiles();

// 页面加载完成后立即获取任务列表
fetchTaskList();

// ==================== 单个文件转写相关 ====================

/**
 * 点击上传区域时触发文件选择
 */
dropArea.addEventListener('click', function () {
    fileInput.click();
});

/**
 * 拖拽文件进入上传区域时改变边框颜色
 * @param {DragEvent} e - 拖拽事件对象
 */
dropArea.addEventListener('dragover', function (e) {
    e.preventDefault(); // 阻止浏览器默认打开文件行为
    dropArea.style.borderColor = '#0078d4';
});

/**
 * 拖拽文件离开上传区域时恢复边框颜色
 */
dropArea.addEventListener('dragleave', function () {
    dropArea.style.borderColor = '#ccc';
});

/**
 * 拖拽文件释放到上传区域时处理文件
 * @param {DragEvent} e - 拖拽事件对象
 */
dropArea.addEventListener('drop', function (e) {
    e.preventDefault(); // 阻止浏览器默认打开文件行为
    dropArea.style.borderColor = '#ccc';

    // 获取拖拽的文件列表
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        // 将第一个文件设置为 fileInput 的值（虽不能直接设置，但用于后续逻辑）
        // 这里我们只处理第一个文件，因为单文件模式只支持一个
        fileInput.files = files;
        updateFileInfo(files[0]);
    }
});

/**
 * 当通过文件选择框选中文件时触发
 */
fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
        // 有文件被选中
        updateFileInfo(fileInput.files[0]);
    } else {
        // 用户取消了选择
        resetFileInfo();
    }
});

/**
 * 更新界面显示已选择的文件信息
 * @param {File} file - 选中的文件对象
 */
function updateFileInfo(file) {
    // 释放之前的音频 Object URL 以避免内存泄漏
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = null;
    }

    // 保存当前文件对象
    currentFile = file;

    // 计算文件大小（KB）
    const fileSizeInKB = (file.size / 1024).toFixed(1);
    fileInfo.textContent = `已选择: ${file.name} (${fileSizeInKB} KB)`;

    // 启用开始转写按钮
    uploadBtn.disabled = false;

    // 显示音频预览区域（之前可能隐藏）
    const previewDiv = document.getElementById('audioPreview');
    previewDiv.style.display = 'block';

    // 设置音频播放器源
    const audioPlayer = document.getElementById('audioPlayer');
    currentAudioUrl = URL.createObjectURL(file);
    audioPlayer.src = currentAudioUrl;

    // 请求后端生成波形数据
    generateWaveform(file);
}

/**
 * 重置文件信息（无文件选择时调用）
 */
function resetFileInfo() {
    // 清空当前文件对象
    currentFile = null;

    // 清空文件信息显示
    fileInfo.textContent = '';

    // 禁用开始转写按钮
    uploadBtn.disabled = true;

    // 可以隐藏预览区域（但这里保持显示也无妨，保留为空）
    // 可根据需要决定是否隐藏
}

/**
 * 从页面控件中收集所有高级设置参数
 * @returns {Object} 包含所有转写参数的配置对象
 */
function getConfig() {
    // 获取各个输入控件的当前值
    const precisionValue = document.getElementById('precision').value;
    const timestampChecked = document.getElementById('timestamp').checked;
    const useDmlChecked = document.getElementById('use_dml').checked;
    const useVulkanChecked = document.getElementById('use_vulkan').checked;
    const nCtxValue = parseInt(document.getElementById('n_ctx').value);
    const languageValue = document.getElementById('language').value || null;
    const contextValue = document.getElementById('context').value;
    const temperatureValue = parseFloat(document.getElementById('temperature').value);
    const seekStartValue = parseFloat(document.getElementById('seek_start').value);

    // duration 是可选参数，空字符串时传 null
    const durationInput = document.getElementById('duration').value;
    const durationValue = durationInput ? parseFloat(durationInput) : null;

    const chunkSizeValue = parseFloat(document.getElementById('chunk_size').value);
    const memoryNumValue = parseInt(document.getElementById('memory_num').value);
    const verboseChecked = document.getElementById('verbose').checked;
    const yesChecked = document.getElementById('yes').checked;

    // 组装配置对象
    const config = {
        precision: precisionValue,
        timestamp: timestampChecked,
        use_dml: useDmlChecked,
        use_vulkan: useVulkanChecked,
        n_ctx: nCtxValue,
        language: languageValue,
        context: contextValue,
        temperature: temperatureValue,
        seek_start: seekStartValue,
        duration: durationValue,
        chunk_size: chunkSizeValue,
        memory_num: memoryNumValue,
        verbose: verboseChecked,
        yes: yesChecked
    };

    return config;
}

/**
 * 点击“开始转写”按钮时触发的异步处理
 */
uploadBtn.addEventListener('click', async function () {
    // 如果没有选择文件，直接返回（按钮应该已被禁用，但保险）
    if (!currentFile) {
        return;
    }

    // 创建 FormData 对象，用于发送文件和数据
    const formData = new FormData();

    // 添加音频文件
    formData.append('audio', currentFile);

    // 添加配置参数（转换为 JSON 字符串）
    const config = getConfig();
    formData.append('config', JSON.stringify(config));

    // 隐藏之前的结果区域，显示进度提示
    resultSection.style.display = 'none';
    progress.style.display = 'block';

    // 禁用按钮，防止重复提交
    uploadBtn.disabled = true;

    try {
        // 发送 POST 请求到后端 /transcribe
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        // 如果响应状态码不是 2xx，则抛出错误
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`服务器错误: ${response.status} - ${errorText}`);
        }

        // 解析返回的 JSON 数据
        const data = await response.json();

        // 更新界面显示识别结果
        textResult.textContent = data.text || '(无文本)';
        srtPreview.textContent = data.srt || '(无字幕)';
        logDetails.textContent = data.log || '(无日志)';
        currentSrtContent = data.srt || '';

        // 显示结果区域
        resultSection.style.display = 'block';

        // 刷新临时文件列表（因为可能新产生了 .txt/.srt 等文件）
        fetchTempFiles();

    } catch (error) {
        // 发生错误时弹出提示
        alert('转写出错: ' + error.message);
    } finally {
        // 无论成功失败，都要隐藏进度条并重新启用按钮
        progress.style.display = 'none';
        uploadBtn.disabled = false;
    }
});

/**
 * 点击“下载 SRT 字幕”按钮时触发
 */
downloadSrt.addEventListener('click', function () {
    // 如果没有字幕内容，不执行
    if (!currentSrtContent) {
        return;
    }

    // 将文本内容转换为 Blob
    const blob = new Blob([currentSrtContent], { type: 'text/plain' });

    // 创建一个临时的 URL 指向该 Blob
    const url = URL.createObjectURL(blob);

    // 创建一个隐藏的 <a> 元素用于触发下载
    const downloadLink = document.createElement('a');
    downloadLink.href = url;
    downloadLink.download = 'subtitle.srt'; // 下载文件名

    // 模拟点击下载
    downloadLink.click();

    // 释放临时 URL 以释放内存
    URL.revokeObjectURL(url);
});

/**
 * 点击“复制文本”按钮时触发
 */
copyTextBtn.addEventListener('click', function () {
    // 获取显示的文本内容
    const text = textResult.textContent;

    // 如果文本存在且不是占位符，则尝试复制
    if (text && text !== '(无文本)') {
        navigator.clipboard.writeText(text)
            .then(() => {
                alert('文本已复制到剪贴板');
            })
            .catch(() => {
                alert('复制失败，请手动选择');
            });
    }
});

// ==================== 批量处理相关 ====================

/**
 * 当批量文件选择 input 变化时，更新已选文件列表
 */
batchFileInput.addEventListener('change', function () {
    // 将 FileList 转换为数组
    selectedFiles = Array.from(batchFileInput.files);
    // 更新界面显示
    updateBatchFileList();
});

/**
 * 点击“选择音频文件”按钮时，触发隐藏的 input 选择文件
 */
document.getElementById('batchSelectBtn').addEventListener('click', function () {
    batchFileInput.click();
});

/**
 * 更新批量文件列表的界面显示
 */
function updateBatchFileList() {
    // 如果没有选择任何文件，显示提示信息
    if (selectedFiles.length === 0) {
        batchFileList.innerHTML = '未选择文件';
        batchFileCount.textContent = '';
        return;
    }

    // 更新已选文件数量
    batchFileCount.textContent = `已选 ${selectedFiles.length} 个文件`;

    // 开始构建 HTML 列表
    let html = '<ul style="margin:0; padding-left:20px;">';

    // 遍历每个文件，生成列表项
    selectedFiles.forEach(function (file, index) {
        const fileSizeKB = (file.size / 1024).toFixed(1);
        html += `<li>${file.name} (${fileSizeKB} KB) `;
        html += `<button class="remove-file btn-primary" data-index="${index}">移除</button></li>`;
    });

    html += '</ul>';
    batchFileList.innerHTML = html;

    // 为每个移除按钮绑定事件
    const removeButtons = document.querySelectorAll('.remove-file');
    removeButtons.forEach(function (button) {
        button.addEventListener('click', function (event) {
            // 获取按钮上 data-index 属性，即文件在数组中的索引
            const index = event.target.getAttribute('data-index');
            // 从数组中移除该文件
            selectedFiles.splice(index, 1);
            // 清空 input 的值（因为无法直接修改文件列表）
            batchFileInput.value = '';
            // 重新渲染列表
            updateBatchFileList();
        });
    });
}

/**
 * 点击“添加到队列”按钮，提交批量任务
 */
batchUploadBtn.addEventListener('click', async function () {
    // 如果没有选择文件，提示并返回
    if (selectedFiles.length === 0) {
        alert('请先选择音频文件');
        return;
    }

    // 创建 FormData 用于发送文件和数据
    const formData = new FormData();

    // 添加所有选中的文件，字段名必须为 'files'（与后端对应）
    selectedFiles.forEach(function (file) {
        formData.append('files', file);
    });

    // 获取当前高级设置并添加为 JSON 字符串
    const config = getConfig();
    formData.append('config', JSON.stringify(config));

    // 禁用按钮，改变文本
    batchUploadBtn.disabled = true;
    batchUploadBtn.textContent = '提交中...';

    try {
        // 发送 POST 请求到 /tasks
        const response = await fetch('/tasks', {
            method: 'POST',
            body: formData
        });

        // 如果响应状态不是 OK，抛出错误
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`提交失败: ${response.status} - ${errorText}`);
        }

        // 解析返回的 JSON，包含 task_ids
        const data = await response.json();
        console.log('批量任务已提交，任务IDs:', data.task_ids);

        // 清空已选文件列表
        selectedFiles = [];
        batchFileInput.value = '';
        updateBatchFileList();

        // 立即刷新任务列表以显示新任务
        fetchTaskList();

    } catch (error) {
        alert('批量提交出错: ' + error.message);
    } finally {
        // 恢复按钮状态
        batchUploadBtn.disabled = false;
        batchUploadBtn.textContent = '添加到队列';
    }
});

// ==================== 任务列表管理 ====================

/**
 * 从后端获取所有任务列表并渲染
 */
async function fetchTaskList() {
    try {
        const response = await fetch('/tasks');
        const tasks = await response.json();
        renderTaskTable(tasks);
    } catch (error) {
        console.error('获取任务列表失败', error);
    }
}

/**
 * 渲染任务列表表格
 * @param {Array} tasks - 任务对象数组
 */
function renderTaskTable(tasks) {
    // 如果没有任务，显示提示行
    if (tasks.length === 0) {
        taskTableBody.innerHTML = '<tr><td colspan="3">暂无任务</td></tr>';
        return;
    }

    // 构建表格行 HTML
    let html = '';

    tasks.forEach(function (task) {
        let statusText = task.status;
        let actionButtons = '';

        // 根据任务状态决定显示内容和操作按钮
        if (task.status === 'completed') {
            // 已完成任务显示“文本”和“下载SRT”按钮
            actionButtons = `
                <button class="view-text-btn btn-primary" data-id="${task.id}">文本</button>
                <button class="download-srt-btn btn-primary" data-id="${task.id}">下载SRT</button>
            `;
        } else if (task.status === 'failed') {
            // 失败任务显示错误信息
            statusText = `失败: ${task.error || ''}`;
        }
        // 其他状态（pending, processing）只显示状态文本，无按钮

        html += `<tr>
            <td>${task.filename}</td>
            <td>${statusText}</td>
            <td>${actionButtons}</td>
        </tr>`;
    });

    taskTableBody.innerHTML = html;

    // 为所有“文本”按钮绑定事件
    const viewTextButtons = document.querySelectorAll('.view-text-btn');
    viewTextButtons.forEach(function (button) {
        button.addEventListener('click', async function (event) {
            const taskId = event.target.getAttribute('data-id');
            await showTaskText(taskId);
        });
    });

    // 为所有“下载SRT”按钮绑定事件
    const downloadSrtButtons = document.querySelectorAll('.download-srt-btn');
    downloadSrtButtons.forEach(function (button) {
        button.addEventListener('click', async function (event) {
            const taskId = event.target.getAttribute('data-id');
            await downloadTaskSrt(taskId);
        });
    });
}

/**
 * 显示指定任务的文本内容（通过模态框）
 * @param {string} taskId - 任务ID
 */
async function showTaskText(taskId) {
    try {
        const response = await fetch(`/tasks/${taskId}`);
        const task = await response.json();

        // 检查任务是否完成且包含文本结果
        if (task.status === 'completed' && task.result && task.result.text) {
            const modalTextContent = document.getElementById('modalTextContent');
            modalTextContent.textContent = task.result.text;

            const textModal = document.getElementById('textModal');
            textModal.style.display = 'flex';
        } else {
            alert('暂无文本');
        }
    } catch (error) {
        alert('获取文本失败');
    }
}

/**
 * 下载指定任务的 SRT 字幕文件
 * @param {string} taskId - 任务ID
 */
async function downloadTaskSrt(taskId) {
    try {
        const response = await fetch(`/tasks/${taskId}`);
        const task = await response.json();

        if (task.status === 'completed' && task.result && task.result.srt) {
            // 将字幕内容转换为 Blob
            const blob = new Blob([task.result.srt], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);

            // 创建下载链接
            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = `${task.filename}.srt`;

            downloadLink.click();
            URL.revokeObjectURL(url);
        } else {
            alert('无字幕文件');
        }
    } catch (error) {
        alert('下载失败');
    }
}

// 设置定时器，每隔 3 秒刷新任务列表
setInterval(fetchTaskList, 3000);

// ==================== 临时文件管理 ====================

/**
 * 从后端获取临时文件列表并渲染表格
 */
async function fetchTempFiles() {
    try {
        const response = await fetch('/tempfiles');
        const data = await response.json();

        // 更新限制显示和输入框
        maxUploadsSpan.textContent = data.max_uploads;
        maxUploadsInput.value = data.max_uploads;

        const files = data.files;

        // 根据文件列表渲染表格主体
        if (files.length === 0) {
            tempFilesBody.innerHTML = '<tr><td colspan="4">暂无临时文件</td></tr>';
        } else {
            let html = '';

            files.forEach(function (fileInfo) {
                // 计算大小（KB）
                const sizeKB = (fileInfo.size / 1024).toFixed(1) + ' KB';

                // 格式化修改时间
                const mtime = new Date(fileInfo.mtime).toLocaleString();

                html += `<tr>
                    <td>${fileInfo.name}</td>
                    <td>${sizeKB}</td>
                    <td>${mtime}</td>
                    <td>${fileInfo.type}</td>
                </tr>`;
            });

            tempFilesBody.innerHTML = html;
        }
    } catch (error) {
        console.error('获取临时文件列表失败', error);
    }
}

/**
 * 点击“应用”按钮时，设置新的文件数量限制
 */
setLimitBtn.addEventListener('click', async function () {
    const newLimit = parseInt(maxUploadsInput.value);

    // 验证输入是否有效
    if (isNaN(newLimit) || newLimit < 1) {
        alert('请输入有效的数字 (≥1)');
        return;
    }

    try {
        const response = await fetch('/tempfiles/set_limit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_uploads: newLimit })
        });

        const data = await response.json();

        if (data.error) {
            alert('设置失败: ' + data.error);
        } else {
            // 更新显示的限制值
            maxUploadsSpan.textContent = data.max_uploads;
            // 刷新文件列表
            fetchTempFiles();
        }
    } catch (error) {
        alert('设置失败: ' + error.message);
    }
});

/**
 * 点击“清理超出文件”按钮，手动触发清理
 */
cleanBtn.addEventListener('click', async function () {
    try {
        await fetch('/tempfiles/clean', { method: 'POST' });
        // 清理后刷新列表
        fetchTempFiles();
    } catch (error) {
        alert('清理失败: ' + error.message);
    }
});

/**
 * 点击“刷新列表”按钮，重新获取文件列表
 */
refreshTempBtn.addEventListener('click', function () {
    fetchTempFiles();
});

// ==================== 高级设置面板 ====================

/**
 * 点击齿轮图标时，切换设置面板的显示/隐藏
 */
settingsToggle.addEventListener('click', function (event) {
    event.stopPropagation(); // 阻止事件冒泡，避免立即触发 document 的点击关闭
    settingsPanel.classList.toggle('hidden');
});

/**
 * 点击面板内的“关闭”按钮，隐藏面板
 */
closeSettings.addEventListener('click', function () {
    settingsPanel.classList.add('hidden');
});

/**
 * 点击页面上除面板和齿轮外的任何地方，关闭面板
 */
document.addEventListener('click', function (event) {
    // 如果点击的目标不在面板内，也不是齿轮图标，则隐藏面板
    if (!settingsPanel.contains(event.target) && event.target !== settingsToggle) {
        settingsPanel.classList.add('hidden');
    }
});

// ==================== 模态框（显示文本） ====================

// 获取模态框元素（注意：这些元素可能还未加载？但脚本在 body 末尾，没问题）
const closeModalBtn = document.getElementById('closeModal');
const textModal = document.getElementById('textModal');
const copyModalTextBtn = document.getElementById('copyModalText');

/**
 * 点击关闭按钮隐藏模态框
 */
closeModalBtn.addEventListener('click', function () {
    textModal.style.display = 'none';
});

/**
 * 点击模态框背景（遮罩）也隐藏
 */
textModal.addEventListener('click', function (event) {
    // 如果点击的是模态框本身（即背景），而不是内容区域，则隐藏
    if (event.target === textModal) {
        textModal.style.display = 'none';
    }
});

/**
 * 点击复制按钮，将模态框内的文本复制到剪贴板
 */
copyModalTextBtn.addEventListener('click', function () {
    const modalText = document.getElementById('modalTextContent').textContent;
    navigator.clipboard.writeText(modalText)
        .then(() => {
            alert('文本已复制到剪贴板');
        })
        .catch(() => {
            alert('复制失败，请手动选择');
        });
});

// ==================== 波形图生成 ====================

/**
 * 请求后端生成音频文件的波形数据
 * @param {File} file - 音频文件对象
 */
async function generateWaveform(file) {
    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/waveform', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('波形生成失败');
        }

        const data = await response.json();
        drawWaveform(data.waveform, data.duration);
    } catch (error) {
        console.error('波形生成错误:', error);
    }
}

/**
 * 在 canvas 上绘制波形图（柱状图样式）
 * @param {Array<number>} waveform - 归一化的波形振幅数组（值范围约 -1..1）
 * @param {number} duration - 音频时长（秒）
 */
function drawWaveform(waveform, duration) {
    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');

    const width = canvas.width;
    const height = canvas.height;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    // 设置波形颜色
    ctx.fillStyle = '#0078d4';

    // 计算每个柱子的宽度
    const step = width / waveform.length;

    // 画布垂直中心线
    const midY = height / 2;

    // 计算波形中的最大绝对值，用于归一化（避免除以0）
    const absoluteValues = waveform.map(Math.abs);
    const maxAmplitude = Math.max(...absoluteValues, 0.001);

    // 遍历每个波形点
    for (let i = 0; i < waveform.length; i++) {
        // 归一化振幅（-1..1 映射到 0..1 的绝对值）
        const normalizedAmp = waveform[i] / maxAmplitude;

        // 柱子高度，占画布高度的 80%（留白）
        const barHeight = Math.abs(normalizedAmp) * (height * 0.8);

        // 柱子的 x 坐标
        const x = i * step;

        // 从中心向上下扩展绘制矩形
        // 柱子的左上角 y 坐标 = 中心 - 高度/2
        const barY = midY - barHeight / 2;

        // 柱子宽度，至少为 1px，且相邻柱子之间留 1px 间隙
        const barWidth = Math.max(1, step - 1);

        ctx.fillRect(x, barY, barWidth, barHeight);
    }

    // 绘制时间刻度文字
    ctx.fillStyle = '#333';
    ctx.font = '10px Arial';

    // 起始时间 0s
    ctx.fillText('0s', 5, 20);

    // 结束时间，保留一位小数
    const durationText = duration.toFixed(1) + 's';
    ctx.fillText(durationText, width - 40, 20);
}

// ==================== 临时文件整理 ====================

// 获取整理按钮
const organizeBtn = document.getElementById('organizeBtn');

/**
 * 点击“整理”按钮时触发，请求后端整理临时文件
 */
organizeBtn.addEventListener('click', async function () {
    // 弹出确认对话框
    const userConfirmed = confirm(
        '确定整理临时文件吗？' +
        '会将每个音频对应的 .txt/.srt/.json 文件移动到同名文件夹内。'
    );

    if (!userConfirmed) {
        return;
    }

    // 禁用按钮，改变文本
    organizeBtn.disabled = true;
    organizeBtn.textContent = '整理中...';

    try {
        const response = await fetch('/tempfiles/organize', { method: 'POST' });
        const data = await response.json();

        if (data.errors && data.errors.length > 0) {
            // 有错误时，拼接错误信息
            const errorMessage = '整理完成，但有错误：\n' + data.errors.join('\n');
            alert(errorMessage);
        } else {
            alert(`整理完成，移动了 ${data.organized_count} 个文件。`);
        }

        // 刷新临时文件列表
        fetchTempFiles();

    } catch (error) {
        alert('整理失败：' + error.message);
    } finally {
        // 恢复按钮状态
        organizeBtn.disabled = false;
        organizeBtn.textContent = '整理';
    }
});

// ==================== 主题切换 ====================

const themeToggle = document.getElementById('themeToggle');

/**
 * 设置主题（日间/夜间）
 * @param {boolean} isDark - true 为夜间模式，false 为日间模式
 */
function setTheme(isDark) {
    if (isDark) {
        // 添加 dark-theme 类到 body
        document.body.classList.add('dark-theme');
        themeToggle.textContent = '🌙'; // 月亮图标
    } else {
        // 移除 dark-theme 类
        document.body.classList.remove('dark-theme');
        themeToggle.textContent = '☀️'; // 太阳图标
    }

    // 将主题偏好保存到 localStorage
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// 初始化主题：从 localStorage 读取用户偏好
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') {
    setTheme(true);
} else {
    setTheme(false); // 默认为日间模式
}

/**
 * 点击主题切换按钮时，切换主题
 */
themeToggle.addEventListener('click', function () {
    const isDark = document.body.classList.contains('dark-theme');
    setTheme(!isDark);
});