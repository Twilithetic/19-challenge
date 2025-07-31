// 获取DOM元素
const testBtn = document.getElementById('testBtn');
const stopBtn = document.getElementById('stopBtn');
const statusOverlay = document.getElementById('statusOverlay');
const distanceValue = document.getElementById('distanceValue');
const areaValue = document.getElementById('areaValue');
const startBtn = document.getElementById('startBtn');
const procBtn = document.getElementById('procBtn');

// 开始测试
testBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/start_process', { method: 'POST' });
        if (response.ok) {
            testBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');
            statusOverlay.classList.remove('hidden');
        }
    } catch (error) {
        console.error('启动测试失败:', error);
    }
});

// 停止测试
stopBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/stop_process', { method: 'POST' });
        if (response.ok) {
            stopBtn.classList.add('hidden');
            testBtn.classList.remove('hidden');
            statusOverlay.classList.add('hidden');
            distanceValue.textContent = '-- cm';
            areaValue.textContent = '-- px';
        }
    } catch (error) {
        console.error('停止测试失败:', error);
    }
});

// 点击“一键启动”按钮触发
startBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/start_process', { method: 'POST' });
        if (response.ok) {
            // 1. 隐藏启动按钮，显示运行中按钮
            startBtn.classList.add('hidden');
            procBtn.classList.remove('hidden');

            // 2. 切换按钮颜色为绿色（移除红色类，添加绿色类）
            procBtn.classList.remove('bg-danger', 'hover:bg-danger/90');
            procBtn.classList.add('bg-green-500', 'hover:bg-green-600'); // 绿色及hover效果

            // 3. 倒计时逻辑（从5秒开始）
            let count = 4;
            // 更新按钮文本（带倒计时）
            const updateText = () => {
                procBtn.innerHTML = `<i class="fa fa-stop-circle mr-2"></i>运行中....${count}`;
            };

            // 初始显示5秒
            updateText();

            // 每秒倒计时一次
            const timer = setInterval(async () => {
                count--;
                updateText();

                // 倒计时结束（到1秒后停止）
                if (count <= 0) {
                    try {
                        const response = await fetch('/stop_process', { method: 'POST' });
                        if (response.ok) {
                            clearInterval(timer);
                            startBtn.classList.remove('hidden');
                            procBtn.classList.add('hidden');
                            distanceValue.textContent = '-- cm';
                            areaValue.textContent = '-- px';
                        }
                    } catch (error) {
                        console.error('停止测试失败:', error);
                    }


                    // 可选：倒计时结束后保持最终状态或执行其他操作
                    // procBtn.innerHTML = `<i class="fa fa-stop-circle mr-2"></i>运行中`;
                }
            }, 1000); // 1000毫秒 = 1秒
        }
    } catch (error) {
        console.error('启动测试失败:', error);
    }
    
});

// 定期获取处理结果更新UI
setInterval(async () => {
    if (stopBtn.classList.contains('hidden') && procBtn.classList.contains('hidden')) return;

    try {
        const response = await fetch('/get_results');
        if (response.ok) {
            const data = await response.json();
            if (data.distance !== null) {
                distanceValue.textContent = `${data.distance} cm`;
            }
        }
    } catch (error) {
        console.error('获取结果失败:', error);
    }
}, 100);