// 获取DOM元素
const testBtn = document.getElementById('testBtn');
const stopBtn = document.getElementById('stopBtn');
const statusOverlay = document.getElementById('statusOverlay');
const distanceValue = document.getElementById('distanceValue');
const areaValue = document.getElementById('areaValue');

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

// 定期获取处理结果更新UI
setInterval(async () => {
    if (stopBtn.classList.contains('hidden')) return;

    try {
        const response = await fetch('/get_results');
        if (response.ok) {
            const data = await response.json();
            if (data.distance !== null) {
                distanceValue.textContent = `${data.distance.toFixed(1)} cm`;
            }
            if (data.area !== null) {
                areaValue.textContent = `${data.area} px`;
            }
        }
    } catch (error) {
        console.error('获取结果失败:', error);
    }
}, 100);