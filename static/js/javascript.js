// WebSocket部分
const wsMessages = document.getElementById('wsMessages');
const wsInput = document.getElementById('wsInput');
const socket = new WebSocket(`ws://${window.location.host}/ws`);

socket.onopen = () => {
    appendMessage('wsMessages', '已连接到WebSocket服务器', 'received');
};

socket.onmessage = (event) => {
    appendMessage('wsMessages', `服务器: ${event.data}`, 'received');
};

socket.onclose = () => {
    appendMessage('wsMessages', 'WebSocket连接已关闭', 'received');
};

function sendWebSocketMessage() {
    const message = wsInput.value.trim();
    if (message) {
        socket.send(message);
        appendMessage('wsMessages', `你: ${message}`, 'sent');
        wsInput.value = '';
    }
}

// Fetch部分
const fetchMessages = document.getElementById('fetchMessages');

async function sendFetchRequest() {
    try {
        const response = await fetch('/time');
        const data = await response.json();
        appendMessage('fetchMessages', `当前时间: ${data.time}`, 'received');
    } catch (error) {
        appendMessage('fetchMessages', `错误: ${error.message}`, 'received');
    }
}

function appendMessage(containerId, text, type) {
    const container = document.getElementById(containerId);
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}