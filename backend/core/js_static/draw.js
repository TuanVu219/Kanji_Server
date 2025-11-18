const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
let drawing = false;

const socket = new WebSocket("ws://" + window.location.host + "/ws/recognize/");

// Bắt đầu vẽ
canvas.addEventListener("mousedown", e => {
    drawing = true;
    ctx.beginPath();
});

canvas.addEventListener("mouseup", e => drawing = false);

canvas.addEventListener("mousemove", e => {
    if (!drawing) return;

    const x = e.offsetX;
    const y = e.offsetY;
    ctx.lineTo(x, y);
    ctx.stroke();

    // Gửi toạ độ đến server
    socket.send(JSON.stringify({ x, y }));
});

// Nhận dữ liệu từ WebSocket để hiển thị nét vẽ từ người khác
socket.onmessage = (e) => {
    const data = JSON.parse(e.data);
    ctx.lineTo(data.x, data.y);
    ctx.stroke();
};
