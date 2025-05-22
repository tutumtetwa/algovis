let arr = [50, 30, 60, 10, 80, 20];
let container = document.getElementById("arrayContainer");

function renderBars() {
    container.innerHTML = '';
    arr.forEach(val => {
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.height = `${val * 2}px`;
        container.appendChild(bar);
    });
}

async function startSort() {
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
                renderBars();
                await new Promise(r => setTimeout(r, 300));
            }
        }
    }
}

renderBars();
