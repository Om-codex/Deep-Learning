function typeLetter() {
    titleElement.textContent = text.slice(0, index);
    index++;

    if (index <= text.length) {
        setTimeout(typeLetter, 65);
    } else {
        document.getElementById("subtitle").style.opacity = 1;
    }
}

window.onload = () => {
    setTimeout(typeLetter, 400);
};
document.querySelector(".math-toggle").onclick = () => {
    const section = document.getElementById("math-section");
    const label = document.querySelector(".math-toggle-text");

    if (section.style.display === "block") {
        section.style.display = "none";
        label.textContent = "▼ Show Mathematical Derivation";
    } else {
        section.style.display = "block";
        label.textContent = "▲ Hide Mathematical Derivation";
        section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
};
document.querySelector(".math-toggle1").onclick = () => {
    const section = document.getElementById("math-section1");
    const label = document.querySelector(".math-toggle-text1");

    if (section.style.display === "block") {
        section.style.display = "none";
        label.textContent = "▼ Show Derivation Of Gradients";
    } else {
        section.style.display = "block";
        label.textContent = "▲ Hide Derivation Of Gradients";
        section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
};
window.MathJax = {
tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
svg: { fontCache: 'global' }
};
let loss = 1.0;
const lossDisplay = document.getElementById("loss-number");
const lossFill = document.querySelector(".loss-fill");

function trainStep() {
  if (loss > 0.05) {
    loss -= 0.03;
    lossDisplay.textContent = loss.toFixed(3);
    lossFill.style.width = (loss * 100) + "%";

    if (loss > 0.6) lossFill.style.background = "#ff4b4b";
    else if (loss > 0.3) lossFill.style.background = "#ffb84b";
    else lossFill.style.background = "#00ff9d";
  }
}

let trainingActive = true;
let trainingLoop = setInterval(trainStep, 1600);

document.getElementById("train-toggle").onclick = () => {
  if (trainingActive) {
    clearInterval(trainingLoop);
    document.getElementById("train-toggle").textContent = "▶ Resume Training";
  } else {
    trainingLoop = setInterval(trainStep, 1600);
    document.getElementById("train-toggle").textContent = "⏸ Pause Training";
  }
  trainingActive = !trainingActive;
};
