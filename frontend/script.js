async function askQuestion() {
    const question = document.getElementById("question").value;
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");

    answerEl.innerText = "Thinking...";
    sourcesEl.innerHTML = "";

    const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question })
    });

    const data = await response.json();

    answerEl.innerText = data.answer;

    if (data.sources) {
        data.sources.forEach(src => {
            const li = document.createElement("li");
            li.innerText = src;
            sourcesEl.appendChild(li);
        });
    }
}
