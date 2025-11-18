const audioInput = document.getElementById("audioInput");
const sendBtn = document.getElementById("sendBtn");
const statusText = document.getElementById("statusText");
const transcriptEl = document.getElementById("transcript");
const translationEl = document.getElementById("translation");

sendBtn.addEventListener("click", () => {
    if (!audioInput.files.length) {
        alert("Selecione um arquivo de áudio primeiro!");
        return;
    }

    const file = audioInput.files[0];
    const formData = new FormData();
    formData.append("audio", file);

    statusText.textContent = "📥 Enviando áudio...";

    fetch("/api/transcribe/", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        transcriptEl.textContent = data.transcript;
        translationEl.textContent = data.translation;
        statusText.textContent = "✅ Áudio processado com sucesso!";
    })
    .catch(err => {
        console.error(err);
        statusText.textContent = "❌ Erro ao processar o áudio.";
    });
});
