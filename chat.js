const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

function addMessage(text, sender, meta) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msgDiv.appendChild(bubble);

  if (meta) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";
    metaDiv.textContent = meta;
    msgDiv.appendChild(metaDiv);
  }

  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  chatInput.value = "";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    if (data.error) {
      addMessage("Error: " + data.error, "bot");
      return;
    }

    addMessage(data.response, "bot", `intent: ${data.intent} · confidence: ${data.confidence}`);
  } catch (err) {
    addMessage("Connection error — is the Flask server running?", "bot");
  }
});
