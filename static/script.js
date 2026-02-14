document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const sectorSelect = document.getElementById("sector-select");
    const clearHistoryButton = document.getElementById("clear-history-button");
    const historyList = document.getElementById("history-list");
    const newChatButton = document.getElementById("new-chat-button");
    const sidebarToggleButton = document.getElementById("sidebar-toggle-button");
    const sidebar = document.querySelector(".sidebar");

    let localHistory = [];
    let currentConversationId = null;
    let conversations = [];

    const getSettings = () => {
        const raw = localStorage.getItem("chatbot_settings");
        if (!raw) {
            return { defaultSector: "passport", showMetadata: true };
        }
        try {
            return JSON.parse(raw);
        } catch (_error) {
            return { defaultSector: "passport", showMetadata: true };
        }
    };

    const settings = getSettings();
    sectorSelect.value = settings.defaultSector || "passport";

    const renderHistoryList = () => {
        historyList.innerHTML = "";
        if (!conversations.length) {
            historyList.innerHTML = `<div class="history-empty">No previous chats</div>`;
            return;
        }

        for (const conv of conversations) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `history-item ${conv.id === currentConversationId ? "active" : ""}`;
            button.textContent = conv.title || "New Chat";
            button.title = conv.title || "New Chat";
            button.addEventListener("click", () => loadHistory(conv.id));
            historyList.appendChild(button);
        }
    };

    const appendMessage = (content, role, metadata = {}) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", role);

        let metadataHtml = "";
        if (role === "system" && settings.showMetadata && metadata.confidence !== undefined) {
            metadataHtml = `<div class="metadata">
                Language: ${metadata.language} | Sector: ${metadata.sector} | Confidence: ${(metadata.confidence * 100).toFixed(1)}%
            </div>`;
        }

        const renderedContent = role === "system" ? marked.parse(content) : content;

        messageDiv.innerHTML = `
            <div class="message-content">${renderedContent}</div>
            ${metadataHtml}
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const renderMessages = (messages) => {
        chatMessages.innerHTML = "";
        for (const msg of messages) {
            if (msg.role === "user") {
                appendMessage(msg.content, "user");
            } else if (msg.role === "assistant") {
                appendMessage(msg.content, "system");
            }
        }
    };

    const appendTypingBubble = () => {
        const typingDiv = document.createElement("div");
        typingDiv.classList.add("message", "system", "typing-message");
        typingDiv.innerHTML = `
            <div class="message-content typing-content">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingDiv;
    };

    const loadHistory = async (conversationId = null) => {
        try {
            const url = conversationId ? `/history?conversation_id=${encodeURIComponent(conversationId)}` : "/history";
            const response = await fetch(url);
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            currentConversationId = data.active_id || conversationId;
            conversations = data.conversations || [];
            localHistory = (data.messages || []).slice(-10);
            renderMessages(data.messages || []);
            renderHistoryList();
        } catch (error) {
            console.error("Failed to load history:", error);
        }
    };

    const createNewChat = async () => {
        try {
            const response = await fetch("/history/new", { method: "POST" });
            if (!response.ok) {
                throw new Error("Failed to create a new chat.");
            }
            const data = await response.json();
            currentConversationId = data.active_id;
            localHistory = [];
            chatMessages.innerHTML = "";
            await loadHistory(currentConversationId);
        } catch (error) {
            console.error(error);
        }
    };

    const clearHistory = async () => {
        try {
            await fetch("/history", { method: "DELETE" });
        } catch (error) {
            console.error("Failed to clear history:", error);
        }
        currentConversationId = null;
        localHistory = [];
        conversations = [];
        chatMessages.innerHTML = "";
        renderHistoryList();
    };

    const handleSend = async () => {
        const text = userInput.value.trim();
        if (!text) {
            return;
        }

        appendMessage(text, "user");
        userInput.value = "";

        const originalButtonContent = sendButton.innerHTML;
        sendButton.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i>';
        sendButton.disabled = true;
        const typingBubble = appendTypingBubble();

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    query: text,
                    sector: sectorSelect.value || null,
                    history: localHistory,
                    conversation_id: currentConversationId,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Network response was not ok");
            }

            const data = await response.json();
            typingBubble.remove();

            appendMessage(data.answer, "system", {
                language: data.language,
                sector: data.sector,
                confidence: data.confidence,
            });

            currentConversationId = data.conversation_id || currentConversationId;
            localHistory.push({ role: "user", content: text });
            localHistory.push({ role: "assistant", content: data.answer });
            localHistory = localHistory.slice(-10);
            await loadHistory(currentConversationId);
        } catch (error) {
            typingBubble.remove();
            console.error("Error:", error);
            appendMessage("Sorry, there was an error processing your request. Please try again.", "system");
        } finally {
            sendButton.innerHTML = originalButtonContent;
            sendButton.disabled = false;
        }
    };

    sendButton.addEventListener("click", handleSend);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            handleSend();
        }
    });
    clearHistoryButton.addEventListener("click", clearHistory);
    newChatButton.addEventListener("click", createNewChat);
    sidebarToggleButton.addEventListener("click", () => {
        document.body.classList.toggle("sidebar-open");
    });
    document.addEventListener("click", (event) => {
        if (window.innerWidth > 768) {
            return;
        }
        if (!document.body.classList.contains("sidebar-open")) {
            return;
        }
        const clickedInsideSidebar = sidebar.contains(event.target);
        const clickedToggle = sidebarToggleButton.contains(event.target);
        if (!clickedInsideSidebar && !clickedToggle) {
            document.body.classList.remove("sidebar-open");
        }
    });

    loadHistory();
});
