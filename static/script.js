document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const languageSelect = document.getElementById('language-select');
    const sectorSelect = document.getElementById('sector-select');

    const appendMessage = (content, role, metadata = {}) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role);

        let metadataHtml = '';
        if (role === 'system' && metadata.confidence) {
            metadataHtml = `<div class="metadata">
                Language: ${metadata.language} | Sector: ${metadata.sector} | Confidence: ${(metadata.confidence * 100).toFixed(1)}%
            </div>`;
        }

        // Render Markdown if it's a system message, otherwise plain text
        const renderedContent = role === 'system' ? marked.parse(content) : content;

        messageDiv.innerHTML = `
            <div class="message-content">${renderedContent}</div>
            ${metadataHtml}
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const handleSend = async () => {
        const text = userInput.value.trim();
        if (!text) return;

        // User message
        appendMessage(text, 'user');
        userInput.value = '';

        // UI state
        const originalButtonContent = sendButton.innerHTML;
        sendButton.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i>';
        sendButton.disabled = true;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: text,
                    language: languageSelect.value || null,
                    sector: sectorSelect.value || null
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }

            const data = await response.json();

            // Bot message
            appendMessage(data.answer, 'system', {
                language: data.language,
                sector: data.sector,
                confidence: data.confidence
            });

        } catch (error) {
            console.error('Error:', error);
            appendMessage('Sorry, there was an error processing your request. Please try again.', 'system');
        } finally {
            sendButton.innerHTML = originalButtonContent;
            sendButton.disabled = false;
        }
    };

    sendButton.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
});
