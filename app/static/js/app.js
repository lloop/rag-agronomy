document.addEventListener("DOMContentLoaded", () => {
    const ragForm = document.getElementById("rag-form");
    const queryInput = document.getElementById("query-input");
    const chatBox = document.getElementById("chat-box");
    const sourcesList = document.getElementById("sources-list");
    const submitBtn = document.getElementById("submit-btn");

    ragForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        appendMessage(query, "user-message");
        queryInput.value = "";
        submitBtn.disabled = true;

        const loadingDiv = appendMessage("Searching context and generating answer...", "system-message");

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, top_k: 4 })
            });

            const data = await response.json();
            loadingDiv.remove();

            if (response.ok) {
                appendMessage(data.answer, "assistant-message", data.confidence_score, data.is_sufficient_context);
                renderCitations(data.citations);
            } else {
                appendMessage(`Error: ${data.error || "Failed to process request"}`, "error-message");
            }
        } catch (error) {
            loadingDiv.remove();
            appendMessage("Error communicating with server.", "error-message");
        } finally {
            submitBtn.disabled = false;
        }
    });

    function appendMessage(text, className, confidence = null, isSufficient = true) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${className}`;

        let contentHtml = `<div>${escapeHtml(text)}</div>`;
        
        if (confidence !== null) {
            const scorePercent = Math.round(confidence * 100);
            const badgeClass = isSufficient ? "badge-sufficient" : "badge-insufficient";
            const badgeText = isSufficient ? "Sufficient Context" : "Insufficient Context";
            
            contentHtml += `
                <div class="message-meta">
                    <span class="badge ${badgeClass}">${badgeText}</span>
                    <span class="confidence">Confidence: ${scorePercent}%</span>
                </div>
            `;
        }

        msgDiv.innerHTML = contentHtml;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    }

    function renderCitations(citations) {
        sourcesList.innerHTML = "";

        if (!citations || citations.length === 0) {
            sourcesList.innerHTML = '<p class="placeholder-text">No citations provided for this answer.</p>';
            return;
        }

        citations.forEach((cite, idx) => {
            const card = document.createElement("div");
            card.className = "source-card";

            card.innerHTML = `
                <div class="source-header">
                    <strong>[${idx + 1}] ${escapeHtml(cite.source_doc)}</strong>
                    <span>Page ${cite.page_number}</span>
                </div>
                <div class="source-folder">Folder: ${escapeHtml(cite.enclosing_folder)}</div>
                <div class="source-text">"${escapeHtml(cite.snippet)}"</div>
            `;
            sourcesList.appendChild(card);
        });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});