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

                const sourceDoc = cite.source_doc || "Unknown Document";
                const pageNum = cite.page_number !== undefined && cite.page_number !== null ? cite.page_number : "N/A";
                const folder = cite.enclosing_folder || "Unknown Folder";
                const snippetText = cite.snippet ? `<div class="source-text">"${escapeHtml(cite.snippet)}"</div>` : "";

                card.innerHTML = `
                    <div class="source-header">
                        <strong>[${idx + 1}] ${escapeHtml(sourceDoc)}</strong>
                        <span>Page ${pageNum}</span>
                    </div>
                    <div class="source-folder">Folder: ${escapeHtml(folder)}</div>
                    ${snippetText}
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