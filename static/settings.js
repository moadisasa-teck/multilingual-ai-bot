document.addEventListener("DOMContentLoaded", () => {
    const defaultSector = document.getElementById("default-sector");
    const showMetadata = document.getElementById("show-metadata");
    const saveSettings = document.getElementById("save-settings");
    const clearHistory = document.getElementById("clear-history");
    const status = document.getElementById("settings-status");

    const loadSettings = () => {
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

    const renderStatus = (message, isError = false) => {
        status.textContent = message;
        status.classList.toggle("error", isError);
    };

    const settings = loadSettings();
    defaultSector.value = settings.defaultSector ?? "passport";
    showMetadata.value = String(settings.showMetadata ?? true);

    saveSettings.addEventListener("click", () => {
        const updated = {
            defaultSector: defaultSector.value,
            showMetadata: showMetadata.value === "true",
        };
        localStorage.setItem("chatbot_settings", JSON.stringify(updated));
        renderStatus("Settings saved.");
    });

    clearHistory.addEventListener("click", async () => {
        try {
            const response = await fetch("/history", { method: "DELETE" });
            if (!response.ok) {
                throw new Error("Failed to clear server history.");
            }
            renderStatus("Server chat history cleared.");
        } catch (error) {
            renderStatus(error.message, true);
        }
    });
});
