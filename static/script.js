const complaintForm = document.getElementById("complaintForm");
const complaintInput = document.getElementById("complaint");
const submitBtn = document.getElementById("submitBtn");
const resultDiv = document.getElementById("result");

const testComplaints = document.getElementById("testComplaints");
const useTestBtn = document.getElementById("useTestBtn");
const charCount = document.getElementById("charCount");

// Update character count
complaintInput.addEventListener("input", () => {
    charCount.textContent = `${complaintInput.value.length} / 1500`;
});

// Fill complaint box from a sample complaint
document.querySelectorAll(".sample-chip").forEach((button) => {
    button.addEventListener("click", () => {
        complaintInput.value = button.dataset.complaint;
        charCount.textContent = `${complaintInput.value.length} / 1500`;

        complaintInput.focus();
        complaintInput.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    });
});

// Use manually entered test complaint
useTestBtn.addEventListener("click", () => {
    const sample = testComplaints.value.trim();

    if (!sample) {
        testComplaints.focus();
        return;
    }

    complaintInput.value = sample;
    charCount.textContent = `${complaintInput.value.length} / 1500`;

    complaintInput.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

    complaintInput.focus();
});

// Submit complaint to Flask backend
complaintForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const complaint = complaintInput.value.trim();

    if (!complaint) {
        complaintInput.focus();
        return;
    }

    submitBtn.disabled = true;
    submitBtn.innerHTML = "<span>Analysing complaint...</span>";
    resultDiv.hidden = false;
    resultDiv.innerHTML = `<p class="result-empty">Please wait while IssueRoute analyses your complaint...</p>`;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ complaint: complaint })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong.");
        }

        displayResult(data);

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="status-box status-error">
                <strong>Unable to analyse complaint</strong>
                <p>${escapeHTML(error.message)}</p>
            </div>
        `;
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<span>Analyse complaint</span><span class="button-arrow">→</span>`;
    }
});

function displayResult(data) {
    const complaintType = data.complaint_type || "Not available";
    const departments = Array.isArray(data.departments)
        ? data.departments
        : [];

    const status = data.status || "Not available";

    let statusClass = "status-review";

    if (status === "Auto Routed") {
        statusClass = "status-auto";
    } else if (status === "Multiple Departments Detected") {
        statusClass = "status-multiple";
    }

    let departmentHTML = "";

    if (departments.length > 0) {
        departmentHTML = departments.map((department) => `
            <span class="department-tag">${escapeHTML(department)}</span>
        `).join("");
    } else {
        departmentHTML = `<span class="result-empty">No department assigned</span>`;
    }

    let marginHTML = "";

    if (data.margin !== undefined && data.margin !== null) {
        marginHTML = `
            <div class="result-item">
                <span class="result-label">Confidence margin</span>
                <span class="result-value">${escapeHTML(data.margin)}</span>
            </div>
        `;
    }

    let messageHTML = "";

    if (data.message) {
        messageHTML = `
            <p>${escapeHTML(data.message)}</p>
        `;
    } else if (status === "Auto Routed") {
        messageHTML = `
            <p>The complaint has been assigned to the predicted department.</p>
        `;
    } else if (status === "Multiple Departments Detected") {
        messageHTML = `
            <p>More than one relevant department was identified.</p>
        `;
    } else {
        messageHTML = `
            <p>The complaint requires manual verification.</p>
        `;
    }

    resultDiv.innerHTML = `
        <h3 class="result-title">Analysis result</h3>

        <div class="result-grid">
            <div class="result-item">
                <span class="result-label">Complaint type</span>
                <span class="result-value">${escapeHTML(complaintType)}</span>
            </div>

            <div class="result-item">
                <span class="result-label">Routing status</span>
                <span class="result-value">${escapeHTML(status)}</span>
            </div>

            ${marginHTML}
        </div>

        <div class="department-block">
            <span class="result-label">Identified department(s)</span>
            <div class="department-list">
                ${departmentHTML}
            </div>
        </div>

        <div class="status-box ${statusClass}">
            <strong>${escapeHTML(status)}</strong>
            ${messageHTML}
        </div>

        <div class="complaint-preview">
            <strong>Submitted complaint</strong>
            ${escapeHTML(data.complaint || "")}
        </div>
    `;

    resultDiv.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}

// Safely display text returned by the backend
function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, (character) => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        };

        return entities[character];
    });
}