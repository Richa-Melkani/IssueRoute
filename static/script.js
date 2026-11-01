
const complaintForm = document.getElementById("complaintForm");
const complaintInput = document.getElementById("complaint");
const result = document.getElementById("result");
const submitBtn = document.getElementById("submitBtn");


complaintForm.addEventListener("submit", async function (event) {

    event.preventDefault();

    const complaint = complaintInput.value.trim();

    if (!complaint) {
        showError("Please enter your complaint.");
        return;
    }

    // Show loading state
    result.hidden = false;
    result.replaceChildren();

    const loading = document.createElement("p");
    loading.textContent = "Submitting your complaint...";
    result.appendChild(loading);

    submitBtn.disabled = true;
    submitBtn.textContent = "Please Wait...";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                complaint: complaint
            })

        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "Something went wrong.");
            return;
        }

        displayResult(data);

    }

    catch (error) {

        showError("Unable to connect to the server. Please try again.");

    }

    finally {

        submitBtn.disabled = false;
        submitBtn.textContent = "Submit Complaint";

    }

});


// Display prediction result
function displayResult(data) {

    result.hidden = false;
    result.replaceChildren();

    const heading = document.createElement("h3");
    heading.textContent = "Complaint Result";

    result.appendChild(heading);

    addResultRow("Complaint Type", data.complaint_type || "Not available");

    addResultRow(
        "Department(s)",
        Array.isArray(data.departments) && data.departments.length
            ? data.departments.join(", ")
            : "Not assigned"
    );

    addResultRow("Status", data.status || "Not available");

    if (data.margin !== undefined) {
        addResultRow("Confidence Margin", data.margin);
    }

    if (data.message) {
        const message = document.createElement("p");
        message.className = "result-message";
        message.textContent = data.message;
        result.appendChild(message);
    }

    const status = document.createElement("span");
    status.className = "result-status";
    status.textContent = data.status || "Result generated";

    result.appendChild(status);

}


// Add a result row safely
function addResultRow(label, value) {

    const row = document.createElement("p");

    const strong = document.createElement("strong");
    strong.textContent = label + ": ";

    const text = document.createElement("span");
    text.textContent = value;

    row.appendChild(strong);
    row.appendChild(text);

    result.appendChild(row);

}


// Display error
function showError(message) {

    result.hidden = false;
    result.replaceChildren();

    const error = document.createElement("p");
    error.className = "error-message";
    error.textContent = message;

    result.appendChild(error);

}
