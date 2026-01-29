const form = document.getElementById("review-form");
const textarea = document.getElementById("review-text");
const batchForm = document.getElementById("batch-form");
const csvInput = document.getElementById("csv-file");
const outputCard = document.getElementById("output-card");
const result = document.getElementById("result");
const emptyState = document.getElementById("empty-state");
const errorCard = document.getElementById("error-card");
const statusPill = document.getElementById("status-pill");
const analyzeButton = form.querySelector("button.primary");
const clearButton = document.getElementById("clear-btn");
const clearCsvButton = document.getElementById("clear-csv-btn");
const resultLabel = document.getElementById("result-label");
const confidenceValue = document.getElementById("confidence-value");
const meterFill = document.getElementById("meter-fill");
const resultNote = document.getElementById("result-note");
const toast = document.getElementById("toast");
const modeButtons = document.querySelectorAll(".mode-btn");
const singlePanel = document.getElementById("single-mode");
const batchPanel = document.getElementById("batch-mode");

const clampPercent = (value) => Math.max(0, Math.min(100, Math.round(value)));

const setStatus = (label, state) => {
  statusPill.textContent = label;
  statusPill.dataset.state = state;
};

const showError = (message) => {
  errorCard.textContent = message;
  errorCard.classList.remove("hidden");
  result.classList.add("hidden");
  emptyState.classList.add("hidden");
  outputCard.dataset.state = "idle";
};

const showToast = (message) => {
  if (!toast) {
    return;
  }
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => {
    toast.classList.add("hidden");
  }, 3000);
};

const resetOutput = () => {
  result.classList.add("hidden");
  errorCard.classList.add("hidden");
  emptyState.classList.remove("hidden");
  outputCard.dataset.state = "idle";
  setStatus("Idle", "idle");
  meterFill.style.width = "0%";
  if (resultNote) {
    resultNote.textContent = "";
  }
};

const setMode = (mode) => {
  const isSingle = mode === "single";
  modeButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  singlePanel.classList.toggle("active", isSingle);
  batchPanel.classList.toggle("active", !isSingle);
  outputCard.classList.toggle("hidden", !isSingle);
  resetOutput();
  if (!isSingle) {
    emptyState.classList.add("hidden");
    result.classList.add("hidden");
    outputCard.dataset.state = "idle";
    setStatus("Idle", "idle");
  }
};

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    setMode(btn.dataset.mode);
  });
});

const updateResult = (data) => {
  const isUncertain = Boolean(data.is_uncertain);
  const predId = Number.isFinite(data.prediction_id)
    ? data.prediction_id
    : parseInt(data.prediction_id, 10);
  const frequentScore = clampPercent((data.prob_class1_frequent || 0) * 100);
  const confidenceScore = predId === 1 ? frequentScore : 100 - frequentScore;

  if (isUncertain) {
    resultLabel.textContent = data.prediction || "Non-frequent";
    confidenceValue.textContent = `${confidenceScore}%`;
    outputCard.dataset.state = "nonfrequent";
    if (resultNote) {
      resultNote.textContent = "Close to the threshold — treated as Non-frequent.";
    }
  } else {
    resultLabel.textContent = data.prediction || "Unknown";
    confidenceValue.textContent = `${confidenceScore}%`;
    outputCard.dataset.state = predId === 1 ? "frequent" : "nonfrequent";
    if (resultNote) {
      resultNote.textContent = "";
    }
  }
  meterFill.style.width = `${frequentScore}%`;

  emptyState.classList.add("hidden");
  errorCard.classList.add("hidden");
  result.classList.remove("hidden");
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textarea.value.trim();

  if (!text) {
    showError("Please paste a review before running analysis.");
    return;
  }

  analyzeButton.disabled = true;
  setStatus("Analyzing", "analyzing");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    updateResult(data);
    setStatus("Done", "done");
  } catch (error) {
    showError(error.message || "Prediction failed.");
    setStatus("Error", "idle");
  } finally {
    analyzeButton.disabled = false;
  }
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = csvInput.files && csvInput.files[0];
  if (!file) {
    showError("Please choose a CSV file to upload.");
    return;
  }

  setStatus("Uploading", "analyzing");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/predict_csv", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "CSV labeling failed.");
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "labeled_reviews.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);

    showToast("Labeled CSV downloaded as labeled_reviews.csv.");
    setStatus("Done", "done");
  } catch (error) {
    if (outputCard.classList.contains("hidden")) {
      showToast(error.message || "CSV labeling failed.");
    } else {
      showError(error.message || "CSV labeling failed.");
    }
    setStatus("Error", "idle");
  }
});

textarea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

clearButton.addEventListener("click", () => {
  textarea.value = "";
  resetOutput();
  textarea.focus();
});

clearCsvButton.addEventListener("click", () => {
  csvInput.value = "";
  resetOutput();
});

resetOutput();
