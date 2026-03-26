const featureGrid = document.getElementById("feature-grid");
const form = document.getElementById("prediction-form");
const sampleButton = document.getElementById("sample-button");
const clearButton = document.getElementById("clear-button");
const formMessage = document.getElementById("form-message");
const healthStatus = document.getElementById("health-status");
const probabilityValue = document.getElementById("probability-value");
const riskBadge = document.getElementById("risk-badge");
const resultSummary = document.getElementById("result-summary");
const meterFill = document.getElementById("meter-fill");
const predictButton = document.getElementById("predict-button");

const featureNames = Array.from({ length: 28 }, (_, index) => `V${index + 1}`);

const samplePayload = {
  Time: 0.0,
  Amount: 149.62,
  V1: -1.3598071336738,
  V2: -0.0727811733098497,
  V3: 2.53634673796914,
  V4: 1.37815522427443,
  V5: -0.338320769942518,
  V6: 0.462387777762292,
  V7: 0.239598554061257,
  V8: 0.0986979012610507,
  V9: 0.363786969611213,
  V10: 0.0907941719789316,
  V11: -0.551599533260813,
  V12: -0.617800855762348,
  V13: -0.991389847235408,
  V14: -0.311169353699879,
  V15: 1.46817697209427,
  V16: -0.470400525259478,
  V17: 0.207971241929242,
  V18: 0.0257905801985591,
  V19: 0.403992960255733,
  V20: 0.251412098239705,
  V21: -0.018306777944153,
  V22: 0.277837575558899,
  V23: -0.110473910188767,
  V24: 0.0669280749146731,
  V25: 0.128539358273528,
  V26: -0.189114843888824,
  V27: 0.133558376740387,
  V28: -0.0210530534538215,
};

function buildFeatureInputs() {
  featureNames.forEach((feature) => {
    const label = document.createElement("label");
    label.className = "field";

    const title = document.createElement("span");
    title.textContent = feature;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.name = feature;
    input.placeholder = "0";

    label.appendChild(title);
    label.appendChild(input);
    featureGrid.appendChild(label);
  });
}

function setFormValues(payload) {
  Object.entries(payload).forEach(([key, value]) => {
    const input = form.elements.namedItem(key);
    if (input) {
      input.value = value;
    }
  });
}

function clearFormValues() {
  Array.from(form.querySelectorAll("input")).forEach((input) => {
    input.value = "";
  });
}

function getPayloadFromForm() {
  const payload = {};
  const fields = ["Time", "Amount", ...featureNames];

  fields.forEach((field) => {
    const rawValue = form.elements.namedItem(field).value;
    payload[field] = Number(rawValue);
  });

  return payload;
}

function setRiskState(probability) {
  const percent = probability * 100;
  probabilityValue.textContent = `${percent.toFixed(2)}%`;
  meterFill.style.width = `${Math.min(Math.max(percent, 0), 100)}%`;

  riskBadge.className = "risk-badge";
  if (probability < 0.2) {
    riskBadge.classList.add("risk-low");
    riskBadge.textContent = "Low signal";
    resultSummary.textContent = "The model sees this transaction as relatively typical compared with known fraud patterns.";
    return;
  }

  if (probability < 0.6) {
    riskBadge.classList.add("risk-medium");
    riskBadge.textContent = "Review";
    resultSummary.textContent = "The transaction has some suspicious characteristics and may deserve secondary review.";
    return;
  }

  riskBadge.classList.add("risk-high");
  riskBadge.textContent = "High signal";
  resultSummary.textContent = "The feature pattern is strongly aligned with transactions the model considers suspicious.";
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    healthStatus.textContent = data.status === "ok" ? "Online" : "Unavailable";
  } catch (error) {
    healthStatus.textContent = "Offline";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formMessage.textContent = "Running model...";
  predictButton.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getPayloadFromForm()),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail ? JSON.stringify(data.detail) : "Prediction failed");
    }

    setRiskState(data.fraud_proba);
    formMessage.textContent = "Prediction complete.";
  } catch (error) {
    probabilityValue.textContent = "--";
    meterFill.style.width = "0%";
    riskBadge.className = "risk-badge risk-neutral";
    riskBadge.textContent = "Error";
    resultSummary.textContent = "The server could not score this transaction. Check the input values and try again.";
    formMessage.textContent = error.message;
  } finally {
    predictButton.disabled = false;
  }
});

sampleButton.addEventListener("click", () => {
  clearFormValues();
  setFormValues(samplePayload);
  formMessage.textContent = "Loaded a sample transaction from the dataset.";
});

clearButton.addEventListener("click", () => {
  clearFormValues();
  probabilityValue.textContent = "--";
  meterFill.style.width = "0%";
  riskBadge.className = "risk-badge risk-neutral";
  riskBadge.textContent = "Waiting";
  resultSummary.textContent = "Submit a transaction to see the model score and a plain-language interpretation.";
  formMessage.textContent = "Form cleared.";
});

buildFeatureInputs();
checkHealth();
