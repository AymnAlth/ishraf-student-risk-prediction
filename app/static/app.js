/**
 * منظومة إشراف التعليمية - وحدة الرصد الاستباقي ومتابعة انتظام الطلاب
 * Enterprise Educational Risk Prediction Dashboard Logic
 */

// Preset profiles for instant testing & evaluation
const PRESETS = {
    stable: {
        student_key: "STD-2026-101",
        grade_level_id: "4",
        lookback_attendance_records: 22,
        lookback_present_count: 22,
        lookback_absent_count: 0,
        lookback_late_count: 0,
        lookback_assessment_count: 4,
        lookback_assessment_average_percent: 92.0,
        lookback_assessment_minimum_percent: 86.0,
        lookback_homework_due_count: 8,
        lookback_homework_completed_count: 8,
        lookback_homework_missing_count: 0,
        lookback_negative_behavior_count: 0,
        lookback_negative_behavior_severity_sum: 0,
        lookback_positive_behavior_count: 3,
        transport_assigned_at_snapshot: "true",
        lookback_transport_boarded_count: 18,
        lookback_transport_absent_count: 0
    },
    moderate: {
        student_key: "STD-2026-204",
        grade_level_id: "3",
        lookback_attendance_records: 22,
        lookback_present_count: 17,
        lookback_absent_count: 3,
        lookback_late_count: 2,
        lookback_assessment_count: 3,
        lookback_assessment_average_percent: 68.0,
        lookback_assessment_minimum_percent: 58.0,
        lookback_homework_due_count: 6,
        lookback_homework_completed_count: 4,
        lookback_homework_missing_count: 2,
        lookback_negative_behavior_count: 1,
        lookback_negative_behavior_severity_sum: 2,
        lookback_positive_behavior_count: 1,
        transport_assigned_at_snapshot: "true",
        lookback_transport_boarded_count: 12,
        lookback_transport_absent_count: 4
    },
    critical: {
        student_key: "STD-2026-309",
        grade_level_id: "5",
        lookback_attendance_records: 22,
        lookback_present_count: 11,
        lookback_absent_count: 7,
        lookback_late_count: 3,
        lookback_assessment_count: 2,
        lookback_assessment_average_percent: 48.0,
        lookback_assessment_minimum_percent: 40.0,
        lookback_homework_due_count: 6,
        lookback_homework_completed_count: 1,
        lookback_homework_missing_count: 5,
        lookback_negative_behavior_count: 3,
        lookback_negative_behavior_severity_sum: 8,
        lookback_positive_behavior_count: 0,
        transport_assigned_at_snapshot: "true",
        lookback_transport_boarded_count: 4,
        lookback_transport_absent_count: 8
    }
};

let activeEvaluatedCsvBlob = null;
let activeEvaluatedFilename = "evaluated_students.csv";
let selectedBatchFile = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    checkSystemHealth();
    setupDropzone();
});

/**
 * Check system connectivity and model status
 */
async function checkSystemHealth() {
    const statusEl = document.getElementById("system-status");
    if (!statusEl) return;

    try {
        const response = await fetch("/health");
        if (response.ok) {
            const data = await response.json();
            const thresholdPct = Math.round((data.calibrated_threshold || 0.385) * 1000) / 10;
            statusEl.innerHTML = `<span class="status-indicator"></span> النظام متصل ومفعل (إصدار ${data.model_version} | عتبة التنبيه: ${thresholdPct}%)`;
        } else {
            statusEl.innerHTML = `<span class="status-indicator" style="background:#ef4444;"></span> تعذر الاتصال بنموذج التقييم`;
        }
    } catch (e) {
        statusEl.innerHTML = `<span class="status-indicator" style="background:#ef4444;"></span> انقطع الاتصال بالخادم`;
    }
}

/**
 * Switch between Single Evaluation and Batch Evaluation views
 */
function switchView(viewName) {
    const singleBtn = document.getElementById("btn-tab-single");
    const batchBtn = document.getElementById("btn-tab-batch");
    const singleView = document.getElementById("view-single");
    const batchView = document.getElementById("view-batch");

    if (viewName === "single") {
        singleBtn.classList.add("active");
        batchBtn.classList.remove("active");
        singleView.classList.add("active");
        batchView.classList.remove("active");
    } else {
        batchBtn.classList.add("active");
        singleBtn.classList.remove("active");
        batchView.classList.add("active");
        singleView.classList.remove("active");
    }
}

/**
 * Load predefined student scenario and trigger evaluation
 */
function applyPreset(presetKey) {
    const preset = PRESETS[presetKey];
    if (!preset) return;

    for (const [key, val] of Object.entries(preset)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = val;
        }
    }

    // Trigger form submission
    const form = document.getElementById("student-eval-form");
    if (form) {
        form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
}

/**
 * Handle Single Student Evaluation Form Submission
 */
async function submitSingleEval(event) {
    if (event) event.preventDefault();

    const submitBtn = document.getElementById("btn-eval-submit");
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = "جاري التحليل واستخراج التقرير...";

    const records = parseInt(document.getElementById("lookback_attendance_records").value) || 22;
    const present = parseInt(document.getElementById("lookback_present_count").value) || 0;
    const attPct = Math.round((present / Math.max(records, 1)) * 1000) / 10;

    const payload = {
        student_key: document.getElementById("student_key").value || "STD-DEFAULT",
        grade_level_id: parseInt(document.getElementById("grade_level_id").value) || 4,
        days_since_enrollment: 280,
        lookback_attendance_records: records,
        lookback_present_count: present,
        lookback_absent_count: parseInt(document.getElementById("lookback_absent_count").value) || 0,
        lookback_late_count: parseInt(document.getElementById("lookback_late_count").value) || 0,
        lookback_excused_count: 0,
        lookback_attendance_percent: attPct,
        lookback_assessment_count: parseInt(document.getElementById("lookback_assessment_count").value) || 0,
        lookback_assessment_average_percent: parseFloat(document.getElementById("lookback_assessment_average_percent").value) || null,
        lookback_assessment_minimum_percent: parseFloat(document.getElementById("lookback_assessment_minimum_percent").value) || null,
        lookback_homework_due_count: parseInt(document.getElementById("lookback_homework_due_count").value) || 0,
        lookback_homework_completed_count: parseInt(document.getElementById("lookback_homework_completed_count").value) || 0,
        lookback_homework_late_count: 0,
        lookback_homework_missing_count: parseInt(document.getElementById("lookback_homework_missing_count").value) || 0,
        lookback_negative_behavior_count: parseInt(document.getElementById("lookback_negative_behavior_count").value) || 0,
        lookback_positive_behavior_count: parseInt(document.getElementById("lookback_positive_behavior_count").value) || 0,
        lookback_negative_behavior_severity_sum: parseInt(document.getElementById("lookback_negative_behavior_severity_sum").value) || 0,
        transport_assigned_at_snapshot: document.getElementById("transport_assigned_at_snapshot").value === "true",
        lookback_transport_boarded_count: parseInt(document.getElementById("lookback_transport_boarded_count").value) || 0,
        lookback_transport_absent_count: parseInt(document.getElementById("lookback_transport_absent_count").value) || 0
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "تعذر معالجة الطلب في الخادم");
        }

        const data = await response.json();
        renderSingleResult(data);
    } catch (err) {
        alert("تنبيه: " + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}

/**
 * Render single evaluation results in UI
 */
function renderSingleResult(data) {
    const emptyState = document.getElementById("eval-empty");
    const resultBody = document.getElementById("eval-result");
    const timestampEl = document.getElementById("eval-timestamp");
    const scoreCard = document.getElementById("score-card-box");
    const riskPctEl = document.getElementById("risk-pct");
    const riskBadge = document.getElementById("risk-badge");
    const riskBarFill = document.getElementById("risk-bar-fill");
    const factorsList = document.getElementById("diagnostic-factors");
    const actionsList = document.getElementById("action-recommendations");

    emptyState.style.display = "none";
    resultBody.style.display = "block";

    // Format current timestamp
    const now = new Date();
    timestampEl.textContent = `تم التقييم: ${now.toLocaleTimeString("ar-YE", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;

    // Percentage
    const pct = Math.round(data.risk_probability * 100);
    riskPctEl.textContent = `${pct}%`;

    // Reset card classes
    scoreCard.className = "score-card";
    riskBarFill.className = "bar-fill";
    riskBarFill.style.width = `${Math.min(pct, 100)}%`;

    let levelClass = "low";
    let badgeText = "وضع مستقر ومنتظم";

    if (data.risk_level === "CRITICAL") {
        levelClass = "critical";
        badgeText = "خطر حرج جداً (انقطاع محتمل)";
    } else if (data.risk_level === "HIGH") {
        levelClass = "high";
        badgeText = "خطر مرتفع (يستلزم التدخل)";
    } else if (data.risk_level === "MODERATE") {
        levelClass = "moderate";
        badgeText = "خطر متوسط (متابعة وقائية)";
    } else {
        levelClass = "low";
        badgeText = "وضع مستقر ومنتظم";
    }

    scoreCard.classList.add(levelClass);
    riskBarFill.classList.add(levelClass);
    riskBadge.textContent = badgeText;

    // Diagnostic factors
    factorsList.innerHTML = "";
    if (data.primary_risk_factors && data.primary_risk_factors.length > 0) {
        data.primary_risk_factors.forEach(factor => {
            const li = document.createElement("li");
            li.textContent = factor;
            factorsList.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.textContent = "لا توجد مؤشرات خطر مسجلة؛ أداء الطالب ضمن الحدود الطبيعية المقبولة.";
        factorsList.appendChild(li);
    }

    // Action recommendations
    actionsList.innerHTML = "";
    if (data.recommended_interventions && data.recommended_interventions.length > 0) {
        data.recommended_interventions.forEach(action => {
            const li = document.createElement("li");
            li.textContent = action;
            actionsList.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.textContent = "الاستمرار في تعزيز التزام الطالب والمتابعة الروتينية لضمان استدامة التفوق.";
        actionsList.appendChild(li);
    }
}

/**
 * Configure drag-and-drop for Batch CSV
 */
function setupDropzone() {
    const dropzone = document.getElementById("drop-area");
    if (!dropzone) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "#2563eb";
            dropzone.style.backgroundColor = "#eff6ff";
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "";
            dropzone.style.backgroundColor = "";
        }, false);
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedBatchFile(files[0]);
        }
    });
}

/**
 * Handle CSV selection from file input
 */
function onBatchFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        handleSelectedBatchFile(file);
    }
}

function handleSelectedBatchFile(file) {
    if (!file.name.endsWith(".csv")) {
        alert("يرجى اختيار ملف بصيغة CSV صالحة.");
        return;
    }

    selectedBatchFile = file;
    const badge = document.getElementById("file-info-badge");
    const btn = document.getElementById("btn-process-batch");

    const sizeKb = (file.size / 1024).toFixed(1);
    badge.textContent = `الملف المختار: ${file.name} (${sizeKb} KB)`;
    badge.style.display = "inline-block";
    btn.disabled = false;
}

/**
 * Process Batch File and download evaluated CSV
 */
async function processBatchFile() {
    if (!selectedBatchFile) return;

    const btn = document.getElementById("btn-process-batch");
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "جاري رفع الكشف وتطبيق نموذج التقييم...";

    const formData = new FormData();
    formData.append("file", selectedBatchFile);

    try {
        const response = await fetch("/predict/csv", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "حدث خطأ أثناء معالجة الكشف");
        }

        const blob = await response.blob();
        activeEvaluatedCsvBlob = blob;
        activeEvaluatedFilename = `evaluated_${selectedBatchFile.name}`;

        // Parse CSV text for summary statistics
        const text = await blob.text();
        const lines = text.trim().split("\n");
        const header = lines[0].split(",");
        const riskIdx = header.indexOf("is_at_risk");

        let total = 0;
        let flagged = 0;
        let stable = 0;

        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            total++;
            const cols = lines[i].split(",");
            if (riskIdx !== -1) {
                const val = cols[riskIdx].trim().toLowerCase();
                if (val === "true" || val === "1") {
                    flagged++;
                } else {
                    stable++;
                }
            } else {
                stable++;
            }
        }

        // Update stats
        document.getElementById("stat-total").textContent = total;
        document.getElementById("stat-flagged").textContent = flagged;
        document.getElementById("stat-stable").textContent = stable;

        // Display results wrapper
        document.getElementById("batch-results-wrapper").style.display = "block";

        // Setup download button
        const dlBtn = document.getElementById("btn-download-evaluated");
        dlBtn.onclick = () => {
            const url = window.URL.createObjectURL(activeEvaluatedCsvBlob);
            const a = document.createElement("a");
            a.href = url;
            a.download = activeEvaluatedFilename;
            document.body.appendChild(a);
            a.click();
            a.remove();
        };

        // Automatically trigger download as well
        dlBtn.click();
    } catch (e) {
        alert("فشل فحص الكشف: " + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}
