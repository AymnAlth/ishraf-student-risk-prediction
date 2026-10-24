// Ishraf Student Risk Web Interface Logic

const PRESETS = {
    stable: {
        student_key: "STUDENT_STABLE_001",
        grade_level_id: 4,
        lookback_attendance_records: 22,
        lookback_present_count: 21,
        lookback_absent_count: 0,
        lookback_late_count: 1,
        lookback_assessment_count: 4,
        lookback_assessment_average_percent: 88.5,
        lookback_assessment_minimum_percent: 82.0,
        lookback_homework_due_count: 6,
        lookback_homework_completed_count: 6,
        lookback_homework_missing_count: 0,
        lookback_negative_behavior_count: 0,
        lookback_negative_behavior_severity_sum: 0,
        lookback_positive_behavior_count: 3,
        transport_assigned_at_snapshot: "true",
        lookback_transport_boarded_count: 16,
        lookback_transport_absent_count: 0
    },
    moderate: {
        student_key: "STUDENT_MODERATE_002",
        grade_level_id: 3,
        lookback_attendance_records: 20,
        lookback_present_count: 16,
        lookback_absent_count: 3,
        lookback_late_count: 1,
        lookback_assessment_count: 3,
        lookback_assessment_average_percent: 71.0,
        lookback_assessment_minimum_percent: 62.0,
        lookback_homework_due_count: 5,
        lookback_homework_completed_count: 3,
        lookback_homework_missing_count: 2,
        lookback_negative_behavior_count: 1,
        lookback_negative_behavior_severity_sum: 2,
        lookback_positive_behavior_count: 1,
        transport_assigned_at_snapshot: "true",
        lookback_transport_boarded_count: 10,
        lookback_transport_absent_count: 4
    },
    critical: {
        student_key: "STUDENT_AT_RISK_003",
        grade_level_id: 5,
        lookback_attendance_records: 18,
        lookback_present_count: 11,
        lookback_absent_count: 6,
        lookback_late_count: 1,
        lookback_assessment_count: 2,
        lookback_assessment_average_percent: 54.0,
        lookback_assessment_minimum_percent: 45.0,
        lookback_homework_due_count: 6,
        lookback_homework_completed_count: 1,
        lookback_homework_missing_count: 5,
        lookback_negative_behavior_count: 3,
        lookback_negative_behavior_severity_sum: 7,
        lookback_positive_behavior_count: 0,
        transport_assigned_at_snapshot: "true",
        lookback_transport_boarded_count: 5,
        lookback_transport_absent_count: 6
    }
};

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
});

async function checkHealth() {
    try {
        const response = await fetch("/health");
        if (response.ok) {
            const data = await response.json();
            const badge = document.getElementById("system-status");
            badge.innerHTML = `<span class="dot"></span> النظام متصل ومفعل (v${data.model_version}) | عتبة الخطر: ${data.calibrated_threshold}`;
        }
    } catch (e) {
        document.getElementById("system-status").innerHTML = `<span class="dot" style="background:#ef4444;"></span> فشل الاتصال بالنموذج`;
    }
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
    
    document.getElementById(`tab-${tabId}`).classList.add("active");
    event.target.classList.add("active");
}

function loadPreset(type) {
    const preset = PRESETS[type];
    if (!preset) return;

    for (const [key, value] of Object.entries(preset)) {
        const el = document.getElementById(key);
        if (el) el.value = value;
    }
    
    // Auto predict preset
    document.getElementById("prediction-form").dispatchEvent(new Event("submit"));
}

async function handleSinglePredict(event) {
    event.preventDefault();
    
    const records = parseInt(document.getElementById("lookback_attendance_records").value) || 20;
    const present = parseInt(document.getElementById("lookback_present_count").value) || 0;
    const attPercent = Math.round((present / Math.max(records, 1)) * 1000) / 10;

    const payload = {
        student_key: document.getElementById("student_key").value,
        grade_level_id: parseInt(document.getElementById("grade_level_id").value),
        days_since_enrollment: 280,
        lookback_attendance_records: records,
        lookback_present_count: present,
        lookback_absent_count: parseInt(document.getElementById("lookback_absent_count").value) || 0,
        lookback_late_count: parseInt(document.getElementById("lookback_late_count").value) || 0,
        lookback_excused_count: 0,
        lookback_attendance_percent: attPercent,
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
            alert("حدث خطأ أثناء إجراء التحليل التنبؤي");
            return;
        }

        const data = await response.json();
        renderResult(data);
    } catch (e) {
        alert("فشل الاتصال بخدمة التنبؤ: " + e.message);
    }
}

function renderResult(data) {
    document.getElementById("empty-state").style.display = "none";
    document.getElementById("result-content").style.display = "block";

    const probPct = Math.round(data.risk_probability * 100);
    document.getElementById("res-prob").textContent = `${probPct}%`;

    const meter = document.getElementById("risk-meter");
    const badge = document.getElementById("res-badge");

    badge.className = "risk-badge";

    if (data.risk_level === "CRITICAL") {
        meter.style.borderColor = "#ef4444";
        meter.style.backgroundColor = "#fee2e2";
        badge.classList.add("badge-critical");
        badge.textContent = "🔴 خطر حرج (CRITICAL)";
    } else if (data.risk_level === "HIGH") {
        meter.style.borderColor = "#f97316";
        meter.style.backgroundColor = "#ffedd5";
        badge.classList.add("badge-high");
        badge.textContent = "🟠 خطر مرتفع (HIGH)";
    } else if (data.risk_level === "MODERATE") {
        meter.style.borderColor = "#f59e0b";
        meter.style.backgroundColor = "#fef3c7";
        badge.classList.add("badge-moderate");
        badge.textContent = "🟡 خطر متوسط (MODERATE)";
    } else {
        meter.style.borderColor = "#10b981";
        meter.style.backgroundColor = "#dcfce7";
        badge.classList.add("badge-low");
        badge.textContent = "🟢 وضع مستقر (LOW RISK)";
    }

    const factorsUl = document.getElementById("res-factors");
    factorsUl.innerHTML = "";
    data.primary_risk_factors.forEach(f => {
        const li = document.createElement("li");
        li.textContent = f;
        factorsUl.appendChild(li);
    });

    const actionsUl = document.getElementById("res-actions");
    actionsUl.innerHTML = "";
    data.recommended_interventions.forEach(a => {
        const li = document.createElement("li");
        li.textContent = a;
        actionsUl.appendChild(li);
    });
}

let selectedCsvFile = null;

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedCsvFile = file;
        document.getElementById("selected-file-name").textContent = `الملف المختار: ${file.name} (${Math.round(file.size / 1024)} KB)`;
        document.getElementById("btn-upload-csv").disabled = false;
    }
}

async function uploadCsvFile() {
    if (!selectedCsvFile) return;

    const formData = new FormData();
    formData.append("file", selectedCsvFile);

    const btn = document.getElementById("btn-upload-csv");
    btn.disabled = true;
    btn.textContent = "جاري معالجة السجلات وتطبيق النموذج...";

    try {
        const response = await fetch("/predict/csv", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            alert("حدث خطأ أثناء معالجة الملف");
            btn.disabled = false;
            btn.textContent = "معالجة الملف وتنزيل النتائج";
            return;
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `evaluated_${selectedCsvFile.name}`;
        document.body.appendChild(a);
        a.click();
        a.remove();

        btn.disabled = false;
        btn.textContent = "✅ اكتملت المعالجة وتنزيل النتائج بنجاح";
    } catch (e) {
        alert("فشل رفع الملف: " + e.message);
        btn.disabled = false;
        btn.textContent = "معالجة الملف وتنزيل النتائج";
    }
}
