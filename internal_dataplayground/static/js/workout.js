// static/js/workout.js
// ─────────────────────────────────────────────────────────────────────────────
// Life OS — Workout Tracker Client Logic
// Handles: exercise search, set logging, session timer, unit toggle,
//          weight modal, end-session modal, body weight sparkline.
// ─────────────────────────────────────────────────────────────────────────────

'use strict';

// ── STATE ─────────────────────────────────────────────────────────────────────
let selectedExerciseId = null;
let selectedExerciseName = null;
let currentUnit = document.getElementById('log-unit-input')?.value || 'lb';
let selectedRPE = null;
let selectedFatigue = null;
let sessionTimerInterval = null;
let sessionStartTime = null;
let setCount = 0;

// ── INIT ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initSparkline();
    initSessionTimer();
    countExistingSets();

    // Close exercise dropdown when clicking elsewhere
    document.addEventListener('click', e => {
        const dropdown = document.getElementById('exercise-dropdown');
        const search = document.getElementById('exercise-search');
        if (dropdown && !dropdown.contains(e.target) && e.target !== search) {
            dropdown.style.display = 'none';
        }
    });

    // Keyboard: Escape closes dropdown
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            document.getElementById('exercise-dropdown').style.display = 'none';
        }
    });
});

function countExistingSets() {
    const rows = document.querySelectorAll('.set-row');
    setCount = rows.length;
    updateSetCount();
}

function updateSetCount() {
    const el = document.getElementById('session-set-count');
    if (el) el.textContent = `${setCount} set${setCount !== 1 ? 's' : ''}`;
}

// ── SESSION TIMER ─────────────────────────────────────────────────────────────
function initSessionTimer() {
    const timerEl = document.getElementById('session-timer');
    if (!timerEl || !SESSION_ID) return;

    const startedAtEl = document.getElementById('session-started-at');
    if (startedAtEl) {
        sessionStartTime = new Date(startedAtEl.dataset.started);
    } else {
        sessionStartTime = new Date();
    }

    sessionTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        const h = Math.floor(elapsed / 3600);
        const m = Math.floor((elapsed % 3600) / 60);
        const s = elapsed % 60;
        timerEl.textContent = h > 0
            ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
            : `${m}:${String(s).padStart(2,'0')}`;
    }, 1000);
}

// ── UNIT TOGGLE ───────────────────────────────────────────────────────────────
function setUnit(unit, btn) {
    currentUnit = unit;
    const input = document.getElementById('unit-input');
    if (input) input.value = unit;
    document.querySelectorAll('.unit-toggle .unit-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.unit === unit);
    });
}

function setLogUnit(unit, btn) {
    currentUnit = unit;
    const input = document.getElementById('log-unit-input');
    if (input) input.value = unit;
    document.querySelectorAll('.unit-toggle-sm .unit-btn-sm').forEach(b => {
        b.classList.toggle('active', b.dataset.unit === unit);
    });
    // Update weight placeholder to match unit
    const weightInput = document.getElementById('weight-input');
    if (weightInput) {
        weightInput.placeholder = unit === 'lb' ? '135' : '60';
    }
}

// ── EXERCISE SEARCH ───────────────────────────────────────────────────────────
function showExerciseDropdown() {
    if (!selectedExerciseId) {
        filterExercises('');
    }
}

function filterExercises(query) {
    const dropdown = document.getElementById('exercise-dropdown');
    if (!dropdown) return;

    const q = query.toLowerCase().trim();
    const filtered = q
        ? ALL_EXERCISES.filter(ex => ex.name.toLowerCase().includes(q))
        : ALL_EXERCISES;

    if (!filtered.length) {
        dropdown.innerHTML = '<div class="exercise-option"><span class="exercise-option-name" style="color:var(--text-muted);">No matches</span></div>';
        dropdown.style.display = 'block';
        return;
    }

    // Group by muscle for better UX when showing all
    const html = filtered.slice(0, 30).map(ex => `
        <div class="exercise-option" onclick="selectExercise(${ex.id}, '${ex.name.replace(/'/g, "\\'")}')">
            <span class="exercise-option-name">${ex.name}${ex.custom ? ' ★' : ''}</span>
            <span class="exercise-option-meta">${ex.muscle} · ${ex.equipment}</span>
        </div>
    `).join('');

    dropdown.innerHTML = html;
    dropdown.style.display = 'block';
}

async function selectExercise(id, name) {
    selectedExerciseId = id;
    selectedExerciseName = name;

    // Update hidden input
    const hiddenInput = document.getElementById('selected-exercise-id');
    if (hiddenInput) hiddenInput.value = id;

    // Hide search, show pill
    const search = document.getElementById('exercise-search');
    const pill = document.getElementById('selected-exercise-display');
    const nameSpan = document.getElementById('selected-exercise-name');
    const dropdown = document.getElementById('exercise-dropdown');

    if (search) search.style.display = 'none';
    if (dropdown) dropdown.style.display = 'none';
    if (pill) pill.style.display = 'flex';
    if (nameSpan) nameSpan.textContent = name;

    // Fetch previous best for this exercise
    await fetchPreviousBest(id);
}

function clearExercise() {
    selectedExerciseId = null;
    selectedExerciseName = null;

    const hiddenInput = document.getElementById('selected-exercise-id');
    if (hiddenInput) hiddenInput.value = '';

    const search = document.getElementById('exercise-search');
    const pill = document.getElementById('selected-exercise-display');

    if (search) { search.style.display = 'block'; search.value = ''; search.focus(); }
    if (pill) pill.style.display = 'none';

    const prevBox = document.getElementById('prev-best-display');
    if (prevBox) prevBox.style.display = 'none';
}

async function fetchPreviousBest(exerciseId) {
    const prevBox = document.getElementById('prev-best-display');
    if (!prevBox) return;

    try {
        const resp = await fetch(`/workout/history/${exerciseId}`);
        if (!resp.ok) { prevBox.style.display = 'none'; return; }

        const html = await resp.text();
        // Parse the first set from the history partial
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const firstRow = doc.querySelector('td[data-weight]');
        if (firstRow) {
            const weight = firstRow.dataset.weight;
            const reps = firstRow.dataset.reps;
            const unit = firstRow.dataset.unit || 'lb';
            prevBox.textContent = `Previous best: ${weight} ${unit} × ${reps} reps`;
            prevBox.style.display = 'flex';

            // Pre-fill weight input with previous weight
            const weightInput = document.getElementById('weight-input');
            if (weightInput && !weightInput.value) {
                weightInput.value = weight;
            }
        } else {
            prevBox.style.display = 'none';
        }
    } catch (e) {
        prevBox.style.display = 'none';
    }
}

// ── RPE ───────────────────────────────────────────────────────────────────────
function toggleRPE() {
    const panel = document.getElementById('rpe-panel');
    if (!panel) return;
    const visible = panel.style.display !== 'none';
    panel.style.display = visible ? 'none' : 'flex';
}

function selectRPE(val, btn) {
    selectedRPE = val;
    const input = document.getElementById('rpe-input');
    if (input) input.value = val;
    document.querySelectorAll('.rpe-btn').forEach(b => {
        b.classList.toggle('selected', parseInt(b.dataset.rpe) === val);
    });
    const display = document.getElementById('rpe-display');
    if (display) display.textContent = val;
}

// ── REPS ADJUSTER ─────────────────────────────────────────────────────────────
function adjustReps(delta) {
    const input = document.getElementById('reps-input');
    if (!input) return;
    const current = parseInt(input.value) || 0;
    input.value = Math.max(1, current + delta);
}

// ── LOG SET ───────────────────────────────────────────────────────────────────
async function logSet() {
    if (!SESSION_ID) { showToast('Start a session first.', true); return; }
    if (!selectedExerciseId) { showToast('Select an exercise first.', true); return; }

    const repsInput = document.getElementById('reps-input');
    const weightInput = document.getElementById('weight-input');
    const isWarmup = document.getElementById('is-warmup-toggle')?.checked || false;
    const rpe = document.getElementById('rpe-input')?.value || '';
    const unit = document.getElementById('log-unit-input')?.value || 'lb';

    const reps = parseInt(repsInput?.value) || 0;
    if (reps < 1) { showToast('Enter reps completed.', true); return; }

    const btn = document.getElementById('log-set-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Logging…'; }

    const fd = new FormData();
    fd.append('exercise_id', selectedExerciseId);
    fd.append('reps_completed', reps);
    fd.append('weight_used', weightInput?.value || '');
    fd.append('weight_unit', unit);
    fd.append('is_warmup', isWarmup ? 'true' : 'false');
    if (rpe) fd.append('rpe', rpe);

    try {
        const resp = await fetch(`/workout/sessions/${SESSION_ID}/sets`, {
            method: 'POST',
            body: fd,
        });

        if (!resp.ok) {
            const text = await resp.text();
            showToast(`Failed: ${text}`, true);
            return;
        }

        const html = await resp.text();

        // Append the new set row to the list
        const list = document.getElementById('sets-logged-list');
        if (list) {
            list.insertAdjacentHTML('beforeend', html);
            list.scrollTop = list.scrollHeight;
            setCount++;
            updateSetCount();
        }

        // Update the set pip for this exercise in the plan section
        updateSetPip(selectedExerciseId, isWarmup);

        // Reset form — keep exercise selected for fast repeat logging
        if (repsInput) repsInput.value = reps; // Keep same reps for next set
        selectedRPE = null;
        const rpeInput = document.getElementById('rpe-input');
        if (rpeInput) rpeInput.value = '';
        document.querySelectorAll('.rpe-btn').forEach(b => b.classList.remove('selected'));
        const rpeDisplay = document.getElementById('rpe-display');
        if (rpeDisplay) rpeDisplay.textContent = '—';
        if (document.getElementById('is-warmup-toggle')) {
            document.getElementById('is-warmup-toggle').checked = false;
        }

        showToast(isWarmup ? 'Warmup logged.' : 'Set logged ✓');

    } catch (e) {
        showToast(`Error: ${e.message}`, true);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Log Set'; }
    }
}

function updateSetPip(exerciseId, isWarmup) {
    if (isWarmup) return; // Warmup sets don't count toward target
    const rows = document.querySelectorAll('.exercise-target-row');
    rows.forEach(row => {
        const onclickAttr = row.getAttribute('onclick') || '';
        if (onclickAttr.includes(`prefillExercise(${exerciseId},`)) {
            const pips = row.querySelectorAll('.set-pip');
            const nextEmpty = Array.from(pips).find(p => !p.classList.contains('filled'));
            if (nextEmpty) nextEmpty.classList.add('filled');
            // Check if all sets done
            const allFilled = Array.from(pips).every(p => p.classList.contains('filled'));
            if (allFilled) row.classList.add('completed-row');
        }
    });
}

// ── PREFILL FROM PLAN ─────────────────────────────────────────────────────────
function prefillExercise(id, name, reps, weight, unit) {
    if (!SESSION_ID) return; // No session active, just viewing

    // Select the exercise
    selectExercise(id, name);

    // Set reps
    const repsInput = document.getElementById('reps-input');
    if (repsInput && reps) repsInput.value = reps;

    // Set weight if provided
    const weightInput = document.getElementById('weight-input');
    if (weightInput && weight > 0) weightInput.value = weight;

    // Set unit
    if (unit) {
        setLogUnit(unit, null);
    }

    // Scroll log form into view
    const logCard = document.getElementById('log-form-card');
    if (logCard) logCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── START SESSION ─────────────────────────────────────────────────────────────
async function startSession() {
    const form = document.getElementById('start-form');
    if (!form) return;

    const btn = form.querySelector('.start-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }

    const fd = new FormData(form);

    try {
        const resp = await fetch('/workout/sessions/start', {
            method: 'POST',
            body: fd,
        });

        if (resp.ok) {
            // Reload page to show the active session UI
            window.location.reload();
        } else {
            showToast('Could not start session. Try again.', true);
            if (btn) { btn.disabled = false; btn.textContent = '▶ Start Session'; }
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, true);
        if (btn) { btn.disabled = false; btn.textContent = '▶ Start Session'; }
    }
}

// ── END SESSION MODAL ─────────────────────────────────────────────────────────
function openEndModal() {
    document.getElementById('end-modal')?.classList.add('open');
}

function closeEndModal() {
    document.getElementById('end-modal')?.classList.remove('open');
}

function selectFatigue(val, btn) {
    selectedFatigue = val;
    document.querySelectorAll('.fatigue-btn').forEach(b => {
        b.classList.toggle('selected', parseInt(b.dataset.val) === val);
    });
    const input = document.getElementById('fatigue-input');
    if (input) input.value = val;
}

async function endSession() {
    if (!SESSION_ID) return;

    const notes = document.getElementById('end-notes')?.value || '';
    const fatigue = document.getElementById('fatigue-input')?.value || '';

    const fd = new FormData();
    if (fatigue) fd.append('fatigue_rating', fatigue);
    if (notes) fd.append('notes', notes);

    try {
        const resp = await fetch(`/workout/sessions/${SESSION_ID}/end`, {
            method: 'PATCH',
            body: fd,
        });

        if (resp.ok) {
            clearInterval(sessionTimerInterval);
            closeEndModal();

            // Replace the session zone with the summary partial
            const zone = document.getElementById('session-zone');
            const logCard = document.getElementById('log-form-card');
            if (zone) zone.innerHTML = await resp.text();
            if (logCard) logCard.style.display = 'none';
            showToast('Session complete 🎉');
        } else {
            showToast('Could not end session. Try again.', true);
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, true);
    }
}

// ── BODY WEIGHT ───────────────────────────────────────────────────────────────
function openWeightModal() {
    document.getElementById('weight-modal')?.classList.add('open');
    document.getElementById('bw-input')?.focus();
}

function closeWeightModal() {
    document.getElementById('weight-modal')?.classList.remove('open');
}

function setBWUnit(unit, btn) {
    const input = document.getElementById('bw-unit-input');
    if (input) input.value = unit;
    document.querySelectorAll('#weight-modal .unit-btn-sm').forEach(b => {
        b.classList.toggle('active', b.dataset.unit === unit);
    });
}

async function logWeight() {
    const weight = document.getElementById('bw-input')?.value;
    const unit = document.getElementById('bw-unit-input')?.value || 'lb';
    const bf = document.getElementById('bf-input')?.value || '';

    if (!weight) { showToast('Enter a weight.', true); return; }

    const fd = new FormData();
    fd.append('weight', weight);
    fd.append('weight_unit', unit);
    if (bf) fd.append('body_fat_pct', bf);

    try {
        const resp = await fetch('/workout/body-metrics', {
            method: 'POST',
            body: fd,
        });

        if (resp.ok) {
            closeWeightModal();
            showToast('Weight logged ✓');
            // Reload page to refresh sparkline
            setTimeout(() => window.location.reload(), 800);
        } else {
            showToast('Could not save weight.', true);
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, true);
    }
}

// ── BODY WEIGHT SPARKLINE ─────────────────────────────────────────────────────
function initSparkline() {
    const canvas = document.getElementById('weight-sparkline');
    if (!canvas || !BODY_METRICS || BODY_METRICS.length === 0) return;

    const validData = BODY_METRICS.filter(w => w !== null);
    if (validData.length < 2) return;

    const labels = BODY_METRIC_DATES;
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const lineColor = isDark ? 'rgba(124,111,255,0.9)' : 'rgba(85,72,232,0.9)';
    const fillColor = isDark ? 'rgba(124,111,255,0.1)' : 'rgba(85,72,232,0.07)';

    new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data: BODY_METRICS,
                borderColor: lineColor,
                backgroundColor: fillColor,
                borderWidth: 2,
                pointRadius: 2,
                tension: 0.3,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(128,128,128,0.1)' },
                    ticks: {
                        font: { family: "'DM Mono', monospace", size: 9 },
                        color: 'rgba(128,128,128,0.6)',
                        maxTicksLimit: 4,
                    },
                },
            },
        },
    });
}

// ── SESSION DETAIL DRAWER ─────────────────────────────────────────────────────
async function loadSessionDetail(sessionId) {
    const drawer = document.getElementById('session-detail-drawer');
    const overlay = document.getElementById('session-overlay');
    if (!drawer) return;

    drawer.innerHTML = '<div style="padding:24px;color:var(--text-muted);font-size:11px;">Loading…</div>';
    drawer.classList.add('open');
    if (overlay) overlay.classList.add('visible');

    try {
        const resp = await fetch(`/workout/sessions/${sessionId}`);
        if (resp.ok) {
            drawer.innerHTML = await resp.text();
        } else {
            drawer.innerHTML = '<div style="padding:24px;color:var(--red);font-size:11px;">Could not load session.</div>';
        }
    } catch (e) {
        drawer.innerHTML = `<div style="padding:24px;color:var(--red);font-size:11px;">Error: ${e.message}</div>`;
    }
}

function closeSessionDetail() {
    document.getElementById('session-detail-drawer')?.classList.remove('open');
    document.getElementById('session-overlay')?.classList.remove('visible');
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeSessionDetail();
});
