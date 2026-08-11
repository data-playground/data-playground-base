// static/js/jobs_enhancements.js
// ─────────────────────────────────────────────────────────────────────────
// Life OS — Jobs Page Enhancements
//   1. Keyboard navigation  (j/k, /, a/p/i/t/r/o/c, Enter, Escape)
//   2. Column sort          (click any header → sort asc/desc)
//   3. Status row tinting   (applied to existing rows via data-status)
// Load this at the bottom of jobs.html via <script src="...">
// ─────────────────────────────────────────────────────────────────────────

(function () {
    'use strict';

    // ── 1. STATUS ROW TINTING ────────────────────────────────────────────────
    // Reads data-status on each <tr> and adds a CSS class for tinting.
    // CSS handles the actual visual — this just stamps the class.

    function applyStatusTinting() {
        document.querySelectorAll('#jobs-table tbody tr').forEach(row => {
            const status = (row.dataset.status || '').toUpperCase();
            // Remove any previous tint class
            row.classList.remove(
                'tint-rejected', 'tint-closed', 'tint-offer',
                'tint-interviewing', 'tint-applied', 'tint-phone'
            );
            if (status === 'REJECTED') row.classList.add('tint-rejected');
            else if (status === 'CLOSED') row.classList.add('tint-closed');
            else if (status === 'OFFER') row.classList.add('tint-offer');
            else if (status === 'INTERVIEWING') row.classList.add('tint-interviewing');
            else if (status === 'APPLIED') row.classList.add('tint-applied');
            else if (status === 'PHONE_SCREEN') row.classList.add('tint-phone');
        });
    }

    // Re-tint after ATS button updates (HTMX swap)
    document.addEventListener('htmx:afterSwap', function (evt) {
        if (evt.target && evt.target.classList.contains('ats-buttons')) {
            // The row's data-status was updated in the htmx:afterSwap handler
            // in jobs.html — run tinting after that runs
            setTimeout(applyStatusTinting, 20);
        }
    });


    // ── 2. COLUMN SORT ───────────────────────────────────────────────────────

    let sortCol  = 'score';   // current sort column key
    let sortDir  = 'desc';    // 'asc' | 'desc'

    // Map header index → data attribute / sort function
    const SORT_DEFS = [
        { key: 'score',   label: 'Score',    getter: r => parseInt(r.dataset.score || '0') },
        { key: 'title',   label: 'Role',     getter: r => r.querySelector('.job-title-link')?.textContent?.trim().toLowerCase() || '' },
        { key: 'loc',     label: 'Location', getter: r => r.querySelector('.tag')?.textContent?.trim().toLowerCase() || '' },
        { key: 'date',    label: 'Posted',   getter: r => r.dataset.postdate || '' },
        { key: 'status',  label: 'Status',   getter: r => r.dataset.status || '' },
    ];

    function initColumnSort() {
        const thead = document.querySelector('#jobs-table thead tr');
        if (!thead) return;

        const headers = thead.querySelectorAll('th');
        headers.forEach((th, idx) => {
            if (!SORT_DEFS[idx]) return;
            th.style.cursor = 'pointer';
            th.style.userSelect = 'none';
            th.title = `Sort by ${SORT_DEFS[idx].label}`;

            // Add sort indicator span
            const indicator = document.createElement('span');
            indicator.className = 'sort-indicator';
            indicator.style.cssText = 'margin-left:4px;font-size:9px;opacity:0.4;';
            indicator.textContent = '↕';
            th.appendChild(indicator);

            th.addEventListener('click', () => {
                const def = SORT_DEFS[idx];
                if (sortCol === def.key) {
                    sortDir = sortDir === 'asc' ? 'desc' : 'asc';
                } else {
                    sortCol = def.key;
                    sortDir = def.key === 'score' ? 'desc' : 'asc';
                }
                sortTable();
                updateSortIndicators(headers);
            });
        });

        // Apply initial sort indicator on Score header
        updateSortIndicators(headers);
    }

    function updateSortIndicators(headers) {
        headers.forEach((th, idx) => {
            const ind = th.querySelector('.sort-indicator');
            if (!ind || !SORT_DEFS[idx]) return;
            if (SORT_DEFS[idx].key === sortCol) {
                ind.textContent = sortDir === 'asc' ? ' ↑' : ' ↓';
                ind.style.opacity = '1';
                ind.style.color = 'var(--accent)';
            } else {
                ind.textContent = ' ↕';
                ind.style.opacity = '0.3';
                ind.style.color = '';
            }
        });
    }

    function sortTable() {
        const tbody = document.querySelector('#jobs-table tbody');
        if (!tbody) return;

        const def = SORT_DEFS.find(d => d.key === sortCol);
        if (!def) return;

        const rows = Array.from(tbody.querySelectorAll('tr'));

        rows.sort((a, b) => {
            const va = def.getter(a);
            const vb = def.getter(b);
            let cmp = 0;
            if (typeof va === 'number') cmp = va - vb;
            else cmp = String(va).localeCompare(String(vb));
            return sortDir === 'asc' ? cmp : -cmp;
        });

        // Re-append in sorted order
        rows.forEach(r => tbody.appendChild(r));
    }


    // ── 3. KEYBOARD NAVIGATION ───────────────────────────────────────────────

    let focusedRowIdx = -1;   // index into visible rows
    let isTyping      = false; // true when focus is inside an input

    function getVisibleRows() {
        return Array.from(
            document.querySelectorAll('#jobs-table tbody tr:not([style*="display: none"])')
        );
    }

    function focusRow(idx) {
        const rows = getVisibleRows();
        if (!rows.length) return;

        // Clamp
        idx = Math.max(0, Math.min(rows.length - 1, idx));
        focusedRowIdx = idx;

        // Remove previous highlight
        document.querySelectorAll('#jobs-table tbody tr').forEach(r =>
            r.classList.remove('row-focused')
        );

        rows[idx].classList.add('row-focused');
        rows[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    function getStatusKeyMap() {
        // Keys to ATS status strings
        return {
            a: 'APPLIED',
            p: 'PHONE_SCREEN',
            i: 'INTERVIEWING',
            t: 'TECHNICAL_ASSESSMENT',
            r: 'REJECTED',
            o: 'OFFER',
            c: 'CLOSED',
        };
    }

    function triggerAtsOnFocusedRow(statusKey) {
        const rows = getVisibleRows();
        if (focusedRowIdx < 0 || focusedRowIdx >= rows.length) return;
        const row = rows[focusedRowIdx];
        const jobId = row.dataset.jobId;
        if (!jobId) return;

        // Find the matching ATS button and click it
        const atsContainer = row.querySelector('.ats-buttons');
        if (!atsContainer) return;

        const buttons = atsContainer.querySelectorAll('.ats-btn');
        const statusMap = {
            APPLIED: 0, PHONE_SCREEN: 1, INTERVIEWING: 2,
            TECHNICAL_ASSESSMENT: 3, REJECTED: 4, OFFER: 5, CLOSED: 6
        };
        const btnIdx = statusMap[statusKey];
        if (buttons[btnIdx]) {
            buttons[btnIdx].click();
            showToast(`[${statusKey.replace('_',' ')}] logged via keyboard`);
        }
    }

    function openDetailForFocusedRow() {
        const rows = getVisibleRows();
        if (focusedRowIdx < 0 || focusedRowIdx >= rows.length) return;
        const row = rows[focusedRowIdx];
        const jobId = row.dataset.jobId;
        if (!jobId) return;

        htmx.ajax('GET', `/jobs/detail/${jobId}`, {
            target: '#job-detail-panel',
            swap: 'innerHTML'
        });
        if (typeof openDetailPanel === 'function') openDetailPanel();
    }

    function initKeyboardNav() {
        // Track whether user is typing in an input
        document.addEventListener('focusin', e => {
            const tag = e.target.tagName;
            isTyping = ['INPUT', 'TEXTAREA', 'SELECT'].includes(tag);
        });
        document.addEventListener('focusout', () => { isTyping = false; });

        document.addEventListener('keydown', e => {
            // Never hijack when user is typing in a form field
            if (isTyping) return;
            // Never hijack with modifier keys (browser shortcuts)
            if (e.metaKey || e.ctrlKey || e.altKey) return;

            const key = e.key.toLowerCase();

            // ── Navigation ──
            if (key === 'j' || key === 'arrowdown') {
                e.preventDefault();
                const rows = getVisibleRows();
                focusRow(focusedRowIdx < 0 ? 0 : focusedRowIdx + 1);
                return;
            }

            if (key === 'k' || key === 'arrowup') {
                e.preventDefault();
                focusRow(focusedRowIdx <= 0 ? 0 : focusedRowIdx - 1);
                return;
            }

            // ── Open detail drawer ──
            if (key === 'enter' && focusedRowIdx >= 0) {
                e.preventDefault();
                openDetailForFocusedRow();
                return;
            }

            // ── Close detail drawer / deselect ──
            if (key === 'escape') {
                if (typeof closeDetailPanel === 'function') closeDetailPanel();
                document.querySelectorAll('#jobs-table tbody tr').forEach(r =>
                    r.classList.remove('row-focused')
                );
                focusedRowIdx = -1;
                return;
            }

            // ── Focus company filter ──
            if (key === '/') {
                e.preventDefault();
                const compInput = document.getElementById('filter-company');
                if (compInput) {
                    compInput.focus();
                    compInput.select();
                }
                return;
            }

            // ── ATS quick-log ──
            const statusMap = getStatusKeyMap();
            if (statusMap[key] && focusedRowIdx >= 0) {
                e.preventDefault();
                triggerAtsOnFocusedRow(statusMap[key]);
                return;
            }

            // ── g → go to top ──
            if (key === 'g') {
                focusRow(0);
                return;
            }

            // ── G → go to bottom ──
            if (e.key === 'G') {
                const rows = getVisibleRows();
                focusRow(rows.length - 1);
                return;
            }
        });
    }

    // Show a keyboard shortcut legend (toggleable)
    function renderShortcutLegend() {
        const existing = document.getElementById('kbd-legend');
        if (existing) { existing.remove(); return; }

        const legend = document.createElement('div');
        legend.id = 'kbd-legend';
        legend.innerHTML = `
            <div style="
                position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
                background:var(--bg-card); border:1px solid var(--border);
                border-radius:8px; padding:16px 20px; z-index:500;
                box-shadow:var(--shadow-lg); font-size:10px;
                display:grid; grid-template-columns:repeat(3,1fr); gap:6px 24px;
                min-width:340px;
            ">
                <div style="grid-column:1/-1;font-family:'Syne',sans-serif;font-weight:700;
                            font-size:11px;margin-bottom:6px;color:var(--text-primary);">
                    Keyboard Shortcuts
                    <button onclick="document.getElementById('kbd-legend').remove()"
                        style="float:right;background:none;border:none;cursor:pointer;
                               color:var(--text-muted);font-size:14px;line-height:1;">✕</button>
                </div>
                ${[
                    ['j / ↓', 'Next row'],
                    ['k / ↑', 'Prev row'],
                    ['Enter', 'Open detail'],
                    ['Esc', 'Close / deselect'],
                    ['/', 'Focus search'],
                    ['g / G', 'First / Last'],
                    ['a', 'Mark Applied'],
                    ['p', 'Mark Phone Screen'],
                    ['i', 'Mark Interviewing'],
                    ['t', 'Mark Technical'],
                    ['r', 'Mark Rejected'],
                    ['o', 'Mark Offer'],
                    ['c', 'Mark Closed'],
                    ['?', 'Toggle this help'],
                ].map(([k, v]) => `
                    <div style="display:flex;gap:8px;align-items:center;">
                        <kbd style="background:var(--bg-hover);border:1px solid var(--border);
                                    border-radius:3px;padding:1px 6px;font-family:'DM Mono',monospace;
                                    font-size:9px;color:var(--accent);">${k}</kbd>
                        <span style="color:var(--text-secondary);">${v}</span>
                    </div>
                `).join('')}
            </div>
        `;
        document.body.appendChild(legend);
    }

    // ? key toggles legend
    document.addEventListener('keydown', e => {
        if (isTyping) return;
        if (e.key === '?') renderShortcutLegend();
    });

    // Add a subtle ? hint in the topbar
    function addShortcutHint() {
        const topbarRight = document.querySelector('.topbar-right');
        if (!topbarRight) return;
        const hint = document.createElement('button');
        hint.className = 'btn';
        hint.title = 'Keyboard shortcuts (?)';
        hint.innerHTML = '⌨ <span style="font-size:9px;opacity:.7;">?</span>';
        hint.onclick = renderShortcutLegend;
        hint.style.padding = '4px 8px';
        topbarRight.insertBefore(hint, topbarRight.firstChild);
    }


    // ── INIT ─────────────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', function () {
        applyStatusTinting();
        initColumnSort();
        initKeyboardNav();
        addShortcutHint();
    });

    // Expose for HTMX re-tinting after ATS updates
    window._reapplyStatusTinting = applyStatusTinting;

    // Expose for jobs.html's fetchJobs() to call after a filter change or
    // Load More — focusedRowIdx lives in this file's closure, so it can't
    // be reset directly from outside; this is the same pattern as
    // window._reapplyStatusTinting above. Only resets on a REPLACE (filters
    // changed), not on an append (Load More) — the previously-focused row
    // is still valid and still in the DOM in that case.
    window._resetKeyboardFocus = function () {
        focusedRowIdx = -1;
        document.querySelectorAll('#jobs-table tbody tr').forEach(r =>
            r.classList.remove('row-focused')
        );
    };

    // Expose for jobs.html's fetchJobs() to call after inserting new rows
    // (filter change or Load More) — keeps the table consistent with
    // whatever column sort is currently active instead of silently
    // reverting to server order (fit_score DESC) while the header still
    // shows a sort arrow pointing at a different column.
    window._reapplySort = sortTable;

})();
