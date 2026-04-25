// static/js/base.js
// ─────────────────────────────────────────────────────────────────────────────
// Life OS — v2.1 Shared JS
// Provides: setTheme(), toggleSidebar(), showToast(),
//           openMobileSidebar(), closeMobileSidebar()
// ─────────────────────────────────────────────────────────────────────────────

// ── THEME ─────────────────────────────────────────────────────────────────────
function setTheme(t, btn) {
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('life-os-theme', t);
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
}

// Apply saved theme on load
(function () {
    const saved = localStorage.getItem('life-os-theme') || 'mixed';
    document.documentElement.setAttribute('data-theme', saved);
    document.querySelectorAll('.theme-btn').forEach(b => {
        b.classList.toggle('active', b.textContent.trim().toLowerCase() === saved);
    });
})();

// ── SIDEBAR COLLAPSE ──────────────────────────────────────────────────────────
let sidebarCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';

function applyCollapsed() {
    const layout = document.getElementById('layout');
    if (!layout) return;
    layout.classList.toggle('sidebar-collapsed', sidebarCollapsed);
    const icon = document.getElementById('collapse-icon');
    if (icon) icon.textContent = sidebarCollapsed ? '▶' : '◀';
}

function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed;
    localStorage.setItem('sidebar-collapsed', sidebarCollapsed);
    applyCollapsed();
}

// Apply on load — desktop only
if (window.innerWidth > 768) {
    applyCollapsed();
}

// ── MOBILE SIDEBAR ────────────────────────────────────────────────────────────
function openMobileSidebar() {
    const sidebar = document.getElementById('sidebar');
    const bd = document.getElementById('sidebar-backdrop');
    if (sidebar) sidebar.classList.add('mobile-open');
    if (bd) bd.classList.add('visible');
    document.body.style.overflow = 'hidden';
}

function closeMobileSidebar() {
    const sidebar = document.getElementById('sidebar');
    const bd = document.getElementById('sidebar-backdrop');
    if (sidebar) sidebar.classList.remove('mobile-open');
    if (bd) bd.classList.remove('visible');
    document.body.style.overflow = '';
}

// Close on Escape
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        closeMobileSidebar();
    }
});

// ── TOAST ─────────────────────────────────────────────────────────────────────
let _toastTimer;

function showToast(msg, isError = false) {
    const t = document.getElementById('toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.toggle('error', isError);
    t.classList.add('show');
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}

// Auto-show toast from HTMX partial data-toast attribute
document.addEventListener('htmx:afterSwap', function (evt) {
    const el = evt.target.closest('[data-toast]') || evt.target;
    const toast = el && el.dataset && el.dataset.toast;
    if (toast) showToast(toast);
});
