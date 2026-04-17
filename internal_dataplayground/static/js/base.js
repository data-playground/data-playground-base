// static/js/base.js
// ─────────────────────────────────────────────────────────────────────────────
// Shared JS for all Life OS pages.
// Provides: setTheme(), toggleSidebar(), showToast(),
//           openMobileSidebar(), closeMobileSidebar()
// Loaded by base.html at the bottom of every page.
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
    document.getElementById('layout').classList.toggle('sidebar-collapsed', sidebarCollapsed);
    const icon = document.getElementById('collapse-icon');
    if (icon) icon.textContent = sidebarCollapsed ? '▶' : '◀';
}

function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed;
    localStorage.setItem('sidebar-collapsed', sidebarCollapsed);
    applyCollapsed();
}

// Apply on load (desktop only — mobile uses slide-in pattern)
if (window.innerWidth > 768) applyCollapsed();

// ── MOBILE SIDEBAR ────────────────────────────────────────────────────────────
function openMobileSidebar() {
    document.getElementById('sidebar').classList.add('mobile-open');
    const bd = document.getElementById('sidebar-backdrop');
    if (bd) bd.classList.add('visible');
}

function closeMobileSidebar() {
    document.getElementById('sidebar').classList.remove('mobile-open');
    const bd = document.getElementById('sidebar-backdrop');
    if (bd) bd.classList.remove('visible');
}

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
    const drawer = evt.target.closest('[data-toast]') || evt.target;
    const toast = drawer && drawer.dataset && drawer.dataset.toast;
    if (toast) showToast(toast);
});
