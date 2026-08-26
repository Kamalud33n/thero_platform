/**
 * Rehabilitation AI System - Main Application JavaScript
 * Handles global functionality, WebSocket connections, and utilities
 */

// Global state
const AppState = {
    isCameraActive: false,
    isSessionActive: false,
    currentPatientId: null,
    wsConnection: null,
    sessionData: {
        startTime: null,
        repCount: 0,
        accuracy: 0,
        rom: 0,
        stability: 0
    }
};

// ============================================
// DOM Ready
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Remove loading overlay
    setTimeout(() => {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
            setTimeout(() => overlay.remove(), 500);
        }
    }, 500);
    
    // Setup event listeners
    setupEventListeners();
    
    // Check system health
    checkSystemHealth();
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Global keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Escape key to close modals
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal.active').forEach(modal => {
                modal.classList.remove('active');
            });
        }
    });
    
    // Auto-hide flash messages
    document.querySelectorAll('.flash-message').forEach(msg => {
        setTimeout(() => {
            msg.style.opacity = '0';
            setTimeout(() => msg.remove(), 300);
        }, 5000);
    });
}

// ============================================
// System Health Check
// ============================================
async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status !== 'healthy') {
            showToast('System health check failed', 'error');
        }
    } catch (error) {
        console.error('Health check error:', error);
    }
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'success', duration = 3000) {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    
    // Trigger show animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });
    
    // Auto-hide
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================
// Date/Time Utilities
// ============================================
function formatDate(date) {
    if (typeof date === 'string') {
        date = new Date(date);
    }
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatTime(date) {
    if (typeof date === 'string') {
        date = new Date(date);
    }
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDateTime(date) {
    return `${formatDate(date)} at ${formatTime(date)}`;
}

function formatDuration(seconds) {
    if (!seconds) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

// ============================================
// API Helpers
// ============================================
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.message || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        showToast(error.message || 'Something went wrong', 'error');
        throw error;
    }
}

// ============================================
// Patient Management
// ============================================
async function loadPatientSelect(selectId) {
    try {
        const patients = await apiRequest('/api/patients');
        const select = document.getElementById(selectId);
        if (!select) return;
        
        // Clear existing options
        select.innerHTML = '<option value="">Select Patient</option>';
        
        patients.forEach(patient => {
            const option = document.createElement('option');
            option.value = patient.id;
            option.textContent = patient.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading patients:', error);
    }
}

// ============================================
// Form Validation
// ============================================
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return true;
    
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('error');
            isValid = false;
        } else {
            input.classList.remove('error');
        }
    });
    
    if (!isValid) {
        showToast('Please fill in all required fields', 'error');
    }
    
    return isValid;
}

// ============================================
// Number Formatting
// ============================================
function formatPercentage(value) {
    return `${Math.round(value)}%`;
}

function formatAngle(value) {
    return `${Math.round(value)}°`;
}

function formatNumber(value, decimals = 1) {
    return Number(value).toFixed(decimals);
}

// ============================================
// Chart Color Palette
// ============================================
const ChartColors = {
    green: '#1D8A6D',
    greenLight: 'rgba(29, 138, 109, 0.2)',
    blue: '#3E6FD9',
    blueLight: 'rgba(33, 150, 243, 0.2)',
    orange: '#D68F35',
    orangeLight: 'rgba(255, 152, 0, 0.2)',
    red: '#D9503E',
    redLight: 'rgba(244, 67, 54, 0.2)',
    purple: '#8B5FC7',
    purpleLight: 'rgba(156, 39, 176, 0.2)',
    cyan: '#1AA6B7',
    cyanLight: 'rgba(0, 188, 212, 0.2)',
    grey: '#9E9E9E',
    greyLight: 'rgba(158, 158, 158, 0.2)'
};

// ============================================
// Export for use in other files
// ============================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AppState,
        showToast,
        formatDate,
        formatTime,
        formatDateTime,
        formatDuration,
        apiRequest,
        loadPatientSelect,
        validateForm,
        formatPercentage,
        formatAngle,
        formatNumber,
        ChartColors
    };
}