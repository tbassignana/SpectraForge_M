/**
 * SpectraForge UI Application
 * Platform-agnostic web interface for the ray tracer
 */

// State
let isRendering = false;
let pollInterval = null;
let currentImageData = null;

// Quality presets (tuned for pure Python performance)
const QUALITY_PRESETS = {
    preview: { width: 160, height: 120, samples: 4, maxDepth: 5 },   // ~10-20 sec
    draft:   { width: 240, height: 180, samples: 8, maxDepth: 8 },   // ~1-2 min
    medium:  { width: 320, height: 240, samples: 16, maxDepth: 10 }, // ~5-10 min
    high:    { width: 480, height: 360, samples: 32, maxDepth: 15 }  // ~20-40 min
};

// DOM Elements
const elements = {
    // Scene
    scenePreset: document.getElementById('scenePreset'),

    // Render settings
    qualityPreset: document.getElementById('qualityPreset'),
    width: document.getElementById('width'),
    height: document.getElementById('height'),
    samples: document.getElementById('samples'),
    maxDepth: document.getElementById('maxDepth'),
    threads: document.getElementById('threads'),

    // Camera
    camPosX: document.getElementById('camPosX'),
    camPosY: document.getElementById('camPosY'),
    camPosZ: document.getElementById('camPosZ'),
    camLookX: document.getElementById('camLookX'),
    camLookY: document.getElementById('camLookY'),
    camLookZ: document.getElementById('camLookZ'),
    fov: document.getElementById('fov'),
    fovValue: document.getElementById('fovValue'),
    aperture: document.getElementById('aperture'),
    apertureValue: document.getElementById('apertureValue'),
    focusDist: document.getElementById('focusDist'),

    // Post-processing
    denoise: document.getElementById('denoise'),
    toneMapping: document.getElementById('toneMapping'),
    bloom: document.getElementById('bloom'),
    exposure: document.getElementById('exposure'),
    exposureValue: document.getElementById('exposureValue'),

    // UI
    renderBtn: document.getElementById('renderBtn'),
    cancelBtn: document.getElementById('cancelBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    progressBar: document.getElementById('progressBar'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    placeholder: document.getElementById('placeholder'),
    renderImage: document.getElementById('renderImage'),
    renderTime: document.getElementById('renderTime'),
    renderSize: document.getElementById('renderSize'),
    statusText: document.getElementById('statusText'),
    platformInfo: document.getElementById('platformInfo'),
};

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
    // Bind events
    elements.renderBtn.addEventListener('click', startRender);
    elements.cancelBtn.addEventListener('click', cancelRender);
    elements.downloadBtn.addEventListener('click', downloadImage);

    // Bind slider value displays
    elements.fov.addEventListener('input', () => {
        elements.fovValue.textContent = elements.fov.value;
    });
    elements.aperture.addEventListener('input', () => {
        elements.apertureValue.textContent = parseFloat(elements.aperture.value).toFixed(2);
    });
    elements.exposure.addEventListener('input', () => {
        elements.exposureValue.textContent = parseFloat(elements.exposure.value).toFixed(1);
    });

    // Scene preset changes
    elements.scenePreset.addEventListener('change', updateCameraForScene);

    // Quality preset changes
    elements.qualityPreset.addEventListener('change', updateQualityPreset);

    // Load platform info
    fetchPlatformInfo();

    console.log('SpectraForge UI initialized');
}

function updateQualityPreset() {
    const preset = QUALITY_PRESETS[elements.qualityPreset.value];
    if (preset) {
        elements.width.value = preset.width;
        elements.height.value = preset.height;
        elements.samples.value = preset.samples;
        elements.maxDepth.value = preset.maxDepth;
    }
}

function updateCameraForScene() {
    const scene = elements.scenePreset.value;

    if (scene === 'demo') {
        setCameraValues(13, 2, 3, 0, 0, 0, 20, 0.1, 10);
    } else if (scene === 'cornell') {
        setCameraValues(0, 5, 15, 0, 5, 0, 40, 0, 15);
    } else {
        setCameraValues(5, 2, 5, 0, 0, 0, 30, 0, 5);
    }
}

function setCameraValues(px, py, pz, lx, ly, lz, fov, aperture, focusDist) {
    elements.camPosX.value = px;
    elements.camPosY.value = py;
    elements.camPosZ.value = pz;
    elements.camLookX.value = lx;
    elements.camLookY.value = ly;
    elements.camLookZ.value = lz;
    elements.fov.value = fov;
    elements.fovValue.textContent = fov;
    elements.aperture.value = aperture;
    elements.apertureValue.textContent = aperture.toFixed(2);
    elements.focusDist.value = focusDist;
}

async function fetchPlatformInfo() {
    try {
        const response = await fetch('/api/platform');
        const info = await response.json();
        elements.platformInfo.textContent = `${info.system} | ${info.machine} | ${info.cpu_count} cores`;
    } catch (error) {
        console.error('Failed to fetch platform info:', error);
    }
}

function getConfig() {
    return {
        scene: elements.scenePreset.value,
        width: parseInt(elements.width.value),
        height: parseInt(elements.height.value),
        samples: parseInt(elements.samples.value),
        max_depth: parseInt(elements.maxDepth.value),
        threads: parseInt(elements.threads.value),
        camera: {
            look_from: [
                parseFloat(elements.camPosX.value),
                parseFloat(elements.camPosY.value),
                parseFloat(elements.camPosZ.value)
            ],
            look_at: [
                parseFloat(elements.camLookX.value),
                parseFloat(elements.camLookY.value),
                parseFloat(elements.camLookZ.value)
            ],
            vfov: parseFloat(elements.fov.value),
            aperture: parseFloat(elements.aperture.value),
            focus_dist: parseFloat(elements.focusDist.value)
        },
        post_processing: {
            denoise: elements.denoise.checked,
            tone_mapping: elements.toneMapping.value,
            bloom: elements.bloom.checked,
            exposure: parseFloat(elements.exposure.value)
        }
    };
}

async function startRender() {
    if (isRendering) return;

    const config = getConfig();

    try {
        const response = await fetch('/api/render', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            isRendering = true;
            updateUIForRendering(true);
            startPolling();
        }
    } catch (error) {
        console.error('Failed to start render:', error);
        setStatus('Error starting render', 'error');
    }
}

async function cancelRender() {
    try {
        await fetch('/api/cancel', { method: 'POST' });
        stopPolling();
        isRendering = false;
        updateUIForRendering(false);
        setStatus('Cancelled', 'warning');
    } catch (error) {
        console.error('Failed to cancel:', error);
    }
}

function startPolling() {
    pollInterval = setInterval(pollStatus, 250);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

async function pollStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        updateProgress(status);

        if (status.status === 'completed') {
            stopPolling();
            isRendering = false;
            updateUIForRendering(false);
            displayImage(status.image_data);

            const duration = (status.end_time - status.start_time).toFixed(1);
            elements.renderTime.textContent = `Time: ${duration}s`;
            elements.renderSize.textContent = `${status.settings.width}x${status.settings.height}`;
            setStatus('Complete', 'success');

        } else if (status.status === 'error' || status.status === 'cancelled') {
            stopPolling();
            isRendering = false;
            updateUIForRendering(false);
            setStatus(status.error || 'Cancelled', 'error');
        }
    } catch (error) {
        console.error('Poll error:', error);
    }
}

function updateProgress(status) {
    const progress = status.progress || 0;
    const percent = Math.round(progress * 100);

    elements.progressFill.style.width = `${percent}%`;

    if (status.status === 'running') {
        const elapsed = (Date.now() / 1000) - status.start_time;
        let eta = '';
        if (progress > 0.05) {
            const totalEstimated = elapsed / progress;
            const remaining = Math.max(0, totalEstimated - elapsed);
            eta = ` | ETA: ${formatTime(remaining)}`;
        }
        elements.progressText.textContent = `${percent}%${eta}`;
        setStatus('Rendering...', 'rendering');
    } else if (status.status === 'pending') {
        elements.progressText.textContent = 'Starting...';
    }
}

function formatTime(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function updateUIForRendering(rendering) {
    elements.renderBtn.disabled = rendering;
    elements.cancelBtn.disabled = !rendering;
    document.body.classList.toggle('rendering', rendering);

    if (rendering) {
        elements.progressFill.style.width = '0%';
        elements.progressText.textContent = 'Starting...';
    }
}

function displayImage(base64Data) {
    if (!base64Data) return;

    currentImageData = base64Data;
    elements.placeholder.style.display = 'none';
    elements.renderImage.src = `data:image/png;base64,${base64Data}`;
    elements.renderImage.style.display = 'block';
    elements.downloadBtn.disabled = false;
}

function downloadImage() {
    if (!currentImageData) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${currentImageData}`;
    link.download = `spectraforge_render_${Date.now()}.png`;
    link.click();
}

function setStatus(text, type) {
    elements.statusText.textContent = text;
    elements.statusText.className = '';

    if (type === 'error') {
        elements.statusText.style.color = '#ef4444';
    } else if (type === 'warning') {
        elements.statusText.style.color = '#fbbf24';
    } else if (type === 'success') {
        elements.statusText.style.color = '#4ade80';
    } else if (type === 'rendering') {
        elements.statusText.style.color = '#e94560';
    } else {
        elements.statusText.style.color = '#a0a0a0';
    }
}

// Panel toggle function
function togglePanel(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');

    content.classList.toggle('collapsed');
    icon.textContent = content.classList.contains('collapsed') ? '+' : '-';
}

// Make togglePanel available globally
window.togglePanel = togglePanel;
