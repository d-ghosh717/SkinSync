// ── DOM ─────────────────────────────────────────────────────────────────────
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const uploadPreview = document.getElementById('upload-preview');
const changeHint = document.getElementById('change-hint');
const analyzeBtn = document.getElementById('analyze-btn');
const loader = document.getElementById('loader');
const resultsCol = document.getElementById('results-col');
const qualityWarns = document.getElementById('quality-warns');

// Camera elements
const startCamBtn = document.getElementById('start-cam-btn');
const stopCamBtn = document.getElementById('stop-cam-btn');
const captureBtn = document.getElementById('capture-btn');
const retakeBtn = document.getElementById('retake-btn');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const camPreview = document.getElementById('cam-preview');
const camPlaceholder = document.getElementById('cam-placeholder');
const camLive = document.getElementById('cam-live');
const camCaptured = document.getElementById('cam-captured');
const camStatus = document.getElementById('cam-status');
const camDot = document.getElementById('cam-dot');
const camStatusText = document.getElementById('cam-status-text');
const camWarnings = document.getElementById('cam-warnings');

// ── State ────────────────────────────────────────────────────────────────────
let imageBlob = null;
let camStream = null;
let qualTimer = null;

const show = el => el.classList.remove('hidden');
const hide = el => el.classList.add('hidden');

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(id) {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.remove('active');
        p.classList.add('hidden');
    });
    document.getElementById('tab-btn-' + id).classList.add('active');
    const pane = document.getElementById('tab-' + id);
    pane.classList.add('active');
    pane.classList.remove('hidden');

    if (id === 'upload') {
        stopCam();               // stop camera if switching away
        show(analyzeBtn);        // show analyze button for upload
    } else {
        hide(analyzeBtn);        // camera tab has its own "Capture & Analyze" button
    }
}


// ── Camera Stage 1 → 2: Turn On ──────────────────────────────────────────────
startCamBtn.addEventListener('click', async () => {
    setStatus('grey', 'Requesting camera access…');
    try {
        camStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
        });

        // Show the live view container FIRST so the <video> element renders in the DOM
        hide(camPlaceholder);
        show(camLive);

        // Then attach stream and play
        webcam.srcObject = camStream;
        webcam.play().catch(() => { });

        captureBtn.disabled = false;
        // Start quality monitor after a short delay to let first frame arrive
        setTimeout(startQualityMonitor, 600);

    } catch (e) {
        console.error('Camera error:', e);
        // Show placeholder again with error message
        show(camPlaceholder);
        hide(camLive);
        const ph = document.getElementById('cam-placeholder');
        ph.querySelector('.cam-ph-title').textContent = '📷 Camera access denied';
        ph.querySelector('.small').textContent = 'Please allow camera permission in your browser and refresh the page.';
        document.getElementById('start-cam-btn').disabled = true;
    }
});


// ── Camera Stage 2 → 1: Turn Off ─────────────────────────────────────────────
stopCamBtn.addEventListener('click', () => { imageBlob = null; stopCam(); });

function stopCam() {
    clearInterval(qualTimer);
    if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }

    // Reset UI back to stage 1
    hide(camLive);
    hide(camCaptured);
    hide(camWarnings);
    show(camPlaceholder);
    captureBtn.disabled = true;
    captureBtn.textContent = '📸 Capture & Analyze';
    // NOTE: imageBlob is NOT cleared here so auto-analyze after capture can still use it
}

// ── Camera capture (Stage 2 → 3) → auto analyze ─────────────────────────────
captureBtn.addEventListener('click', () => {
    if (!camStream) return;

    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext('2d');
    // Mirror to match the flipped video display
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(webcam, 0, 0);

    canvas.toBlob(blob => {
        imageBlob = blob;
        camPreview.src = URL.createObjectURL(blob);

        // Show stage 3
        stopCam();        // stop stream
        hide(camPlaceholder);
        show(camCaptured);

        // Auto-analyze immediately
        runAnalysis();
    }, 'image/jpeg', 0.92);
});

// ── Retake (Stage 3 → 1) ─────────────────────────────────────────────────────
retakeBtn.addEventListener('click', () => {
    hide(camCaptured);
    hide(resultsCol);
    hide(qualityWarns);
    show(camPlaceholder);
    imageBlob = null;
});

// ── Quality Monitor ───────────────────────────────────────────────────────────
function startQualityMonitor() {
    clearInterval(qualTimer);
    const qc = document.createElement('canvas');
    qc.width = 80; qc.height = 60;
    const qctx = qc.getContext('2d');

    qualTimer = setInterval(() => {
        if (!camStream || !webcam.videoWidth) return;

        qctx.drawImage(webcam, 0, 0, 80, 60);
        const data = qctx.getImageData(0, 0, 80, 60).data;

        // ── Brightness ─────────────────────────────────────────────────
        let total = 0, sumSq = 0;
        const px = data.length / 4;
        for (let i = 0; i < data.length; i += 4) {
            const lum = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
            total += lum;
            sumSq += lum * lum;
        }
        const brightness = total / px;

        // ── Blur (variance of luma = proxy for sharpness) ──────────────
        const lumMean = brightness;
        const variance = sumSq / px - lumMean * lumMean;
        const blurry = variance < 80;   // low variance → blurry/flat

        // ── Build warnings ─────────────────────────────────────────────
        const warns = [];
        if (brightness < 55) warns.push('Too dark – move into better lighting');
        if (brightness > 215) warns.push('Too bright – avoid direct flash or sunlight');
        if (blurry) warns.push('Image looks blurry – hold the device still');
        if (brightness > 60 && brightness < 215 && !blurry) {
            // All good – enable capture, clear warns
        }

        // ── Update status pill ─────────────────────────────────────────
        if (warns.length === 0) {
            setStatus('green', '✓ Good lighting – ready to capture');
            captureBtn.disabled = false;
        } else if (warns.some(w => w.startsWith('Too dark') || w.startsWith('Too bright'))) {
            setStatus('yellow', warns[0]);
            captureBtn.disabled = false;  // still allow capture, just warn
        } else {
            setStatus('yellow', warns[0]);
            captureBtn.disabled = false;
        }

        // ── Update warnings panel ──────────────────────────────────────
        if (warns.length) {
            camWarnings.innerHTML = warns.map(w => `<span class="cam-warn-row">${w}</span>`).join('');
            show(camWarnings);
        } else {
            hide(camWarnings);
        }
    }, 700);
}

function setStatus(color, text) {
    camDot.className = `cam-dot cam-dot-${color}`;
    camStatusText.textContent = text;
}

// ── Upload Tab ────────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
uploadPreview.addEventListener('click', () => fileInput.click());
changeHint.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; });
dropZone.addEventListener('dragleave', () => dropZone.style.borderColor = '');
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.style.borderColor = '';
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });

function loadFile(file) {
    if (!file.type.startsWith('image/')) { alert('Please select an image file.'); return; }
    imageBlob = file;
    const reader = new FileReader();
    reader.onload = ev => {
        uploadPreview.src = ev.target.result;
        show(uploadPreview);
        show(changeHint);
        hide(dropZone);
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Upload tab Analyze button
analyzeBtn.addEventListener('click', () => {
    if (!imageBlob) return;
    runAnalysis();
});

// ── Core Analysis ─────────────────────────────────────────────────────────────
async function runAnalysis() {
    show(loader);
    hide(resultsCol);
    hide(qualityWarns);
    analyzeBtn.disabled = true;

    try {
        const fd = new FormData();
        fd.append('image', imageBlob, 'photo.jpg');
        const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok || !data.success) throw new Error(data.error || 'Analysis failed.');

        if (data.quality_warnings && data.quality_warnings.length) {
            qualityWarns.innerHTML = data.quality_warnings.map(w => `<p>⚠️ ${w}</p>`).join('');
            show(qualityWarns);
        }
        renderResults(data);
        show(resultsCol);
        resultsCol.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        alert('⚠️  ' + err.message);
    } finally {
        hide(loader);
        analyzeBtn.disabled = false;
    }
}

// ── Render Results ─────────────────────────────────────────────────────────────
function renderResults(data) {
    const p = data.profile;

    document.getElementById('big-chip').style.background = p.hex_color;
    document.getElementById('res-hex').textContent = p.hex_color.toUpperCase();
    document.getElementById('res-rgb').textContent = `RGB(${p.r}, ${p.g}, ${p.b})`;
    document.getElementById('res-lum').textContent = `Luminance: ${p.luminance}`;

    const toneBadge = document.getElementById('res-tone-badge');
    toneBadge.textContent = p.tone_label;
    toneBadge.className = 'badge badge-tone';
    document.getElementById('res-tone-type').textContent = p.tone_type;
    document.getElementById('res-tone-desc').textContent = p.tone_desc;
    const tonePos = { Fair: 8, Medium: 33, Tan: 60, Deep: 88 };
    document.getElementById('tone-marker').style.left = (tonePos[p.tone] || 50) + '%';

    const utBadge = document.getElementById('res-ut-badge');
    utBadge.textContent = p.undertone;
    utBadge.className = 'badge ' + ({ Warm: 'badge-warm', Cool: 'badge-cool', Neutral: 'badge-neutral' }[p.undertone] || 'badge-neutral');
    document.getElementById('res-ut-conf').textContent = `${p.confidence}% confidence`;
    const barsEl = document.getElementById('ut-bars');
    barsEl.innerHTML = '';
    ['Warm', 'Cool', 'Neutral'].forEach(k => {
        const pct = Math.round(((p.undertone_scores || {})[k] || 0) * 100);
        barsEl.insertAdjacentHTML('beforeend',
            `<div class="ut-row">
                <span class="ut-label">${k}</span>
                <div class="ut-track"><div class="ut-fill" style="width:${pct}%"></div></div>
                <span class="ut-pct">${pct}%</span>
            </div>`);
    });

    const txBadge = document.getElementById('res-tx-badge');
    txBadge.textContent = p.texture_label;
    txBadge.className = 'badge badge-warm';
    document.getElementById('res-tx-conf').textContent = `${p.confidence}% confidence`;
    document.getElementById('res-tx-tags').innerHTML = (p.texture_tags || []).map(t => `<li>${t}</li>`).join('');
    const fb = document.getElementById('finish-box');
    if (p.finish_rec) {
        document.getElementById('finish-rec').textContent = `Recommended finish: ${p.finish_rec}`;
        document.getElementById('foundation-types').textContent = `Best types: ${p.foundation_types}`;
        show(fb);
    } else { hide(fb); }

    renderFoundations(data.recommendations || [], p.hex_color);
}

function renderFoundations(recs, skinHex) {
    const el = document.getElementById('found-list');
    el.innerHTML = '';
    if (!recs.length) { el.innerHTML = '<p class="muted small">No matching foundations found.</p>'; return; }

    const best = recs[0];
    el.insertAdjacentHTML('beforeend', `
        <div class="found-best">
            <div class="found-best-hdr">
                <div class="found-best-title"><span class="found-best-ico">⭐</span>Best Match</div>
                <span class="match-pill">${best.score}% Match</span>
            </div>
            <div class="found-color-row">
                <div class="found-swatch-group">
                    <div>
                        <div class="found-swatch" style="background:${skinHex}"></div>
                        <p class="swatch-label">Your Skin</p>
                    </div>
                    <span class="swatch-dash">—</span>
                    <div>
                        <div class="found-swatch" style="background:${best.hex}"></div>
                        <p class="swatch-label">Foundation</p>
                    </div>
                </div>
                <div class="found-meta">
                    <p class="found-brand">${best.brand}</p>
                    <p class="found-name">${best.type} Foundation</p>
                    <p class="found-shade-lbl">Shade: ${best.shade}</p>
                    <div class="found-tags">
                        <span class="found-tag">${best.type.toLowerCase()}</span>
                        <span class="found-tag">${best.finish} finish</span>
                        <span class="found-tag">${best.coverage} coverage</span>
                        <span class="found-tag ut-tag">${best.undertone} undertone</span>
                    </div>
                    <p class="found-desc">${best.description}</p>
                    <ul class="found-checks">${(best.bullets || []).map(b => `<li>${b}</li>`).join('')}</ul>
                    <div class="found-footer">
                        <span class="found-price">$${best.price}</span>
                        <span class="found-skintype">For ${best.skin_type}</span>
                    </div>
                </div>
            </div>
        </div>
    `);

    const alts = recs.slice(1);
    if (alts.length) {
        const row = document.createElement('div');
        row.className = 'found-alt-row';
        alts.forEach((m, i) => row.insertAdjacentHTML('beforeend', `
            <div class="found-alt">
                <div class="found-alt-hdr">
                    <span class="found-alt-rank">${i + 2}</span>
                    <span class="found-alt-pct">${m.score}%</span>
                </div>
                <div class="found-alt-chip" style="background:${m.hex}"></div>
                <p class="found-alt-name">${m.shade}</p>
                <p class="found-alt-brand">${m.brand}</p>
            </div>
        `));
        el.appendChild(row);
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
// Start on Upload tab: show analyze button
show(analyzeBtn);
