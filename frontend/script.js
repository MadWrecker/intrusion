// script.js
// Factory-Grade AI Face Recognition Dashboard Logic

const API_BASE = ''; // Same origin

async function authFetch(url, options = {}) {
    const token = localStorage.getItem('factoryGuardToken');
    if (token) {
        options.headers = options.headers || {};
        if (options.headers instanceof Headers) {
            options.headers.append('Authorization', `Bearer ${token}`);
        } else {
            options.headers['Authorization'] = `Bearer ${token}`;
        }
    }
    options.credentials = 'include';
    const res = await fetch(url, options);
    if (res.status === 401 && !url.includes('/login')) {
        logout();
        throw new Error("Session expired or unauthorized");
    }
    return res;
}

// Basic State
let currentEmployees = [];
let currentIntruders = [];
let currentLogs = [];
let alertInterval = null;
let healthInterval = null;
let attendanceInterval = null;

// --- Init & Auth ---
document.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('factoryGuardAuth') === 'true') {
        document.getElementById('login-overlay').classList.add('hidden');
        initDashboard();
    } else {
        document.getElementById('login-overlay').classList.remove('hidden');
    }
});

async function login(e) {
    if(e) e.preventDefault();
    const user = document.getElementById('username').value;
    const pass = document.getElementById('password').value;

    try {
        const res = await fetch(`${API_BASE}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        });
        if (res.ok) {
            const data = await res.json();
            localStorage.setItem('factoryGuardAuth', 'true');
            localStorage.setItem('factoryGuardToken', data.token);
            document.getElementById('login-overlay').classList.add('hidden');
            initDashboard();
        } else {
            document.getElementById('login-error').textContent = 'Invalid credentials';
        }
    } catch(err) {
        document.getElementById('login-error').textContent = 'Connection error';
    }
}

function logout() {
    clearInterval(alertInterval);
    clearInterval(healthInterval);
    clearInterval(attendanceInterval);
    localStorage.removeItem('factoryGuardAuth');
    localStorage.removeItem('factoryGuardToken');
    document.getElementById('login-overlay').classList.remove('hidden');
    document.getElementById('dashboard').classList.add('hidden');
    document.getElementById('password').value = '';
}

function initDashboard() {
    document.getElementById('dashboard').classList.remove('hidden');

    // Start video feed
    const img = document.getElementById('videoFeed');
    if(img) {
        img.src = `${API_BASE}/camera_feed`;
        img.onerror = () => { img.alt = "Camera feed offline or reconnecting..."; };
    }

    // Initial data loads
    fetchEmployees();
    fetchAttendance();
    fetchIntruders();
    fetchHealth();
    fetchTempPasses();

    // Start polling
    alertInterval = setInterval(pollAlerts, 2000);
    healthInterval = setInterval(fetchHealth, 10000);
    attendanceInterval = setInterval(fetchAttendance, 15000); // refresh attendance every 15s

    showSection('live');
}

// --- Navigation ---
function showSection(sectionId) {
    document.querySelectorAll('.page-section').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    document.getElementById(`section-${sectionId}`).classList.remove('hidden');
    document.getElementById(`nav-${sectionId}`).classList.add('active');

    if (sectionId === 'employees') fetchEmployees();
    if (sectionId === 'attendance') fetchAttendance();
    if (sectionId === 'intruders') fetchIntruders();
    if (sectionId === 'temppass') fetchTempPasses();
}

// --- API Calls & Renders ---

// 1. Employees
async function fetchEmployees() {
    try {
        const res = await authFetch(`${API_BASE}/employees`);
        if (res.status === 401) { logout(); return; }
        if (res.ok) {
            currentEmployees = await res.json();
            const stat = document.getElementById('stat-employees');
            if(stat) stat.textContent = currentEmployees.length;
            renderEmployees();
        }
    } catch(err) { console.error("Err fetching employees", err); }
}

function renderEmployees() {
    const container = document.getElementById('employee-grid');
    if(!container) return;
    if(currentEmployees.length === 0) {
        container.innerHTML = `<p class="text-slate-500 font-mono text-center col-span-full">No personnel registered in the facility database.</p>`;
        return;
    }
    container.innerHTML = currentEmployees.map(emp => `
        <div class="emp-card hover-card">
            <div class="avatar shadow-inner">${emp.name.charAt(0).toUpperCase()}</div>
            <h4 class="font-bold tracking-wide">${emp.name}</h4>
            <p class="font-mono">ID: ${emp.employee_id} <br> Dept: <span class="text-slate-300 font-sans">${emp.department || 'N/A'}</span> <br> Phone: <span class="text-slate-300 font-sans">${emp.phone || 'N/A'}</span></p>
            <button class="btn-danger mt-2" onclick="deleteEmployee('${emp.employee_id}')">Revoke Access</button>
        </div>
    `).join('');
}

async function addEmployee(e) {
    e.preventDefault();
    const id = document.getElementById('emp-id').value;
    const name = document.getElementById('emp-name').value;
    const dept = document.getElementById('emp-dept').value;
    const phone = document.getElementById('emp-phone').value;
    const files = document.getElementById('emp-images').files;
    const submitBtn = e.target.querySelector('button[type="submit"]');

    if (!files.length) { alert("Please provide at least one image"); return; }

    const fd = new FormData();
    fd.append('employee_id', id);
    fd.append('name', name);
    fd.append('department', dept);
    fd.append('phone', phone);
    for(let i=0; i<files.length; i++) {
        fd.append('images', files[i]);
    }

    const originalText = submitBtn.innerText;
    submitBtn.innerText = "Processing Details... Using AI Model...";
    submitBtn.disabled = true;

    try {
        const res = await authFetch(`${API_BASE}/add_employee`, { method: 'POST', body: fd });
        if (res.ok) {
            alert("Employee added successfully!");
            e.target.reset();
            document.getElementById('file-label').innerText = 'Browse files';
            fetchEmployees();
            document.getElementById('add-emp-modal').classList.add('hidden');
        } else {
            const data = await res.json();
            alert("Error: " + (data.detail || "Failed to add"));
        }
    } catch(err) { 
        alert("Network error: " + err.message); 
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
    }
}

async function deleteEmployee(empId) {
    if(!confirm("Are you sure you want to revoke this personnel's access and delete their biometrics?")) return;
    try {
        const res = await authFetch(`${API_BASE}/employee/${empId}`, { method: 'DELETE' });
        if(res.ok) fetchEmployees();
        else alert("Failed to revoke access.");
    } catch(err) { console.error(err); }
}

// 2. Attendance
async function fetchAttendance() {
    try {
        const res = await authFetch(`${API_BASE}/attendance`);
        if (res.status === 401) { logout(); return; }
        if(res.ok) {
            const records = await res.json();
            
            // Calculate stats for today (Local timezone safe)
            const dateObj = new Date();
            const year = dateObj.getFullYear();
            const month = String(dateObj.getMonth() + 1).padStart(2, '0');
            const day = String(dateObj.getDate()).padStart(2, '0');
            const today = `${year}-${month}-${day}`;
            const todayRecords = records.filter(r => r.date === today);
            const presentCount = todayRecords.filter(r => r.status === 'PRESENT' || r.status === 'HALF DAY' || r.status === 'LATE').length;
            const lateCount = todayRecords.filter(r => r.status === 'HALF DAY' || r.status === 'LATE').length; 
            
            const statPresent = document.getElementById('stat-present');
            if(statPresent) statPresent.textContent = presentCount;
            
            const statLate = document.getElementById('stat-late');
            if(statLate) statLate.textContent = lateCount;

            const tbody = document.getElementById('attendance-tbody');
            if(!tbody) return;
            if(records.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-slate-500 font-mono">No ledger entries found.</td></tr>`;
                return;
            }
            tbody.innerHTML = records.map(r => {
                let badgeClass = 'status-present';
                if(r.status === 'HALF DAY') badgeClass = 'status-halfday';
                if(r.status === 'LATE') badgeClass = 'status-late';
                if(r.status === 'LEAVE') badgeClass = 'status-leave';

                return `<tr class="hover:bg-factory-800 transition-colors">
                    <td class="p-4 text-sm font-mono text-slate-300">${r.date}</td>
                    <td class="p-4 text-sm">
                        <div class="flex items-center gap-3">
                            ${r.image_path ? `<img src="${API_BASE}/${r.image_path}" class="w-10 h-10 rounded-full object-cover border border-factory-600 cursor-pointer hover:border-factory-accent transition-colors" onclick="openModal(this.src)" onerror="this.src=''; this.className='hidden'">` : `<div class="w-10 h-10 rounded-full bg-factory-900 border border-factory-600 flex items-center justify-center text-xs text-slate-500">N/A</div>`}
                            <div>
                                <div class="font-bold text-slate-100">${r.employee_name}</div>
                                <div class="text-xs text-slate-500 font-mono">${r.employee_id}</div>
                            </div>
                        </div>
                    </td>
                    <td class="p-4 text-sm font-mono text-center text-slate-300">${r.check_in_time || '-'}</td>
                    <td class="p-4 text-sm font-mono text-center text-slate-300">${r.check_out_time || '-'}</td>
                    <td class="p-4 text-sm text-right"><span class="badge ${badgeClass} shadow-sm">${r.status}</span></td>
                </tr>`;
            }).join('');
        }
    } catch(err) { console.error("Err fetching attendance", err); }
}

// 3. Intruders & Logs
async function fetchIntruders() {
    try {
        const res = await authFetch(`${API_BASE}/intruder_logs`);
        if (res.status === 401) { logout(); return; }
        if(res.ok) {
            const logs = await res.json();
            const stat = document.getElementById('stat-intruders');
            if(stat) stat.textContent = logs.length;
            
            const badge = document.getElementById('nav-badge-intruders');
            if(badge) badge.textContent = logs.length;
            
            const grid = document.getElementById('intruder-grid');
            if(!grid) return;
            if(logs.length === 0) {
                grid.innerHTML = `<p class="text-slate-500 font-mono col-span-full text-center">No unauthorized access detected.</p>`;
                return;
            }
            grid.innerHTML = logs.map(log => {
                const imgPath = log.image_path ? log.image_path.split(/[\\/]/).pop() : '';
                return `
                <div class="intruder-card hover-card relative group">
                    <img src="${API_BASE}/intruder_detected/${imgPath}" onerror="this.src=''" onclick="openModal(this.src)" class="border-b border-factory-600">
                    <button onclick="deleteIntruderLog(${log.id})" class="absolute top-2 right-2 bg-factory-900/80 hover:bg-factory-danger text-slate-400 hover:text-white p-2 rounded-lg backdrop-blur opacity-0 group-hover:opacity-100 transition-all border border-factory-600 hover:border-factory-danger shadow-lg" title="Delete Log">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                    </button>
                    <div class="intruder-details bg-factory-800">
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <p class="text-xs font-mono text-slate-500 uppercase tracking-wide">Event Time</p>
                                <p class="text-sm font-bold text-slate-200">${log.date} at ${log.time}</p>
                            </div>
                            <span class="badge status-leave shadow-sm shadow-factory-danger/20">${log.status}</span>
                        </div>
                        <p class="text-xs font-mono text-slate-400 border-t border-factory-600/50 pt-2 mt-2">
                            <span class="text-slate-500 uppercase">Sensory Node:</span> ${log.camera_location}
                        </p>
                    </div>
                </div>
            `;}).join('');
        }
    } catch(err) { console.error("Err fetching intruders", err); }
}

async function deleteIntruderLog(logId) {
    if(!confirm("Are you sure you want to delete this intruder log and its associated image permanently?")) return;
    try {
        const res = await authFetch(`${API_BASE}/intruder_logs/${logId}`, { method: 'DELETE' });
        if(res.ok) fetchIntruders();
        else alert("Failed to delete intruder log.");
    } catch(err) { console.error(err); }
}

// 4. Temporary Passes
async function fetchTempPasses() {
    try {
        const res = await authFetch(`${API_BASE}/temp-passes`); // Assumes backend supports it
        if (res.status === 401) { logout(); return; }
        if (res.ok) {
            const passes = await res.json();
            const tbody = document.getElementById('temppass-tbody');
            if(!tbody) return;
            if(passes.length === 0) {
                tbody.innerHTML = `<tr><td colspan="4" class="p-4 text-center text-slate-500 font-mono">No temporary passes requested.</td></tr>`;
                return;
            }
            tbody.innerHTML = passes.map(p => {
                let badgeClass = 'status-leave'; // pending
                if(p.status === 'approved') badgeClass = 'status-present';
                if(p.status === 'rejected') badgeClass = 'status-late';
                
                return `<tr class="hover:bg-factory-800 transition-colors">
                    <td class="p-4 text-sm font-bold text-slate-100">${p.name}</td>
                    <td class="p-4 text-sm font-mono text-slate-300">${p.purpose || '-'}</td>
                    <td class="p-4 text-sm text-center"><span class="badge ${badgeClass} shadow-sm px-2 uppercase text-[10px]">${p.status}</span></td>
                    <td class="p-4 text-sm text-right flex justify-end gap-2">
                        ${p.status === 'pending' ? `
                            <button onclick="updateTempPassStatus(${p.id}, 'approved')" class="bg-factory-success/20 text-factory-success hover:bg-factory-success hover:text-white px-3 py-1 rounded text-xs font-bold transition-colors">Approve</button>
                            <button onclick="updateTempPassStatus(${p.id}, 'rejected')" class="bg-factory-danger/20 text-factory-danger hover:bg-factory-danger hover:text-white px-3 py-1 rounded text-xs font-bold transition-colors">Reject</button>
                        ` : `
                            <button onclick="deleteTempPass(${p.id})" class="text-slate-500 hover:text-factory-danger px-3 py-1 text-xs transition-colors">Delete</button>
                        `}
                    </td>
                </tr>`;
            }).join('');
        }
    } catch(err) { console.error("Err fetching temp passes", err); }
}

async function requestTempPass(e) {
    e.preventDefault();
    const name = document.getElementById('temppass-name').value;
    const purpose = document.getElementById('temppass-purpose').value;
    const submitBtn = e.target.querySelector('button[type="submit"]');
    
    // Note: To match backend schema exactly: name, purpose
    const payload = { name, purpose, image: "" };
    
    submitBtn.disabled = true;
    submitBtn.innerText = "Submitting...";
    try {
        const res = await authFetch(`${API_BASE}/temp-passes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if(res.ok) {
            alert("Temporary pass submitted!");
            e.target.reset();
            document.getElementById('add-temppass-modal').classList.add('hidden');
            fetchTempPasses();
        } else {
            alert("Server returned error when requesting pass.");
        }
    } catch(err) { alert("Network Error"); }
    finally {
        submitBtn.disabled = false;
        submitBtn.innerText = "Submit Request";
    }
}

async function updateTempPassStatus(id, newStatus) {
    try {
        const res = await authFetch(`${API_BASE}/temp-passes/${id}/status`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus })
        });
        if(res.ok) fetchTempPasses();
        else alert("Update failed");
    } catch(err) { alert("Network Error"); }
}

async function deleteTempPass(id) {
    if(!confirm("Delete this temporary pass?")) return;
    try {
        const res = await authFetch(`${API_BASE}/temp-passes/${id}`, { method: 'DELETE' });
        if(res.ok) fetchTempPasses();
        else alert("Delete failed");
    } catch(err) { alert("Network Error"); }
}

// 5. Alerts Polling (Live feed)
let lastAlertId = 0;
async function pollAlerts() {
    try {
        const res = await authFetch(`${API_BASE}/alerts`);
        if (res.status === 401) { logout(); return; }
        if(res.ok) {
            const alerts = await res.json();
            const list = document.getElementById('live-alerts-list');
            if(!list) return;
            
            if(alerts.length === 0) {
                list.innerHTML = `<p class="text-center text-slate-500 text-sm mt-10 font-mono">No active threats detected.</p>`;
                return;
            }

            const maxId = Math.max(...alerts.map(a => a.id));
            if (lastAlertId !== 0 && maxId > lastAlertId) {
                playAlertBeep();
            }
            lastAlertId = maxId;

            list.innerHTML = alerts.map(a => {
                const imgPath = a.image_path ? a.image_path.split(/[\\/]/).pop() : '';
                return `
                <div class="alert-item animate-fade-in">
                    <img src="${API_BASE}/intruder_detected/${imgPath}" onerror="this.style.display='none'" onclick="openModal(this.src)">
                    <div class="alert-info">
                        <p class="text-factory-danger flex items-center gap-2">
                          <span class="w-1.5 h-1.5 rounded-full bg-factory-danger animate-pulse"></span>
                          THREAT DETECTED
                        </p>
                        <small>${a.time} • CAM_01</small>
                    </div>
                </div>
            `;}).join('');

        }
    } catch(err) {} 
}

function playAlertBeep() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 800; // Hz
        osc.type = "square";
        gain.gain.value = 0.5;
        osc.start();
        setTimeout(() => { osc.stop(); }, 500);
    } catch(e) {}
}

// 5. System Health
async function fetchHealth() {
    try {
        const res = await authFetch(`${API_BASE}/system_health`);
        if (res.status === 401) { logout(); return; }
        if(res.ok) {
            const h = await res.json();
            const cpuEl = document.getElementById('health-cpu');
            const memEl = document.getElementById('health-mem');
            const diskEl = document.getElementById('health-disk');
            const dbEl = document.getElementById('health-db');

            // Add camera status at top if it exists
            const camStatEl = document.getElementById('stat-cam-status');
            if(camStatEl) {
                camStatEl.textContent = h.camera.toUpperCase();
                camStatEl.parentElement.querySelector('div').className = 
                    h.camera === 'Connected' 
                    ? 'w-3 h-3 bg-factory-success rounded-full shadow-[0_0_10px_rgba(16,185,129,0.8)]'
                    : 'w-3 h-3 bg-factory-warning rounded-full shadow-[0_0_10px_rgba(245,158,11,0.8)] animate-pulse';
            }

            if(cpuEl) cpuEl.textContent = h.cpu_percent + '%';
            if(memEl) memEl.textContent = h.memory_percent + '%';
            if(diskEl) {
                diskEl.textContent = h.disk_space_percent + '% (' + h.disk_status + ')';
                diskEl.className = h.disk_space_percent > 90 ? 'text-2xl font-mono text-factory-danger font-bold' : 'text-2xl font-mono text-white';
            }
            if(dbEl) {
                dbEl.textContent = h.database;
                dbEl.className = h.database === 'Connected' ? 'text-2xl font-mono text-factory-success' : 'text-2xl font-mono text-factory-danger font-bold';
            }
        }
    } catch(err) {}
}

// 6. Live Clock
setInterval(() => {
    const clock = document.getElementById('live-clock');
    if(clock) {
        clock.textContent = new Date().toLocaleTimeString('en-US', {hour12: false});
    }
}, 1000);

// --- Image Modal ---
function openModal(src) {
    document.getElementById('modalImage').src = src;
    document.getElementById('imageModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
}

// Close modal if clicked outside image
window.onclick = function(event) {
    if (event.target == document.getElementById('imageModal')) {
        closeModal();
    }
}
