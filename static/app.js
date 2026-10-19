(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pad = n => String(Math.floor(n)).padStart(2, '0');
const mmss = s => `${pad(s / 60)}:${pad(s % 60)}`;
const store = {
  get(k, d) { try { const v = localStorage.getItem('fbc_' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('fbc_' + k, JSON.stringify(v)); } catch { /* private window: fine */ } },
};

const COMMANDS = [
  { id: 'start_recording', label: 'Record', say: '“record”' },
  { id: 'scan_license', label: 'Scan license', say: '' },
  { id: 'scan_cr', label: 'Scan CR', say: '' },
  { id: 'capture', label: 'Capture', say: '' },
  { id: 'end_recording', label: 'End case', say: '“case done”' },
];
const STEPS = [
  { title: "Driver's Information", sub: "Scan the driver's license" },
  { title: 'Vehicle Information', sub: 'Scan the CR and photograph the plate' },
  { title: 'Violation Offense Tier', sub: 'Fill in the details to create a violation ticket' },
  { title: 'Preview', sub: 'Review the case, the video and the transcript' },
];

const S = {
  step: 3, ws: null, wsReady: false, hello: null,
  live: false, replay: false, stream: null, ctx: null, node: null, outBuf: [], sent: 0, carry: new Float32Array(0), phase: 0,
  recorder: null, chunks: [], vid: null, vidN: 0, rotateTimer: null,
  awakeUntil: 0, awakeTimer: null, hearing: false,
  caseActive: false, sid: null, caseStartWall: 0, caseEnded: null, controlNo: newControl(),
  violLive: {}, violManual: [], dismissed: new Set(), tags: [], images: { license: null, cr: null, plates: [] }, scanAt: {},
  form: { location: '', date: new Date().toISOString().slice(0, 10), time: new Date().toTimeString().slice(0, 5), actions: {} },
  offenses: [], finals: {}, finalLines: [], progress: null, finalErr: null, videoURL: null, counters: { queued: 0, ignored: 0, lines: 0, dropped: 0 },
  debug: store.get('debug', false), follow: store.get('follow', true), wakeMode: store.get('wakeMode2', 'audio'), beepOn: store.get('beep', true), fastCmd: store.get('fastCmd', true), muteUntil: 0, beepCtx: null, join: store.get('join', true), vadOn: store.get('vadOn', 0.35),
  mic: store.get('mic', { echo: false, noise: false, agc: false }), dbg: [],
};
function newControl() { return 'CL-' + String(Math.floor(Math.random() * 1e7)).padStart(7, '0'); }

/* ------------------------------------------------------------------------------------------------ toasts */
function toast(html, cls = '', ms = 4200) {
  const t = document.createElement('div');
  t.className = 'toast ' + cls;
  t.innerHTML = html;
  $('#toasts').appendChild(t);
  setTimeout(() => t.remove(), ms);
}

/* ------------------------------------------------------------------------------------------------ websocket */
function connect() {
  return new Promise(res => {
    if (S.ws && S.ws.readyState <= 1) { if (S.wsReady) res(); else S.ws.addEventListener('open', () => res(), { once: true }); return; }
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    S.ws = ws;
    ws.onopen = () => { S.wsReady = true; res(); };
    ws.onmessage = m => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    ws.onclose = () => { S.wsReady = false; if (S.live || S.replay) toast('Connection to the app server was lost. Reconnecting…', 'err'); setTimeout(() => { if (S.live) connect(); }, 2000); };
    ws.onerror = () => { };
  });
}
const send = o => { if (S.ws && S.wsReady) S.ws.send(JSON.stringify(o)); };

/* ------------------------------------------------------------------------------------------------ events from the server */
function onEvent(ev) {
  switch (ev.type) {
    case 'hello': S.hello = ev; send({ type: 'settings', WAKE_MODE: S.wakeMode, JOIN_LINES: S.join, ON: S.vadOn, FAST_CMD: S.fastCmd }); renderTools(); break;
    case 'vad': S.hearing = ev.speech; if (!ev.speech) flushWakeBeep(); setPill(); break;
    case 'wake': wake(ev); break;
    case 'awake_end': beep('timeout'); if (!S.caseActive) dropCamera(); S.awakeUntil = 0; setPill(); tagline('Nothing heard after “Hey Jarvis”: back to idle. Say it again, or press Record.'); break;
    case 'segment': if (ev.action === 'ignored') S.counters.ignored++; else S.counters.queued++; if (S.toolsOpen) renderTools(); break;
    case 'line': onLine(ev); break;
    case 'command': onCommand(ev); break;
    case 'violations': onViolations(ev); break;
    case 'case': ev.state === 'started' ? onCaseStart(ev) : onCaseEnd(ev); break;
    case 'final_start': if (ev.sid === S.sid) { S.progress = { done: 0, total: 0, source: ev.source }; S.finalLines = []; S.finalErr = null; renderIf(4); syncNext(); } break;
    case 'final_progress': if (ev.sid === S.sid) { S.progress = { done: ev.done, total: ev.total, source: ev.source }; renderIf(4); } break;
    case 'final_line': if (ev.sid === S.sid) { S.finalLines.push(ev.line); renderIf(4); } break;
    case 'final_done': if (ev.sid === S.sid) { S.finals[ev.source] = ev.report; S.progress = null; renderIf(4); syncNext(); toast('Transcript ready (' + ev.report.lines.length + ' lines, ' + ev.report.seconds + ' s)'); } break;
    case 'final_error': if (ev.sid === S.sid) { S.finalErr = ev.message; S.progress = null; renderIf(4); syncNext(); toast('After-stop pass failed: ' + esc(ev.message), 'err'); } break;
    case 'error': toast('Server: ' + esc(ev.message), 'err'); break;
  }
}
const renderIf = n => { if (S.step === n) renderStep(); };

function onLine(ev) {
  const t = ev.timing || {};
  if (ev.dropped) S.counters.dropped++; else S.counters.lines++;
  if (S.debug) {
    const txt = ev.text || (ev.heard ? `(idle, ignored) ${ev.heard}` : `(${ev.dropped})`);
    S.dbg.push(`${ev.time}  ${txt}\n    wait ${t.wait_s}s + queue ${t.queue_s}s + whisper ${t.whisper_s}s = ${t.total_s}s${t.merged > 1 ? ' (merged ' + t.merged + ' lines)' : ''}${t.late ? ' LATE' : ''}`);
    S.dbg = S.dbg.slice(-40);
  }
  if (S.toolsOpen) renderTools();
}

/* Short sounds so the enforcer knows what the system did. They are the only feedback: no beep after "Jarvis" = it was not heard.
   While a beep plays the microphone stream is sent as silence, so the system never hears its own beep. */
const BEEPS = { wake: [[880, 0.12]], case: [[660, 0.12], [990, 0.16]], timeout: [[220, 0.30]] };
function beep(kind) {
  if (!S.beepOn) return;
  try {
    S.beepCtx = S.beepCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (S.beepCtx.state === 'suspended') S.beepCtx.resume();
    const ctx = S.beepCtx; let t = ctx.currentTime + 0.01, total = 0;
    for (const [hz, dur] of BEEPS[kind]) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = hz; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.start(t); o.stop(t + dur + 0.02); t += dur + 0.04; total += dur + 0.04;
    }
    send({ type: 'beep', ms: Math.round(total * 1000) });      // the server ignores a short sound that starts right now (the beep leaking into the microphone)
  } catch { /* no sound available: the screen still shows it */ }
}

/* The wake beep waits until you stop talking, so it never lands in the middle of "Hey Jarvis, mag-record ka na". */
function flushWakeBeep() { clearTimeout(S.wakeBeepTimer); if (S.wakeBeepPending) { S.wakeBeepPending = false; beep('wake'); } }
function wake(ev) {
  ensureCamera();
  S.wakeBeepPending = true; clearTimeout(S.wakeBeepTimer);
  if (!S.hearing) flushWakeBeep(); else S.wakeBeepTimer = setTimeout(flushWakeBeep, 4000);
  S.awakeUntil = Date.now() + (ev.awake_seconds || 8) * 1000;
  setPill();
  clearTimeout(S.awakeTimer);
  S.awakeTimer = setTimeout(setPill, (ev.awake_seconds || 8) * 1000 + 100);
  fire(null, true);
  tagline(`<b>Hey Jarvis</b> heard${ev.source === 'text' ? ' (in the text)' : ''}. Say a command.`);
}

function onCommand(ev) {
  const c = COMMANDS.find(x => x.id === ev.name);
  if (!c) return;
  if (ev.note) { toast(`${c.label}: ${esc(ev.note)}`); return; }
  fire(ev.name);
  S.awakeUntil = 0; setPill();
  const t = ev.timing;
  if (t) S.tags.push({ kind: 'command', label: c.label, ...t, at: ev.time });
  tagline(`<b>${c.label}</b> ${ev.source === 'app' ? '(button)' : 'heard'}${t ? ` · <span class="t">${t.total_s.toFixed(1)} s</span> after you stopped talking <small>(wait ${t.wait_s} + queue ${t.queue_s} + Whisper ${t.whisper_s})</small>` : ''}`);
  if (ev.name === 'scan_license') { S.images.license = snap('LICENSE'); S.scanAt.license = ev.clock; follow(1); }
  if (ev.name === 'scan_cr') { S.images.cr = snap('CR'); S.scanAt.cr = ev.clock; follow(2); }
  if (ev.name === 'capture') { S.images.plates.push(snap('PHOTO')); follow(2); }
  if (S.toolsOpen) renderTools();
  renderStep();
}

function onViolations(ev) {
  const t = ev.timing;
  ev.labels.forEach(l => {
    if (!S.violLive[l]) S.violLive[l] = { at: ev.time, timing: t, isNew: true };
  });
  Object.keys(S.violLive).forEach(l => { if (!ev.labels.includes(l)) delete S.violLive[l]; });
  if (ev.new.length) {
    if (t) S.tags.push({ kind: 'violation', label: ev.new.join(', '), ...t, at: ev.time });
    toast(`<b>Violation</b> ${esc(ev.new.join(', '))}${t ? ` · ${t.total_s.toFixed(1)} s` : ''}`);
    tagline(`<b>Violation: ${esc(ev.new.join(', '))}</b>${t ? ` · <span class="t">${t.total_s.toFixed(1)} s</span> after the line ended <small>(wait ${t.wait_s} + queue ${t.queue_s} + Whisper ${t.whisper_s})</small>` : ''}`);
    follow(3);
  }
  S.dismissed = new Set([...S.dismissed].filter(l => ev.labels.includes(l)));
  renderStep();
}

function onCaseStart(ev) {
  beep('case');
  ensureCamera();
  S.caseActive = true; S.sid = ev.sid; S.caseStartWall = Date.now(); S.caseEnded = null; S.controlNo = newControl();
  S.violLive = {}; S.violManual = []; S.dismissed = new Set(); S.finals = {}; S.finalLines = []; S.progress = null; S.finalErr = null; S.tags = [];
  S.saved = false; S.upload = null;
  S.images = { license: null, cr: null, plates: [] };
  if (S.videoURL && S.videoURL.startsWith('blob:')) URL.revokeObjectURL(S.videoURL);
  S.videoURL = null;
  $('#pillRec').classList.remove('hidden');
  S.recTimer = setInterval(() => { $('#recTime').textContent = mmss((Date.now() - S.caseStartWall) / 1000); }, 500);
  $('#ctrlNo').textContent = S.controlNo;
  toast('<b>Case started</b> · ' + S.controlNo);
  follow(3);
  renderStep();
}

function onCaseEnd(ev) {
  S.caseActive = false; S.caseEnded = ev;
  clearInterval(S.recTimer);
  $('#pillRec').classList.add('hidden');
  setPill();
  S.saved = false; S.upload = null;
  toast(`<b>Case closed</b> · ${ev.violations.length} violation${ev.violations.length === 1 ? '' : 's'}. Check the ticket, then tap SAVE TICKET.`);
  follow(4, true);
  if (S.recorder && S.recorder.state !== 'inactive') finishVideo(ev.sid);
  gotoStep(S.step);
}

/* The transcript is made after the ticket is saved (not when the case closes). */
async function saveTicket() {
  if (!S.sid || S.saved) return;
  S.saved = true;
  S.progress = { done: 0, total: 0, source: S.videoURL ? 'video' : 'stream' };
  gotoStep(4);
  if (S.upload) await S.upload;                        // the video must be on the server first
  const v = allViolations();
  const ticket = {
    control_no: S.controlNo, saved: new Date().toISOString(), location: S.form.location, date: S.form.date, time: S.form.time,
    violations: v.map(x => x.label), actions: Object.keys(S.form.actions).filter(k => S.form.actions[k]),
  };
  try {
    await fetch(`/api/session/${S.sid}/save`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ticket) });
    toast('<b>Ticket saved</b> · transcribing the recording…');
  } catch (e) { S.saved = false; S.progress = null; toast('Could not save: ' + esc(e), 'err'); gotoStep(4); }
}

/* ------------------------------------------------------------------------------------------------ the little live UI bits */
function setPill() {
  const p = $('#pillState');
  const awake = Date.now() < S.awakeUntil;
  p.className = 'pill';
  if (!S.live && !S.replay) p.textContent = 'Idle';
  else if (awake) { p.textContent = 'Hey Jarvis · say a command'; p.classList.add('on'); }
  else if (S.hearing) { p.textContent = 'Hearing you…'; p.classList.add('hear'); }
  else p.textContent = S.caseActive ? 'Listening · case open' : 'Listening';
}
function tagline(html) { $('#tagLine').innerHTML = html; }
function fire(id, all) {
  document.querySelectorAll('.cmd').forEach(b => {
    if (all) { b.classList.add('fire'); setTimeout(() => b.classList.remove('fire'), 350); }
    else if (b.dataset.id === id) { b.classList.add('fire'); setTimeout(() => b.classList.remove('fire'), 1400); }
  });
}
function follow(step, force) { if (S.follow || force) gotoStep(step); }

function snap(label) {
  const v = $('#cam');
  const c = document.createElement('canvas');
  c.width = 640; c.height = 360;
  const g = c.getContext('2d');
  if (v && v.videoWidth) { g.drawImage(v, 0, 0, 640, 360); }
  else { g.fillStyle = '#2a3b5c'; g.fillRect(0, 0, 640, 360); g.fillStyle = '#9db0cf'; g.font = '28px sans-serif'; g.textAlign = 'center'; g.fillText('no camera frame', 320, 190); }
  g.fillStyle = 'rgba(15,39,72,.8)'; g.fillRect(0, 322, 640, 38);
  g.fillStyle = '#fff'; g.font = '20px sans-serif'; g.textAlign = 'left'; g.fillText(`${label}  ${new Date().toLocaleTimeString()}`, 12, 348);
  return c.toDataURL('image/jpeg', 0.82);
}

/* ------------------------------------------------------------------------------------------------ audio in: microphone -> 16 kHz int16 -> websocket */
const WORKLET = `class Tap extends AudioWorkletProcessor { process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; } } registerProcessor('tap', Tap);`;

function resample(x, rate) {                       // box-filter down to 16 kHz
  const r = rate / 16000;
  const buf = new Float32Array(S.carry.length + x.length);
  buf.set(S.carry); buf.set(x, S.carry.length);
  const out = [];
  let p = S.phase;
  while (p + r <= buf.length) {
    const a = Math.floor(p); let b = Math.floor(p + r); if (b <= a) b = a + 1;
    let s = 0; for (let i = a; i < b; i++) s += buf[i];
    out.push(s / (b - a)); p += r;
  }
  const keep = Math.floor(p);
  S.carry = buf.slice(keep); S.phase = p - keep;
  return out;
}
function pushPCM(floats) {
  for (const f of floats) S.outBuf.push(Math.max(-1, Math.min(1, f)) * 32767);
  while (S.outBuf.length >= 1600) {
    const chunk = Int16Array.from(S.outBuf.splice(0, 1600));
    if (S.ws && S.wsReady) { S.ws.send(chunk.buffer); S.sent += 1600; }
  }
}

async function goLive() {
  if (S.live) return stopLive();
  if (S.replay) return;
  if (transcribing()) { toast('The transcript is still being made. Wait until it is ready.', 'err'); return; }
  try { await connect(); } catch { toast('Cannot reach the app server', 'err'); return; }
  let stream;
  const audio = { channelCount: 1, echoCancellation: S.mic.echo, noiseSuppression: S.mic.noise, autoGainControl: S.mic.agc };
  try { stream = await navigator.mediaDevices.getUserMedia({ audio }); }
  catch (e2) { toast('The microphone is blocked or missing: ' + esc(e2.name || e2) + '. Allow it in the browser, or use “Replay a recording” in the test tools.', 'err', 8000); return; }
  S.stream = stream;
  try { (await navigator.mediaDevices.getUserMedia({ video: true })).getTracks().forEach(t => t.stop()); }      // ask for the camera once now; it only runs during a case
  catch { toast('No camera found: cases will be audio only.'); }
  S.ctx = new (window.AudioContext || window.webkitAudioContext)();
  await S.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })));
  const src = S.ctx.createMediaStreamSource(stream);
  S.node = new AudioWorkletNode(S.ctx, 'tap');
  const mute = S.ctx.createGain(); mute.gain.value = 0;
  src.connect(S.node); S.node.connect(mute); mute.connect(S.ctx.destination);
  S.carry = new Float32Array(0); S.phase = 0; S.outBuf = [];
  S.node.port.onmessage = m => {
    const x = m.data; let e = 0; for (let i = 0; i < x.length; i += 4) e += x[i] * x[i];
    $('#meterBar').style.width = Math.min(100, Math.sqrt(e / (x.length / 4)) * 400) + '%';
    pushPCM(resample(x, S.ctx.sampleRate));
  };
  S.live = true;
  const b = $('#btnLive'); b.textContent = 'STOP'; b.classList.add('live');
  setPill();
  tagline('Idle: listening for <b>“Hey Jarvis”</b>. The camera starts with the case.');
}

function stopLive() {
  S.live = false;
  send({ type: 'listening', on: false });
  if (S.caseActive) send({ type: 'command', name: 'end_recording' });
  clearInterval(S.rotateTimer);
  if (S.recorder && S.recorder.state !== 'inactive' && !S.caseActive) { S.recorder.onstop = null; S.recorder.stop(); }
  setTimeout(() => {
    if (S.node) { S.node.disconnect(); S.node = null; }
    if (S.ctx) { S.ctx.close(); S.ctx = null; }
    if (S.stream && !S.caseActive) { S.stream.getTracks().forEach(t => t.stop()); S.stream = null; $('#cam').srcObject = null; $('#noVideo').classList.remove('hidden'); }
  }, 600);
  const b = $('#btnLive'); b.textContent = 'GO LIVE'; b.classList.remove('live');
  $('#meterBar').style.width = '0';
  setPill();
}

/* ------------------------------------------------------------------------------------------------ video recording (runs from "Go live"; the case marks where it starts and ends) */
function startRecorder() {
  if (!S.stream || typeof MediaRecorder === 'undefined') return;
  const types = ['video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4'];
  const hasV = S.stream.getVideoTracks().length > 0;
  const mime = (hasV ? types : ['audio/webm;codecs=opus', 'audio/webm']).find(t => MediaRecorder.isTypeSupported(t));
  try { S.recorder = new MediaRecorder(S.stream, mime ? { mimeType: mime } : undefined); } catch { S.recorder = null; return; }
  S.chunks = [];
  const vid = 'v' + (++S.vidN);
  S.vid = vid;
  S.recorder.ondataavailable = e => { if (e.data && e.data.size) S.chunks.push(e.data); };
  S.recorder.onstart = () => send({ type: 'marker', vid });
  S.recorder.start(1000);
}

/* The camera runs only from "Hey Jarvis" until the case is closed. Idle = microphone + wake word only. */
async function ensureCamera() {
  if (!S.live || !S.stream || S.camBusy || typeof MediaRecorder === 'undefined') return;
  if (S.recorder && S.recorder.state !== 'inactive') return;
  S.camBusy = true;
  try {
    const vs = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } });
    const vt = vs.getVideoTracks()[0];
    if (!S.live || !S.stream) { vt.stop(); return; }
    S.stream.addTrack(vt);
    const cam = $('#cam'); cam.srcObject = S.stream; await cam.play().catch(() => { }); $('#noVideo').classList.add('hidden');
  } catch { toast('No camera found: this case is audio only.'); }
  finally { S.camBusy = false; }
  if (S.live && S.stream && !(S.recorder && S.recorder.state !== 'inactive')) startRecorder();
}
function stopCamera() {
  if (S.stream) S.stream.getVideoTracks().forEach(t => { t.stop(); S.stream.removeTrack(t); });
  $('#cam').srcObject = null; $('#noVideo').classList.remove('hidden');
}
function dropCamera() {                            // "Hey Jarvis" but no case: throw the short recording away
  if (S.recorder && S.recorder.state !== 'inactive') { S.recorder.onstop = null; S.recorder.stop(); }
  S.recorder = null; stopCamera();
}
function finishVideo(sid) {
  const rec = S.recorder, vid = S.vid, mime = rec.mimeType || 'video/webm';
  rec.onstop = async () => {
    const blob = new Blob(S.chunks, { type: mime });
    S.videoURL = URL.createObjectURL(blob);
    renderIf(4);
    const fd = new FormData(); fd.append('video', blob, 'video.webm'); fd.append('vid', vid);
    S.upload = fetch(`/api/session/${sid}/video`, { method: 'POST', body: fd }).catch(e => toast('Could not send the video: ' + esc(e), 'err'));
    await S.upload;
    S.recorder = null;
    stopCamera();                                   // the case is closed: the camera is off, and the listener is back to idle (microphone + "Hey Jarvis" only)
    if (S.live) tagline('Idle: listening for <b>“Hey Jarvis”</b>. The transcript is being made after you save the ticket.');
  };
  rec.stop();
}

/* ------------------------------------------------------------------------------------------------ replay a recording as if it were the microphone */
async function replayFile(file) {
  if (S.live || S.replay) { toast('Stop the live microphone first.'); return; }
  await connect();
  toast('Replaying “' + esc(file.name) + '” in real time…');
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ac.decodeAudioData(await file.arrayBuffer());
  ac.close();
  const off = new OfflineAudioContext(1, Math.ceil(buf.duration * 16000), 16000);
  const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  S.replay = true; S.outBuf = []; setPill();
  let i = 0;
  const total = pcm.length + 16000 * 2;
  const tm = setInterval(() => {
    const n = Math.min(1600, total - i);
    const chunk = new Int16Array(n);
    for (let k = 0; k < n; k++) { const v = i + k < pcm.length ? pcm[i + k] : 0; chunk[k] = Math.max(-1, Math.min(1, v)) * 32767; }
    if (S.ws && S.wsReady) S.ws.send(chunk.buffer);
    i += n;
    $('#meterBar').style.width = Math.round(100 * i / total) + '%';
    if (i >= total) { clearInterval(tm); S.replay = false; $('#meterBar').style.width = '0'; setPill(); toast('Replay finished.'); }
  }, 100);
}

/* ------------------------------------------------------------------------------------------------ steps */
/* The transcript is being made (or about to be): nothing may cancel it or start a new ticket until it is done. */
function transcribing() { return !!S.progress || !!(S.saved && S.sid && S.caseEnded && !S.finals.video && !S.finals.stream && !S.finalErr); }
function syncNext() {
  const busy = transcribing(), b = $('#btnNext');
  b.textContent = S.step < 4 ? 'NEXT' : (S.caseEnded && !S.saved ? 'SAVE TICKET' : (busy ? 'TRANSCRIBING…' : 'NEW TICKET'));
  b.disabled = S.step === 4 && busy;
  $('#btnCancel').disabled = busy;
  $('#btnLive').disabled = busy && !S.live;
}

function gotoStep(n) {
  S.step = Math.max(1, Math.min(4, n));
  document.querySelectorAll('.stepper li').forEach(li => {
    const k = +li.dataset.step;
    li.classList.toggle('active', k === S.step);
    li.classList.toggle('done', k < S.step);
  });
  $('#stepTitle').textContent = STEPS[S.step - 1].title;
  $('#stepSub').textContent = STEPS[S.step - 1].sub;
  syncNext();
  renderStep();
}

function allViolations() {
  const out = [];
  Object.keys(S.violLive).forEach(l => { if (!S.dismissed.has(l)) out.push({ label: l, live: true, ...S.violLive[l] }); });
  S.violManual.forEach(l => { if (!out.find(o => o.label === l)) out.push({ label: l, live: false }); });
  return out;
}
function offense(l) { return S.offenses.find(o => o.label === l) || { label: l, code: '', name: l, full: l }; }

function renderStep() {
  const b = $('#stepBody');
  const keepScroll = $('#sheet').scrollTop;
  if (S.step === 1) b.innerHTML = stepDriver();
  else if (S.step === 2) b.innerHTML = stepVehicle();
  else if (S.step === 3) b.innerHTML = stepViolation();
  else b.innerHTML = stepPreview();
  $('#sheet').scrollTop = keepScroll;
  const n = allViolations().length;
  const c = $('.stepper li[data-step="3"] .count'); c.textContent = n; c.classList.toggle('hidden', !n);
  S.violLive && Object.values(S.violLive).forEach(v => { v.isNew = false; });
}

function stepDriver() {
  const i = S.images.license;
  return `<div class="stephead"><span class="n">1</span>Driver's license</div>
    <div class="scanbox">${i ? `<img src="${i}" alt="license frame">` : 'Say <b>“Hey Jarvis, scan license”</b><br>or tap <b>Scan license</b> above'}</div>
    <p class="note">${i ? 'Frame taken when the command was heard. Reading the license text (OCR) is not part of this fallback.' : 'The camera frame is saved here when the command is heard.'}</p>
    <label class="f">Full name</label><input class="input" placeholder="Type or leave blank">
    <div class="row2"><div><label class="f">License no.</label><input class="input" placeholder="N00-00-000000"></div><div><label class="f">Expiry</label><input class="input" type="date"></div></div>`;
}
function stepVehicle() {
  const g = S.images.plates.map((p, k) => `<figure><img src="${p}" alt="photo ${k + 1}">Photo ${k + 1}</figure>`).join('');
  return `<div class="stephead"><span class="n">2</span>Vehicle &amp; plate</div>
    <div class="scanbox">${S.images.cr ? `<img src="${S.images.cr}" alt="CR frame">` : 'Say <b>“scan CR”</b> for the registration<br>or <b>“capture”</b> for the plate photo'}</div>
    ${S.images.plates.length ? `<div class="gallery">${g}</div>` : ''}
    <div class="row2"><div><label class="f">Plate no.</label><input class="input" placeholder="ABC 1234"></div><div><label class="f">Vehicle type</label><input class="input" placeholder="Motorcycle, car…"></div></div>`;
}
function stepViolation() {
  const v = allViolations();
  const chips = v.length ? v.map(x => {
    const o = offense(x.label), t = x.timing;
    return `<span class="chip ${x.live ? 'live' : ''} ${x.isNew ? 'new' : ''}"><span class="top2">${esc(o.label)}<button class="x" data-rm="${esc(x.label)}" aria-label="remove">✕</button></span><span class="meta">${esc(o.full)}${t ? ` · heard ${t.total_s.toFixed(1)} s after the line` : x.live ? '' : ' · added by hand'}</span></span>`;
  }).join('') : '<div class="none">No Violation Selected</div>';
  const act = k => `<label><input type="checkbox" data-act="${k}" ${S.form.actions[k] ? 'checked' : ''}>${k}</label>`;
  return `<div class="stephead"><span class="n">3</span>Apprehended Violation (City Ordinance)</div>
    <label class="f">Location</label><input class="input" id="fLoc" value="${esc(S.form.location)}" placeholder="Street / barangay">
    <div class="row2"><div><label class="f">Date</label><input class="input" id="fDate" type="date" value="${esc(S.form.date)}"></div><div><label class="f">Time</label><input class="input" id="fTime" type="time" value="${esc(S.form.time)}"></div></div>
    <div class="vhead"><b>Violations</b><button class="choose" id="btnChoose">Choose</button></div>
    <div class="chips">${chips}</div>
    <p class="note">${S.caseActive ? 'Case open: violations the enforcer says are added here as they are heard.' : 'Say <b>“Hey Jarvis, mag-record ka na”</b> to open a case; violations you say after that are added here.'}</p>
    <div class="sec">Other Apprehension Action</div>
    <div class="checks">${act("Driver's License Confiscated")}${act('Plate Confiscated')}${act('Towed')}${act('Impounded')}</div>`;
}

function median(a) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

function stepPreview() {
  const rep = S.finals.video || S.finals.stream;
  const end = S.caseEnded;
  const v = allViolations();
  const dur = end ? Math.max(0, end.end - end.start) : 0;
  const tot = S.tags.map(t => t.total_s);
  let h = `<div class="stephead"><span class="n">4</span>Case summary</div>
    <div class="kv"><span>Control no.</span><span>${esc(S.controlNo)}</span><span>Case length</span><span>${end ? mmss(dur) : '—'}</span>
    <span>Violations (live)</span><span>${v.length}</span><span>Commands / violations tagged</span><span>${S.tags.length}</span>
    <span>Tag speed (median / slowest)</span><span>${tot.length ? median(tot).toFixed(1) + ' s / ' + Math.max(...tot).toFixed(1) + ' s' : '—'}</span></div>`;
  if (S.videoURL) h += `<video class="player" id="player" src="${S.videoURL}" controls playsinline></video>`;
  else if (!S.caseEnded) h += '<p class="note">Open a case with “Hey Jarvis, mag-record ka na”, say your lines, then “case done”. The recording appears here.</p>';
  if (end && !S.saved) h += '<p class="note"><b>Not saved yet.</b> Check the violations, then tap <b>SAVE TICKET</b>. The transcript of everything said is made after you save.</p>';
  if (S.progress) {
    const pc = S.progress.total ? Math.round(100 * S.progress.done / S.progress.total) : 3;
    h += `<div class="sec">Transcribing the recording… ${S.progress.total ? S.progress.done + ' / ' + S.progress.total : ''}</div><div class="progress"><i style="width:${pc}%"></i></div>`;
  }
  if (S.finalErr) h += `<p class="note" style="color:var(--bad)">After-stop pass failed: ${esc(S.finalErr)}</p>`;
  if (rep) {
    h += `<div class="sec">Cross-check: live list vs the recording</div><div class="xcheck">`;
    rep.confirmed.forEach(l => h += `<div class="ok">✓ ${esc(offense(l).label)}: heard live and confirmed in the recording</div>`);
    rep.live_only.forEach(l => h += `<div class="warn">⚠ ${esc(offense(l).label)}: heard live, NOT found in the recording: check it</div>`);
    rep.final_only.forEach(l => h += `<div class="bad">＋ ${esc(offense(l).label)}: found only after the stop: add it?</div>`);
    if (!rep.confirmed.length && !rep.live_only.length && !rep.final_only.length) h += '<div>No violations in either list.</div>';
    h += '</div>';
  }
  const lines = rep ? rep.lines : S.finalLines;
  if (lines.length) {
    h += `<div class="sec">Transcript (everything said, with times)${rep ? ` · ${rep.source === 'video' ? 'from the video' : 'from the live audio'}` : ''}</div><ul class="tx">` + lines.map((l, k) => {
      const cls = (l.in_case ? 'incase ' : '') + (l.commands.length ? 'cmd-line ' : '') + (l.violations.length ? 'viol-line' : '');
      const tg = l.commands.map(c => `<span class="tag c">${esc((COMMANDS.find(x => x.id === c) || { label: c }).label)}</span>`).join('') +
        l.violations.map(x => `<span class="tag v">${esc(x)}</span>`).join('') + l.hints.map(x => `<span class="tag h">${esc(x)}?</span>`).join('');
      return `<li class="${cls}" data-seek="${l.t0}"><span class="ts">${mmss(l.t0)}</span>${esc(l.text)}${tg}</li>`;
    }).join('') + '</ul><p class="note">Speaker labels (enforcer / driver) come later with the speaker embedding. Lines with a dark edge are inside the case.</p>';
  }
  if (rep) {
    h += `<div class="sec">Timing</div><table class="t"><tr><th>After-stop pass</th><td>${rep.audio_s} s of audio in ${rep.seconds} s (×${rep.rtf}), beam ${rep.beam}</td></tr>`
      + `<tr><th>Live tags</th><td>${tot.length ? tot.map(x => x.toFixed(1)).join(' · ') + ' s' : '—'}</td></tr></table>`;
  }
  if (end && S.saved) h += `<div class="sec"></div><button class="btn ghost small" id="btnExport">Export report (JSON)</button> <button class="btn ghost small" id="btnRerun">Re-run from live audio</button>`;
  return h;
}

/* ------------------------------------------------------------------------------------------------ choose-violation modal */
function openChoose() {
  $('#chooseModal').classList.remove('hidden');
  $('#searchViol').value = ''; renderViolList(''); $('#searchViol').focus();
}
function renderViolList(q) {
  q = q.trim().toLowerCase();
  const list = S.offenses.filter(o => !q || (o.label + ' ' + o.name + ' ' + o.code).toLowerCase().includes(q));
  $('#violList').innerHTML = list.map(o => `<li data-pick="${esc(o.label)}" class="${allViolations().find(x => x.label === o.label) ? 'sel' : ''}">${esc(o.name)}<small>${esc(o.code)} · ${esc(o.label)}</small></li>`).join('') || '<li>No match</li>';
}

/* ------------------------------------------------------------------------------------------------ test tools */
function renderTools() {
  const h = S.hello || {};
  const cfg = h.config || {};
  const med = median(S.tags.map(t => t.total_s));
  const rows = S.tags.slice(-8).reverse().map(t => `<tr><td>${esc(t.label)}</td><td><b>${t.total_s.toFixed(2)}</b></td><td>${t.wait_s}</td><td>${t.queue_s}</td><td>${t.whisper_s}</td></tr>`).join('');
  $('#toolsBody').innerHTML = `
    <h3>Try it without a microphone</h3>
    <button class="btn primary small" id="btnSample">Play sample conversation (about 40 s)</button>
    <p class="note" style="margin-bottom:10px">An enforcer + driver stop made from your own Set 3 clips. Watch the commands, the violation tags and the after-stop transcript.</p>
    <h3>Replay a recording (no microphone needed)</h3>
    <input type="file" id="fileReplay" accept="audio/*,.wav,.m4a,.mp3">
    <p class="note">It plays into the app in real time, exactly like the microphone. Use your own recordings, with or without road noise.</p>
    <h3>What the app hears</h3>
    <label class="tg">Wake word<select id="tWake"><option value="audio">Wake word only: Whisper stays idle until “Hey Jarvis” (lightest)</option><option value="both">audio + “Jarvis” in the text (Whisper reads idle talk)</option><option value="off">off: any line can be a command</option></select></label>
    <label class="tg">Faster commands: shorter Whisper window right after “Hey Jarvis” (about 3x faster, slightly less accurate)<input type="checkbox" id="tFast"></label>
    <label class="tg">Read two neighbouring lines together<input type="checkbox" id="tJoin"></label>
    <label class="tg">Speech sensitivity <span id="vOnLbl">${S.vadOn}</span><input type="range" id="tOn" min="0.2" max="0.6" step="0.05" value="${S.vadOn}"></label>
    <h3>Screen</h3>
    <label class="tg">Jump to the step a command belongs to<input type="checkbox" id="tFollow"></label>
    <label class="tg">Beep sounds (wake heard, case open, nothing heard)<input type="checkbox" id="tBeep"></label>
    <label class="tg">Show what was heard (text) below<input type="checkbox" id="tDebug"></label>
    <h3>Microphone (takes effect at the next Go live)</h3>
    <label class="tg">Echo cancellation<input type="checkbox" id="mEcho"></label>
    <label class="tg">Noise suppression (the Pi has none)<input type="checkbox" id="mNoise"></label>
    <label class="tg">Automatic gain<input type="checkbox" id="mAgc"></label>
    <h3>Numbers</h3>
    <table class="t"><tr><th>Whisper</th><td>${esc(h.whisper || '—')}, live beam ${cfg.LIVE_BEAM ?? '—'}, after-stop beam ${cfg.FINAL_BEAM ?? '—'}</td></tr>
    <tr><th>Wake word</th><td>${h.wake_word ? 'audio model ready (threshold ' + cfg.WAKE_THRESHOLD + ')' : 'audio model missing: text backup only'}</td></tr>
    <tr><th>Line ends after</th><td>${cfg.END_SILENCE ?? '—'} s quiet (${cfg.AWAKE_END_SILENCE ?? '—'} s after the wake word), cap ${cfg.CAP ?? '—'} s</td></tr>
    <tr><th>Lines</th><td>${S.counters.queued} sent to Whisper · ${S.counters.ignored} skipped · ${S.counters.dropped} idle talk dropped</td></tr>
    <tr><th>Tag speed</th><td>${med == null ? '—' : 'median ' + med.toFixed(2) + ' s'}</td></tr></table>
    ${rows ? `<table class="t" style="margin-top:8px"><tr><th>Tag</th><th>total</th><th>wait</th><th>queue</th><th>Whisper</th></tr>${rows}</table>` : ''}
    <h3>Past cases</h3><div id="pastBox">loading…</div>
    ${S.debug ? `<h3>Heard (debug)</h3><div class="debuglog">${esc(S.dbg.join('\n')) || 'nothing yet'}</div>` : ''}`;
  $('#tWake').value = S.wakeMode; $('#tJoin').checked = S.join; $('#tFollow').checked = S.follow; $('#tBeep').checked = S.beepOn; $('#tFast').checked = S.fastCmd; $('#tDebug').checked = S.debug;
  $('#mEcho').checked = S.mic.echo; $('#mNoise').checked = S.mic.noise; $('#mAgc').checked = S.mic.agc;
  loadPast();
}
async function loadPast() {
  try {
    const list = await (await fetch('/api/sessions')).json();
    const box = $('#pastBox'); if (!box) return;
    box.innerHTML = list.length ? list.slice(0, 12).map(s => `<div class="tg" style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line)"><span>${esc(s.sid)}<br><small class="note">${esc(s.live_names.length)} live violation(s)${s.video ? ' · video' : ''}${s.final.length ? ' · transcript' : ''}</small></span><button class="btn ghost small" data-open="${esc(s.sid)}">Open</button></div>`).join('') : '<p class="note">No cases yet.</p>';
  } catch { /* server busy */ }
}
async function openPast(sid) {
  const m = await (await fetch('/api/session/' + sid)).json();
  S.sid = sid; S.caseEnded = { start: m.start, end: m.end, violations: m.live_labels }; S.controlNo = 'CL-' + sid.replace(/\D/g, '').slice(-7);
  S.violLive = {}; m.live_labels.forEach(l => S.violLive[l] = { at: '' }); S.violManual = []; S.dismissed = new Set(); S.tags = [];
  S.saved = true; S.upload = null;
  S.finals = m.finals || {}; S.finalLines = []; S.progress = null; S.videoURL = m.video ? `/api/session/${sid}/video` : null;
  $('#ctrlNo').textContent = S.controlNo;
  closeTools(); gotoStep(4);
}
function openTools() { S.toolsOpen = true; $('#tools').classList.remove('hidden'); renderTools(); }
function closeTools() { S.toolsOpen = false; $('#tools').classList.add('hidden'); }

/* ------------------------------------------------------------------------------------------------ wiring */
function exportReport() {
  const data = { control_no: S.controlNo, sid: S.sid, case: S.caseEnded, tags: S.tags, finals: S.finals, config: S.hello && S.hello.config, exported: new Date().toISOString() };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }));
  a.download = `fallback_camote_${S.sid || 'case'}.json`; a.click();
}

document.addEventListener('click', e => {
  const t = e.target;
  const cmd = t.closest('.cmd');
  if (cmd) { send({ type: 'command', name: cmd.dataset.id }); if (!S.wsReady) connect().then(() => send({ type: 'command', name: cmd.dataset.id })); return; }
  const li = t.closest('.stepper li'); if (li) return gotoStep(+li.dataset.step);
  if (t.closest('#btnChoose')) return openChoose();
  if (t.closest('#closeChoose')) return $('#chooseModal').classList.add('hidden');
  const pick = t.closest('[data-pick]');
  if (pick) { const l = pick.dataset.pick; if (!S.violManual.includes(l) && !allViolations().find(x => x.label === l)) S.violManual.push(l); S.dismissed.delete(l); $('#chooseModal').classList.add('hidden'); renderStep(); return; }
  const rm = t.closest('[data-rm]');
  if (rm) { const l = rm.dataset.rm; S.violManual = S.violManual.filter(x => x !== l); S.dismissed.add(l); renderStep(); return; }
  const seek = t.closest('[data-seek]');
  if (seek) { const p = $('#player'); const rep = S.finals.video; if (p && rep) { p.currentTime = +seek.dataset.seek; p.play().catch(() => { }); } return; }
  const op = t.closest('[data-open]'); if (op) return openPast(op.dataset.open);
  if (t.closest('#btnSample')) {
    closeTools();
    fetch('/samples/sample_conversation.wav').then(r => { if (!r.ok) throw new Error('sample not found'); return r.blob(); })
      .then(b => replayFile(new File([b], 'sample_conversation.wav'))).catch(err => toast('Could not play the sample: ' + esc(err.message || err), 'err'));
    return;
  }
  if (t.closest('#btnExport')) return exportReport();
  if (t.closest('#btnRerun')) { fetch(`/api/session/${S.sid}/final?source=stream`, { method: 'POST' }); toast('Re-running from the live audio…'); return; }
});
document.addEventListener('input', e => {
  const t = e.target;
  if (t.id === 'fLoc') S.form.location = t.value;
  if (t.id === 'fDate') S.form.date = t.value;
  if (t.id === 'fTime') S.form.time = t.value;
  if (t.id === 'searchViol') renderViolList(t.value);
  if (t.id === 'tOn') { S.vadOn = +t.value; $('#vOnLbl').textContent = t.value; store.set('vadOn', S.vadOn); send({ type: 'settings', ON: S.vadOn }); }
});
document.addEventListener('change', e => {
  const t = e.target;
  if (t.dataset && t.dataset.act) S.form.actions[t.dataset.act] = t.checked;
  if (t.id === 'tWake') { S.wakeMode = t.value; store.set('wakeMode2', t.value); send({ type: 'settings', WAKE_MODE: t.value }); }
  if (t.id === 'tJoin') { S.join = t.checked; store.set('join', t.checked); send({ type: 'settings', JOIN_LINES: t.checked }); }
  if (t.id === 'tFast') { S.fastCmd = t.checked; store.set('fastCmd', t.checked); send({ type: 'settings', FAST_CMD: S.fastCmd }); }
  if (t.id === 'tBeep') { S.beepOn = t.checked; store.set('beep', t.checked); if (t.checked) beep('wake'); }
  if (t.id === 'tFollow') { S.follow = t.checked; store.set('follow', t.checked); }
  if (t.id === 'tDebug') { S.debug = t.checked; store.set('debug', t.checked); renderTools(); }
  if (t.id === 'mEcho' || t.id === 'mNoise' || t.id === 'mAgc') { S.mic = { echo: $('#mEcho').checked, noise: $('#mNoise').checked, agc: $('#mAgc').checked }; store.set('mic', S.mic); }
  if (t.id === 'fileReplay' && t.files[0]) { const f = t.files[0]; t.value = ''; closeTools(); replayFile(f).catch(err => toast('Could not play that file: ' + esc(err.message || err), 'err')); }
});
$('#btnLive').addEventListener('click', goLive);
$('#btnTools').addEventListener('click', openTools);
$('#closeTools').addEventListener('click', closeTools);
$('#btnBack').addEventListener('click', () => gotoStep(S.step - 1));
$('#btnNext').addEventListener('click', () => {
  if (S.step < 4) return gotoStep(S.step + 1);
  if (S.caseEnded && !S.saved) return saveTicket();
  newTicket();
});
let cancelArmed = 0;
$('#btnCancel').addEventListener('click', () => {            // two taps instead of a browser pop-up
  const b = $('#btnCancel');
  if (cancelArmed) { cancelArmed = 0; b.textContent = 'CANCEL TICKET'; newTicket(true); return; }
  cancelArmed = 1; b.textContent = 'TAP AGAIN TO CLEAR';
  setTimeout(() => { cancelArmed = 0; b.textContent = 'CANCEL TICKET'; }, 3000);
});
$('#chooseModal').addEventListener('click', e => { if (e.target.id === 'chooseModal') e.target.classList.add('hidden'); });

function newTicket(cancelled) {
  if (transcribing()) { toast('The transcript is still being made. Wait until it is ready.', 'err'); return; }
  if (S.caseActive) send({ type: 'reset' });
  S.caseActive = false; clearInterval(S.recTimer); $('#pillRec').classList.add('hidden');
  S.sid = null; S.caseEnded = null; S.saved = false; S.upload = null; S.controlNo = newControl(); $('#ctrlNo').textContent = S.controlNo;
  S.violLive = {}; S.violManual = []; S.dismissed = new Set(); S.tags = []; S.finals = {}; S.finalLines = []; S.progress = null; S.finalErr = null;
  S.images = { license: null, cr: null, plates: [] }; S.videoURL = null; S.form.actions = {}; S.form.location = '';
  gotoStep(3);
  if (cancelled) toast('Ticket cancelled.');
}

/* ------------------------------------------------------------------------------------------------ start */
(async function init() {
  $('#cmdRow').innerHTML = COMMANDS.map(c => `<button class="cmd" data-id="${c.id}">${c.label}${c.say ? `<small>${c.say}</small>` : ''}</button>`).join('');
  $('#ctrlNo').textContent = S.controlNo;
  try { S.offenses = await (await fetch('/api/offenses')).json(); } catch { /* shown empty */ }
  gotoStep(3);
  connect().catch(() => { });
})();
})();
