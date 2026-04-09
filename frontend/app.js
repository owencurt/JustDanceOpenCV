const poseConnections = [
  [11,13],[13,15],[12,14],[14,16],[11,12],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28]
];

const stateEls = {
  score: document.getElementById('score'),
  combo: document.getElementById('combo'),
  bestCombo: document.getElementById('bestCombo'),
  status: document.getElementById('status'),
  feedback: document.getElementById('feedback'),
  moveName: document.getElementById('moveName'),
  currentPoseCanvas: document.getElementById('currentPoseCanvas'),
  upcomingMoves: document.getElementById('upcomingMoves'),
  referenceVideo: document.getElementById('referenceVideo'),
  referenceAudio: document.getElementById('referenceAudio'),
};

function drawPose(canvas, normXY) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0e0f15';
  ctx.fillRect(0, 0, w, h);
  if (!normXY) return;

  const pts = {};
  Object.entries(normXY).forEach(([k, [x,y]]) => { pts[parseInt(k,10)] = [x*w, y*h]; });

  ctx.strokeStyle = '#dedede';
  ctx.lineWidth = 2;
  for (const [a,b] of poseConnections) {
    if (pts[a] && pts[b]) {
      ctx.beginPath();
      ctx.moveTo(...pts[a]);
      ctx.lineTo(...pts[b]);
      ctx.stroke();
    }
  }

  ctx.fillStyle = '#ffd34d';
  Object.values(pts).forEach(([x,y]) => {
    ctx.beginPath();
    ctx.arc(x,y,4,0,Math.PI*2);
    ctx.fill();
  });
}

function updateUpcoming(moves) {
  stateEls.upcomingMoves.innerHTML = '';
  moves.forEach((m, idx) => {
    const card = document.createElement('div');
    card.className = 'upcoming-card';
    const name = document.createElement('h3');
    name.textContent = `${idx === 0 ? 'NEXT • ' : ''}${m.name || 'Move'}`;
    const c = document.createElement('canvas');
    c.width = 220;
    c.height = 120;
    drawPose(c, m.norm_xy);
    card.append(name, c);
    stateEls.upcomingMoves.appendChild(card);
  });
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : '{}',
  });
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}) for ${path}`);
  }
}

function applyState(s) {
  stateEls.score.textContent = s.score;
  stateEls.combo.textContent = s.combo;
  stateEls.bestCombo.textContent = s.best_combo;
  stateEls.status.textContent = `${s.status} @ ${Math.floor(s.game_ts_ms/1000)}s`;

  if (s.feedback) {
    stateEls.feedback.textContent = `${s.feedback.tier.toUpperCase()} +${s.feedback.gained} (${s.feedback.move_name})`;
  } else if (s.runtime_error) {
    stateEls.feedback.textContent = `Backend issue: ${s.runtime_error}`;
  } else {
    stateEls.feedback.textContent = 'Keep moving!';
  }

  stateEls.moveName.textContent = s.current_move?.name || '—';
  drawPose(stateEls.currentPoseCanvas, s.current_move?.norm_xy);
  updateUpcoming(s.upcoming_moves || []);
}

async function bootstrap() {
  const initial = await fetch('/api/config').then(r => r.json());
  applyState(initial);

  const displayVideo = initial.chart_meta?.display?.video_path;
  if (displayVideo) {
    const mediaPath = `/media/${displayVideo}`;
    stateEls.referenceVideo.src = mediaPath;
    stateEls.referenceAudio.src = mediaPath;
  }

  document.getElementById('startBtn').onclick = async () => {
    try { await post('/api/session/start'); } catch (e) { stateEls.feedback.textContent = String(e.message || e); }
  };
  document.getElementById('pauseBtn').onclick = async () => {
    try { await post('/api/session/pause-toggle'); } catch (e) { stateEls.feedback.textContent = String(e.message || e); }
  };
  document.getElementById('resetBtn').onclick = async () => {
    try { await post('/api/session/reset'); } catch (e) { stateEls.feedback.textContent = String(e.message || e); }
  };

  const optMap = {
    webcamToggle: 'webcam_enabled',
    overlayToggle: 'show_landmarks',
    refVideoToggle: 'reference_video_enabled',
    refAudioToggle: 'reference_audio_enabled',
  };

  Object.entries(optMap).forEach(([id, key]) => {
    const el = document.getElementById(id);
    el.checked = initial.options[key];
    el.onchange = async () => {
      try {
        await post('/api/options', { [key]: el.checked });
      } catch (e) {
        stateEls.feedback.textContent = String(e.message || e);
      }
      if (key === 'reference_video_enabled') {
        stateEls.referenceVideo.style.display = el.checked ? 'block' : 'none';
      }
      if (key === 'reference_audio_enabled') {
        stateEls.referenceAudio.style.display = el.checked ? 'block' : 'none';
      }
      if (key === 'webcam_enabled') {
        const img = document.getElementById('webcamFeed');
        img.style.opacity = el.checked ? '1' : '0.2';
      }
    };
  });

  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/state');
  ws.onmessage = (ev) => applyState(JSON.parse(ev.data));
  ws.onclose = async () => {
    // Fallback when WS is unavailable (keeps controls/state functional).
    setInterval(async () => {
      try {
        const state = await fetch('/api/config').then(r => r.json());
        applyState(state);
      } catch (_err) {
        // no-op
      }
    }, 500);
  };
}

bootstrap();
