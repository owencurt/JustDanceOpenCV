const state = {
  status: "idle", // idle | countdown | running | paused
  chart: null,
  startEpoch: null,
  pausedMs: 0,
  audioOn: true,
  videoOn: true,
  timer: null,
};

const el = {
  start: document.getElementById("startBtn"),
  pause: document.getElementById("pauseBtn"),
  reset: document.getElementById("resetBtn"),
  audio: document.getElementById("audioBtn"),
  video: document.getElementById("videoBtn"),
  nowMove: document.getElementById("nowMove"),
  nextMove: document.getElementById("nextMove"),
  nextEta: document.getElementById("nextEta"),
  timeline: document.getElementById("timeline"),
  referenceVideo: document.getElementById("referenceVideo"),
  countdown: document.getElementById("countdown"),
  videoPanel: document.getElementById("videoPanel"),
};

async function loadChart() {
  const res = await fetch("/api/chart");
  if (!res.ok) {
    throw new Error("Failed to load chart");
  }
  state.chart = await res.json();
  if (state.chart.media_available) {
    el.referenceVideo.src = state.chart.media_url;
  } else {
    el.videoPanel.insertAdjacentHTML("beforeend", "<div style='position:absolute;left:12px;bottom:12px;background:#2d3349;padding:8px;border-radius:8px'>No reference media found for this chart.</div>");
  }
  renderFrame(0);
}

function getGameMs() {
  if (state.status === "running" && state.startEpoch != null) {
    return performance.now() - state.startEpoch;
  }
  return state.pausedMs;
}

function findMoves(tsMs) {
  const moves = state.chart?.moves ?? [];
  if (!moves.length) {
    return { current: null, upcoming: [] };
  }
  let idx = 0;
  while (idx + 1 < moves.length && moves[idx + 1].start_ms <= tsMs) idx += 1;
  return {
    current: moves[idx],
    upcoming: moves.slice(idx + 1, idx + 6),
  };
}

function renderFrame(tsMs) {
  if (!state.chart) return;
  const { current, upcoming } = findMoves(tsMs);
  el.nowMove.textContent = current?.name || "-";
  el.nextMove.textContent = upcoming[0]?.name || "-";
  const eta = upcoming[0] ? Math.max(0, upcoming[0].start_ms - tsMs) : 0;
  el.nextEta.textContent = upcoming[0] ? `In ${(eta / 1000).toFixed(1)}s` : "--";

  el.timeline.innerHTML = "";
  upcoming.forEach((m, i) => {
    const etaMs = Math.max(0, m.start_ms - tsMs);
    const div = document.createElement("div");
    div.className = `timeline-item ${i === 0 ? "next" : ""}`;
    div.innerHTML = `<div class='label'>${i === 0 ? "Next" : `+${i + 1}`}</div><div class='move-name'>${m.name}</div><div class='eta'>${(etaMs / 1000).toFixed(1)}s</div>`;
    el.timeline.appendChild(div);
  });
}

function updateLoop() {
  const tsMs = getGameMs();
  renderFrame(tsMs);

  if (state.videoOn && state.chart?.media_available && state.status !== "idle") {
    const target = tsMs + (state.chart.offset_ms || 0);
    const drift = Math.abs(el.referenceVideo.currentTime * 1000 - target);
    if (drift > 120) {
      el.referenceVideo.currentTime = Math.max(0, target / 1000);
    }
  }
}

function setToggle(button, isOn, onLabel, offLabel) {
  button.classList.toggle("active", isOn);
  button.textContent = isOn ? onLabel : offLabel;
}

function startCountdown() {
  state.status = "countdown";
  let count = 3;
  el.countdown.textContent = String(count);
  el.countdown.classList.remove("hidden");

  const interval = setInterval(() => {
    count -= 1;
    if (count <= 0) {
      clearInterval(interval);
      el.countdown.classList.add("hidden");
      state.status = "running";
      state.startEpoch = performance.now() - state.pausedMs;
      if (state.chart?.media_available && state.videoOn) {
        el.referenceVideo.currentTime = Math.max(0, state.pausedMs / 1000);
        el.referenceVideo.play().catch(() => {});
      }
      if (state.audioOn) {
        el.referenceVideo.muted = false;
        el.referenceVideo.volume = 1;
      }
      if (!state.timer) {
        state.timer = setInterval(updateLoop, 50);
      }
      return;
    }
    el.countdown.textContent = String(count);
  }, 1000);
}

el.start.addEventListener("click", () => {
  if (!state.chart) return;
  if (state.status === "running") {
    state.pausedMs = 0;
    state.startEpoch = null;
    el.referenceVideo.pause();
    el.referenceVideo.currentTime = 0;
  }
  startCountdown();
});

el.pause.addEventListener("click", () => {
  if (state.status === "running") {
    state.pausedMs = getGameMs();
    state.status = "paused";
    el.referenceVideo.pause();
    return;
  }
  if (state.status === "paused") {
    state.status = "running";
    state.startEpoch = performance.now() - state.pausedMs;
    if (state.videoOn) el.referenceVideo.play().catch(() => {});
  }
});

el.reset.addEventListener("click", () => {
  state.status = "idle";
  state.startEpoch = null;
  state.pausedMs = 0;
  renderFrame(0);
  el.referenceVideo.pause();
  el.referenceVideo.currentTime = 0;
});

el.audio.addEventListener("click", () => {
  state.audioOn = !state.audioOn;
  setToggle(el.audio, state.audioOn, "Audio On", "Audio Off");
  el.referenceVideo.muted = !state.audioOn;
});

el.video.addEventListener("click", () => {
  state.videoOn = !state.videoOn;
  setToggle(el.video, state.videoOn, "Video On", "Video Off");
  el.referenceVideo.style.opacity = state.videoOn ? "1" : "0";
  if (!state.videoOn) {
    el.referenceVideo.pause();
  } else if (state.status === "running") {
    el.referenceVideo.play().catch(() => {});
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "a") el.audio.click();
  if (e.key.toLowerCase() === "v") el.video.click();
});

loadChart().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("beforeend", `<pre style='color:#ff7070'>${err.message}</pre>`);
});
