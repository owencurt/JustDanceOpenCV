import { useEffect, useMemo, useRef, useState } from 'react'

const POSE_CONNECTIONS = [
  [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 12], [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
]

const FEEDBACK_COLORS = {
  perfect: '#50ff78',
  great: '#50c8ff',
  good: '#ffc850',
  ok: '#f0b450',
  miss: '#888',
}

const MEDIA_SYNC_TOLERANCE_SEC = 0.12
const MOVE_TRANSITION_MS = 240
const QUEUE_TRANSITION_MS = 260

const clamp01 = (v) => Math.max(0, Math.min(1, v))
const easeOutCubic = (t) => 1 - Math.pow(1 - clamp01(t), 3)
const lerp = (a, b, t) => a + (b - a) * t

const moveKey = (move, index = 0) => {
  if (!move) return `none-${index}`
  return `${move.name || 'move'}-${move.start_ms ?? index}`
}

function lerpNormPose(fromPose, toPose, t) {
  const keys = new Set([
    ...Object.keys(fromPose || {}),
    ...Object.keys(toPose || {}),
  ])
  const out = {}
  keys.forEach((key) => {
    const from = fromPose?.[key]
    const to = toPose?.[key]
    if (from && to) {
      out[key] = [lerp(from[0], to[0], t), lerp(from[1], to[1], t)]
      return
    }
    out[key] = to || from
  })
  return out
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + radius, y)
  ctx.lineTo(x + w - radius, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius)
  ctx.lineTo(x + w, y + h - radius)
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h)
  ctx.lineTo(x + radius, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius)
  ctx.lineTo(x, y + radius)
  ctx.quadraticCurveTo(x, y, x + radius, y)
  ctx.closePath()
}

function drawNormPose(ctx, normXY, x, y, w, h, { lineColor = '#ddd', pointColor = '#ffc800' } = {}) {
  if (!normXY) return
  const pts = {}
  Object.entries(normXY).forEach(([k, [nx, ny]]) => {
    const idx = Number(k)
    pts[idx] = [x + nx * w, y + ny * h]
  })

  ctx.strokeStyle = lineColor
  ctx.lineWidth = 2
  POSE_CONNECTIONS.forEach(([a, b]) => {
    if (!pts[a] || !pts[b]) return
    ctx.beginPath()
    ctx.moveTo(...pts[a])
    ctx.lineTo(...pts[b])
    ctx.stroke()
  })

  ctx.fillStyle = pointColor
  Object.values(pts).forEach(([px, py]) => {
    ctx.beginPath()
    ctx.arc(px, py, 3, 0, Math.PI * 2)
    ctx.fill()
  })
}

export default function App() {
  const videoRef = useRef(null)
  const audioRef = useRef(null)
  const canvasRef = useRef(null)
  const captureRef = useRef(document.createElement('canvas'))
  const wsRef = useRef(null)
  const sendBusyRef = useRef(false)
  const inFlightSocketRef = useRef(null)
  const serverFrameRef = useRef(null)
  const mediaUrlRef = useRef('')
  const audioEnabledRef = useRef(true)
  const currentMoveRef = useRef(null)
  const previousMoveRef = useRef(null)
  const moveTransitionStartRef = useRef(performance.now())
  const queueTransitionRef = useRef({
    fromKeys: [],
    toKeys: [],
    startAt: performance.now(),
  })

  const [status, setStatus] = useState('Connecting...')
  const [charts, setCharts] = useState([])
  const [chartMeta, setChartMeta] = useState({})
  const [scoringChart, setScoringChart] = useState('ymca.json')
  const [choreoChart, setChoreoChart] = useState('ymca_extra.json')
  const [audioEnabled, setAudioEnabled] = useState(true)
  const [audioStatus, setAudioStatus] = useState('Audio follows the chart media when available.')

  const canSendFrames = useMemo(() => !!wsRef.current && wsRef.current.readyState === WebSocket.OPEN, [status])
  const mediaUrl = chartMeta[choreoChart]?.media_url || ''

  useEffect(() => {
    mediaUrlRef.current = mediaUrl
  }, [mediaUrl])

  useEffect(() => {
    audioEnabledRef.current = audioEnabled
  }, [audioEnabled])

  const seekAndPause = (targetSec = 0) => {
    const media = audioRef.current
    const activeMediaUrl = mediaUrlRef.current
    if (!media || !activeMediaUrl) return

    const clampedTarget = Math.max(0, targetSec)
    try {
      if (Math.abs((media.currentTime || 0) - clampedTarget) > 0.05) {
        media.currentTime = clampedTarget
      }
    } catch (_) {
      // Metadata may not be available yet.
    }
    media.pause()
  }

  const syncAudioToFrame = (frame) => {
    const media = audioRef.current
    const activeMediaUrl = mediaUrlRef.current
    if (!media || !activeMediaUrl || !frame) return

    if (!audioEnabledRef.current) {
      seekAndPause((frame.game_ts_ms || 0) / 1000)
      return
    }

    if (frame.state === 'running') {
      const targetSec = Math.max(0, (frame.game_ts_ms || 0) / 1000)
      if (Math.abs((media.currentTime || 0) - targetSec) > MEDIA_SYNC_TOLERANCE_SEC) {
        media.currentTime = targetSec
      }
      if (media.paused) {
        media.play().catch(() => {
          setAudioStatus('Audio blocked by browser autoplay policy. Click Audio Off/On, then Resume.')
        })
      }
      return
    }

    if (frame.state === 'paused') {
      seekAndPause((frame.game_ts_ms || 0) / 1000)
      return
    }

    if (frame.state === 'countdown' || frame.state === 'idle') {
      seekAndPause(0)
    }
  }

  useEffect(() => {
    fetch('/api/charts')
      .then((r) => r.json())
      .then((d) => {
        setCharts(d.charts || [])
        setChartMeta(d.chart_meta || {})
        if (d.default_scoring) setScoringChart(d.default_scoring)
        if (d.default_choreo) setChoreoChart(d.default_choreo)
      })
      .catch(() => setStatus('Backend unreachable'))
  }, [])

  useEffect(() => {
    if (!mediaUrl) {
      setAudioStatus('No chart media found. Gameplay runs without soundtrack.')
      return
    }
    setAudioStatus(audioEnabled ? 'Audio enabled.' : 'Audio disabled.')
  }, [audioEnabled, mediaUrl])

  useEffect(() => {
    const media = audioRef.current
    if (!media) return

    media.pause()
    media.load()

    if (mediaUrl) {
      setAudioStatus(audioEnabled ? 'Media loaded. Audio will start with gameplay.' : 'Media loaded. Audio is disabled.')
    } else {
      setAudioStatus('No chart media found. Gameplay runs without soundtrack.')
    }
  }, [mediaUrl, choreoChart, audioEnabled])

  useEffect(() => {
    let stream
    async function setupMedia() {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false,
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
    }

    setupMedia().catch((err) => {
      setStatus(`Camera error: ${err.message}`)
    })

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const connectSocket = () => {
    if (wsRef.current) {
      wsRef.current.close()
    }

    sendBusyRef.current = false
    inFlightSocketRef.current = null

    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${scheme}://${window.location.host}/ws/game`
    const ws = new WebSocket(url)

    ws.onopen = () => {
      sendBusyRef.current = false
      inFlightSocketRef.current = null
      setStatus('Connected')
      ws.send(JSON.stringify({
        scoring_chart: scoringChart,
        choreo_chart: choreoChart,
      }))
    }

    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data)
      if (data.type === 'ready') {
        setStatus(`Ready (${data.state})`)
      } else if (data.type === 'frame_result') {
        const nextMoveKey = moveKey(data.current_move)
        const prevMoveKey = moveKey(currentMoveRef.current)
        if (nextMoveKey !== prevMoveKey) {
          previousMoveRef.current = currentMoveRef.current
          currentMoveRef.current = data.current_move
          moveTransitionStartRef.current = performance.now()
        }

        const nextQueueKeys = (data.upcoming_moves || []).map((m, i) => moveKey(m, i))
        const activeQueueKeys = queueTransitionRef.current.toKeys
        if (nextQueueKeys.join('|') !== activeQueueKeys.join('|')) {
          queueTransitionRef.current = {
            fromKeys: activeQueueKeys.length ? activeQueueKeys : nextQueueKeys,
            toKeys: nextQueueKeys,
            startAt: performance.now(),
          }
        }
        serverFrameRef.current = data
        syncAudioToFrame(data)
      } else if (data.type === 'ack') {
        setStatus(`State: ${data.state}`)
      } else if (data.type === 'error') {
        setStatus(`Error: ${data.error}`)
      }

      if (inFlightSocketRef.current === ws) {
        sendBusyRef.current = false
        inFlightSocketRef.current = null
      }
    }

    ws.onclose = () => {
      if (inFlightSocketRef.current === ws) {
        sendBusyRef.current = false
        inFlightSocketRef.current = null
      }
      setStatus('Disconnected')
    }

    ws.onerror = () => {
      if (inFlightSocketRef.current === ws) {
        sendBusyRef.current = false
        inFlightSocketRef.current = null
      }
    }

    wsRef.current = ws
  }

  useEffect(() => {
    connectSocket()
    return () => wsRef.current?.close()
  }, [scoringChart, choreoChart])

  useEffect(() => {
    const timer = setInterval(() => {
      const ws = wsRef.current
      const video = videoRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN || !video) return
      if (video.videoWidth === 0 || video.videoHeight === 0) return

      if (sendBusyRef.current) {
        if (inFlightSocketRef.current !== ws) {
          sendBusyRef.current = false
          inFlightSocketRef.current = null
        } else {
          return
        }
      }

      const c = captureRef.current
      c.width = 640
      c.height = 360
      const cctx = c.getContext('2d')
      cctx.drawImage(video, 0, 0, c.width, c.height)
      const dataUrl = c.toDataURL('image/jpeg', 0.6)
      const imageB64 = dataUrl.split(',')[1]

      sendBusyRef.current = true
      inFlightSocketRef.current = ws
      ws.send(JSON.stringify({ type: 'frame', image_b64: imageB64 }))
    }, 66)

    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    let rafId
    const render = () => {
      const canvas = canvasRef.current
      const video = videoRef.current
      const frame = serverFrameRef.current
      if (!canvas || !video) {
        rafId = requestAnimationFrame(render)
        return
      }

      const ctx = canvas.getContext('2d')
      const W = canvas.width
      const H = canvas.height
      const now = performance.now()

      const bg = ctx.createLinearGradient(0, 0, W, H)
      bg.addColorStop(0, '#0d0f21')
      bg.addColorStop(0.45, '#191334')
      bg.addColorStop(1, '#0b1a2f')
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, W, H)

      const pulse = 0.5 + 0.5 * Math.sin(now / 380)
      const stageW = W * 0.48
      const stageH = H * 0.78
      const stageX = W * 0.23
      const stageY = H * 0.1
      ctx.save()
      ctx.globalAlpha = 0.92
      drawRoundedRect(ctx, stageX, stageY, stageW, stageH, 24)
      const stageGrad = ctx.createLinearGradient(stageX, stageY, stageX, stageY + stageH)
      stageGrad.addColorStop(0, '#24154f')
      stageGrad.addColorStop(1, '#111328')
      ctx.fillStyle = stageGrad
      ctx.fill()
      ctx.strokeStyle = `rgba(126, 233, 255, ${0.35 + pulse * 0.25})`
      ctx.lineWidth = 3
      ctx.stroke()
      ctx.restore()

      const moveProgress = easeOutCubic((now - moveTransitionStartRef.current) / MOVE_TRANSITION_MS)
      const blendedPose = lerpNormPose(
        previousMoveRef.current?.norm_xy || currentMoveRef.current?.norm_xy,
        currentMoveRef.current?.norm_xy,
        moveProgress,
      )

      if (blendedPose) {
        drawNormPose(
          ctx,
          blendedPose,
          stageX + 45,
          stageY + 62,
          stageW - 90,
          stageH - 140,
          { lineColor: '#f0f5ff', pointColor: '#47ffd5' },
        )
      }

      const nowLabelY = stageY + 36
      ctx.fillStyle = '#7efff3'
      ctx.font = '700 20px Inter, system-ui'
      ctx.fillText('NOW', stageX + 28, nowLabelY)
      ctx.fillStyle = '#ffffff'
      ctx.font = '700 34px Inter, system-ui'
      ctx.fillText(frame?.current_move?.name || 'Get Ready', stageX + 28, nowLabelY + 36)

      const cardW = 265
      const cardH = 118
      const queueX = W - cardW - 24
      const queueY = 82
      const queueGap = 16
      const currentQueue = frame?.upcoming_moves || []
      const prevQueue = queueTransitionRef.current.fromKeys
      const queueProgress = easeOutCubic((now - queueTransitionRef.current.startAt) / QUEUE_TRANSITION_MS)

      ctx.fillStyle = '#9ab2ff'
      ctx.font = '700 18px Inter, system-ui'
      ctx.fillText('UP NEXT', queueX, queueY - 20)

      currentQueue.forEach((m, i) => {
        const key = moveKey(m, i)
        const prevIndex = prevQueue.indexOf(key)
        const fromIndex = prevIndex === -1 ? -1 : prevIndex
        const targetY = queueY + i * (cardH + queueGap)
        const fromY = queueY + fromIndex * (cardH + queueGap)
        const animatedY = lerp(fromY, targetY, queueProgress)
        const isSoon = i === 0

        ctx.save()
        ctx.globalAlpha = fromIndex === -1 ? queueProgress : 1
        drawRoundedRect(ctx, queueX, animatedY, cardW, cardH, 18)
        const cardGrad = ctx.createLinearGradient(queueX, animatedY, queueX + cardW, animatedY + cardH)
        cardGrad.addColorStop(0, isSoon ? '#33205e' : '#23253f')
        cardGrad.addColorStop(1, isSoon ? '#4f2c95' : '#1a1d35')
        ctx.fillStyle = cardGrad
        ctx.fill()
        ctx.strokeStyle = isSoon ? '#4affd7' : '#92a6d6'
        ctx.lineWidth = isSoon ? 3 : 2
        ctx.stroke()

        drawNormPose(
          ctx,
          m.norm_xy,
          queueX + 9,
          animatedY + 9,
          98,
          cardH - 18,
          { lineColor: isSoon ? '#f2f8ff' : '#bbc8e3', pointColor: isSoon ? '#40ffd0' : '#ffd36f' },
        )

        ctx.fillStyle = '#f7f8ff'
        ctx.font = isSoon ? '700 23px Inter, system-ui' : '600 20px Inter, system-ui'
        ctx.fillText(m.name || 'Move', queueX + 120, animatedY + 45)
        const eta = Math.max(0, ((m.start_ms || 0) - (frame?.game_ts_ms || 0)) / 1000)
        ctx.fillStyle = isSoon ? '#7ef9e1' : '#a4b0d5'
        ctx.font = '600 16px Inter, system-ui'
        ctx.fillText(isSoon ? `Next • ${eta.toFixed(1)}s` : `In ${eta.toFixed(1)}s`, queueX + 120, animatedY + 74)
        ctx.restore()
      })

      const pipW = 360
      const pipH = 202
      const pipX = 16
      const pipY = H - pipH - 16
      ctx.save()
      ctx.translate(pipX + pipW, pipY)
      ctx.scale(-1, 1)
      ctx.drawImage(video, 0, 0, pipW, pipH)
      ctx.restore()

      const lms = frame?.pose_landmarks
      if (lms?.length) {
        ctx.strokeStyle = '#f0f0f0'
        ctx.lineWidth = 2
        POSE_CONNECTIONS.forEach(([a, b]) => {
          const pa = lms[a]
          const pb = lms[b]
          if (!pa || !pb) return
          ctx.beginPath()
          ctx.moveTo(pipX + pipW - pa.x * pipW, pipY + pa.y * pipH)
          ctx.lineTo(pipX + pipW - pb.x * pipW, pipY + pb.y * pipH)
          ctx.stroke()
        })
        ctx.fillStyle = '#4cff6a'
        lms.forEach((p) => {
          const x = pipX + pipW - p.x * pipW
          const y = pipY + p.y * pipH
          ctx.beginPath()
          ctx.arc(x, y, 2.5, 0, Math.PI * 2)
          ctx.fill()
        })
      }

      ctx.fillStyle = '#fff'
      ctx.font = '600 32px system-ui'
      ctx.fillText(`Score: ${frame?.total_points ?? 0}`, 16, 42)

      if (frame?.feedback) {
        const tier = frame.feedback.tier
        ctx.fillStyle = FEEDBACK_COLORS[tier] ?? '#fff'
        ctx.font = '700 28px system-ui'
        ctx.fillText(`${tier.toUpperCase()} +${Math.round(frame.feedback.move_score)}`, 16, 76)
      }

      if (frame?.state === 'countdown') {
        const sec = Math.max(0, Math.ceil((frame.countdown_ms_left || 0) / 1000))
        ctx.fillStyle = '#3cf078'
        ctx.font = '700 36px system-ui'
        ctx.fillText(`Starting in ${sec}...`, 16, 118)
      } else if (frame?.state === 'idle') {
        ctx.fillStyle = '#d0d0ff'
        ctx.font = '600 30px system-ui'
        ctx.fillText('Ready.', 16, 118)
      } else if (frame?.state === 'paused') {
        ctx.fillStyle = '#ffe066'
        ctx.font = '700 34px system-ui'
        ctx.fillText('PAUSED', 16, 118)
      }

      ctx.fillStyle = '#ccc'
      ctx.font = '400 16px system-ui'
      ctx.fillText('Browser mode: webcam in frontend, scoring in Python backend', 16, H - 16)

      rafId = requestAnimationFrame(render)
    }

    rafId = requestAnimationFrame(render)
    return () => cancelAnimationFrame(rafId)
  }, [])

  const sendCommand = (action) => {
    if (!canSendFrames) return
    wsRef.current.send(JSON.stringify({ type: 'command', action }))

    if (action === 'start' || action === 'resume') {
      const frame = serverFrameRef.current
      if (frame) syncAudioToFrame(frame)
    }

    if (action === 'reset') {
      seekAndPause(0)
    }
  }

  const toggleAudio = () => {
    const next = !audioEnabled
    setAudioEnabled(next)

    if (!next) {
      seekAndPause((serverFrameRef.current?.game_ts_ms || 0) / 1000)
      return
    }

    const frame = serverFrameRef.current
    if (frame) {
      syncAudioToFrame(frame)
    }
  }

  return (
    <div className="app">
      <header className="toolbar">
        <h1>JustDance Browser</h1>
        <div className="selectors">
          <label>Scoring chart
            <select value={scoringChart} onChange={(e) => setScoringChart(e.target.value)}>
              {charts.map((c) => <option key={`s-${c}`} value={c}>{c}</option>)}
            </select>
          </label>
          <label>Display chart
            <select value={choreoChart} onChange={(e) => setChoreoChart(e.target.value)}>
              {charts.map((c) => <option key={`c-${c}`} value={c}>{c}</option>)}
            </select>
          </label>
        </div>
        <div className="controls">
          <button onClick={() => sendCommand('start')}>Start</button>
          <button onClick={() => sendCommand('pause')}>Pause</button>
          <button onClick={() => sendCommand('resume')}>Resume</button>
          <button onClick={() => sendCommand('reset')}>Reset</button>
          <button onClick={connectSocket}>Reconnect</button>
          <button className="audio-toggle" onClick={toggleAudio} disabled={!mediaUrl}>
            Audio: {audioEnabled ? 'On' : 'Off'}
          </button>
        </div>
        <p className="status">{status}</p>
        <p className="status">Media: {mediaUrl || 'none'}</p>
        <p className="status">{audioStatus}</p>
      </header>

      <canvas ref={canvasRef} width={1280} height={720} className="game-canvas" />
      <video ref={videoRef} className="hidden-video" playsInline muted />
      <video
        ref={audioRef}
        className="hidden-video"
        src={mediaUrl || undefined}
        playsInline
        preload="auto"
        onCanPlay={() => {
          if (mediaUrlRef.current) {
            setAudioStatus(audioEnabledRef.current ? 'Media can play. Audio will start with gameplay.' : 'Media can play. Audio is disabled.')
          }
        }}
        onPlay={() => setAudioStatus('Audio is playing.')}
        onPause={() => {
          if (audioEnabledRef.current && serverFrameRef.current?.state === 'running') {
            setAudioStatus('Audio paused unexpectedly. Browser policy or focus may have interrupted playback.')
          }
        }}
        onError={() => setAudioStatus('Failed to load chart media file. Check path and file location.')}
      />
    </div>
  )
}
