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
  const canvasRef = useRef(null)
  const captureRef = useRef(document.createElement('canvas'))
  const wsRef = useRef(null)
  const sendBusyRef = useRef(false)
  const serverFrameRef = useRef(null)

  const [status, setStatus] = useState('Connecting...')
  const [charts, setCharts] = useState([])
  const [scoringChart, setScoringChart] = useState('ymca.json')
  const [choreoChart, setChoreoChart] = useState('ymca_extra.json')

  const canSendFrames = useMemo(() => !!wsRef.current && wsRef.current.readyState === WebSocket.OPEN, [status])

  useEffect(() => {
    fetch('/api/charts')
      .then((r) => r.json())
      .then((d) => {
        setCharts(d.charts || [])
        if (d.default_scoring) setScoringChart(d.default_scoring)
        if (d.default_choreo) setChoreoChart(d.default_choreo)
      })
      .catch(() => setStatus('Backend unreachable'))
  }, [])

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

    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${scheme}://${window.location.host}/ws/game`
    const ws = new WebSocket(url)

    ws.onopen = () => {
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
        serverFrameRef.current = data
      } else if (data.type === 'ack') {
        setStatus(`State: ${data.state}`)
      } else if (data.type === 'error') {
        setStatus(`Error: ${data.error}`)
      }
      sendBusyRef.current = false
    }

    ws.onclose = () => {
      setStatus('Disconnected')
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
      if (!ws || ws.readyState !== WebSocket.OPEN || !video || sendBusyRef.current) return
      if (video.videoWidth === 0 || video.videoHeight === 0) return

      const c = captureRef.current
      c.width = 640
      c.height = 360
      const cctx = c.getContext('2d')
      cctx.drawImage(video, 0, 0, c.width, c.height)
      const dataUrl = c.toDataURL('image/jpeg', 0.6)
      const imageB64 = dataUrl.split(',')[1]

      sendBusyRef.current = true
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
      ctx.fillStyle = '#101010'
      ctx.fillRect(0, 0, W, H)

      if (frame?.current_move?.norm_xy) {
        drawNormPose(ctx, frame.current_move.norm_xy, 0, 0, W, H, { lineColor: '#e5e5e5', pointColor: '#ffc800' })
      }

      const cardW = 220
      const cardH = 140
      frame?.upcoming_moves?.forEach((m, i) => {
        const x = W - cardW - 16
        const y = 60 + i * (cardH + 12)
        ctx.fillStyle = '#fff'
        ctx.fillRect(x, y, cardW, cardH)
        ctx.strokeStyle = i === 0 ? '#3cb478' : '#888'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, cardW, cardH)
        drawNormPose(ctx, m.norm_xy, x + 12, y + 12, cardW - 24, cardH - 24, { lineColor: '#aaa', pointColor: '#ffcf57' })
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
        </div>
        <p className="status">{status}</p>
      </header>

      <canvas ref={canvasRef} width={1280} height={720} className="game-canvas" />
      <video ref={videoRef} className="hidden-video" playsInline muted />
    </div>
  )
}
