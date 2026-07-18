from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PHONE_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#0f172a">
  <link rel="manifest" href="/manifest.webmanifest">
  <title>Scooter Guidance Live</title>
  <style>
    :root {
      --bg: #08111f;
      --panel: rgba(13, 23, 42, 0.82);
      --panel-strong: rgba(15, 23, 42, 0.94);
      --text: #ecf4ff;
      --muted: #9fb3c8;
      --accent: #2dd4bf;
      --warn: #f59e0b;
      --danger: #ef4444;
      --ok: #22c55e;
      --line: rgba(159, 179, 200, 0.18);
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Avenir Next, Segoe UI, Helvetica Neue, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(45, 212, 191, 0.18), transparent 35%),
        radial-gradient(circle at bottom right, rgba(245, 158, 11, 0.16), transparent 30%),
        linear-gradient(160deg, #06101c 0%, #0f172a 55%, #1f2937 100%);
    }

    .shell {
      width: min(100%, 760px);
      margin: 0 auto;
      padding: 20px 16px 32px;
    }

    .hero,
    .panel {
      backdrop-filter: blur(18px);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }

    .hero {
      padding: 18px;
      margin-bottom: 14px;
    }

    .title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    h1 {
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: 0.01em;
    }

    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid var(--line);
      font-weight: 600;
    }

    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 16px rgba(45, 212, 191, 0.9);
    }

    .instruction {
      margin-top: 16px;
      padding: 18px;
      border-radius: 20px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }

    .instruction-label {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.72rem;
    }

    .instruction-text {
      margin-top: 8px;
      font-size: clamp(1.4rem, 4.7vw, 2.3rem);
      font-weight: 700;
      line-height: 1.05;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0;
    }

    .metric {
      padding: 16px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric-value {
      margin-top: 6px;
      font-size: 1.5rem;
      font-weight: 700;
    }

    .panel {
      overflow: hidden;
    }

    .frame-wrap {
      position: relative;
      aspect-ratio: 16 / 9;
      background: rgba(8, 17, 31, 0.92);
    }

    .frame-wrap img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .frame-empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 18px;
      text-align: center;
      color: var(--muted);
    }

    .details {
      padding: 16px;
      display: grid;
      gap: 12px;
    }

    .reason-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .reason-chip {
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(45, 212, 191, 0.12);
      border: 1px solid rgba(45, 212, 191, 0.22);
      color: #d7fff8;
      font-size: 0.88rem;
    }

    .voice-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    button {
      appearance: none;
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      background: linear-gradient(135deg, #f59e0b, #fb7185);
      color: #08111f;
      font-weight: 800;
      font-size: 0.95rem;
    }

    .small {
      color: var(--muted);
      font-size: 0.9rem;
    }

    .command-stop .instruction-text,
    .command-stop .metric-value.command-value { color: #fecaca; }
    .command-wait .instruction-text,
    .command-wait .metric-value.command-value { color: #fed7aa; }
    .command-slow_down .instruction-text,
    .command-slow_down .metric-value.command-value { color: #fde68a; }
    .command-maintain_speed .instruction-text,
    .command-maintain_speed .metric-value.command-value { color: #bbf7d0; }
    .command-speed_up .instruction-text,
    .command-speed_up .metric-value.command-value { color: #bfdbfe; }

    @media (max-width: 560px) {
      .meta-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell" id="app">
    <section class="hero command-maintain_speed" id="hero">
      <div class="title-row">
        <div>
          <h1>Scooter Guidance Live</h1>
          <p class="subtitle">Open this on your phone while the detector runs on your laptop.</p>
        </div>
        <div class="status-pill">
          <span class="status-dot"></span>
          <span id="statusText">Connecting</span>
        </div>
      </div>
      <div class="instruction">
        <div class="instruction-label">Current Instruction</div>
        <div class="instruction-text" id="instructionText">Waiting for guidance...</div>
      </div>
      <div class="meta-grid">
        <div class="panel metric">
          <div class="metric-label">Command</div>
          <div class="metric-value command-value" id="commandValue">-</div>
        </div>
        <div class="panel metric">
          <div class="metric-label">Risk</div>
          <div class="metric-value" id="riskValue">-</div>
        </div>
        <div class="panel metric">
          <div class="metric-label">Heading</div>
          <div class="metric-value" id="headingValue">-</div>
        </div>
        <div class="panel metric">
          <div class="metric-label">Last Update</div>
          <div class="metric-value" id="updatedValue">-</div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="frame-wrap">
        <img id="frameImage" alt="Live annotated guidance frame" hidden>
        <div class="frame-empty" id="frameEmpty">Waiting for the first annotated frame...</div>
      </div>
      <div class="details">
        <div>
          <div class="metric-label">Why This Command</div>
          <div class="reason-list" id="reasonList"></div>
        </div>
        <div class="voice-row">
          <button id="voiceButton" type="button">Enable Voice Guidance</button>
          <div class="small" id="voiceStatus">Tap once to allow spoken instructions in your browser.</div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const hero = document.getElementById("hero");
    const instructionText = document.getElementById("instructionText");
    const commandValue = document.getElementById("commandValue");
    const riskValue = document.getElementById("riskValue");
    const headingValue = document.getElementById("headingValue");
    const updatedValue = document.getElementById("updatedValue");
    const reasonList = document.getElementById("reasonList");
    const frameImage = document.getElementById("frameImage");
    const frameEmpty = document.getElementById("frameEmpty");
    const statusText = document.getElementById("statusText");
    const voiceButton = document.getElementById("voiceButton");
    const voiceStatus = document.getElementById("voiceStatus");

    let voiceEnabled = false;
    let lastSpokenInstruction = "";
    let lastFrameKey = "";

    function formatCommand(command) {
      return (command || "-").replaceAll("_", " ");
    }

    function formatHeading(value) {
      if (typeof value !== "number") {
        return "-";
      }
      if (Math.abs(value) < 20) {
        return "center";
      }
      return value < 0 ? "left" : "right";
    }

    function updateReasons(reasons) {
      reasonList.innerHTML = "";
      const items = reasons && reasons.length ? reasons : ["clear_path"];
      for (const reason of items) {
        const chip = document.createElement("span");
        chip.className = "reason-chip";
        chip.textContent = reason.replaceAll("_", " ");
        reasonList.appendChild(chip);
      }
    }

    function speak(text) {
      if (!voiceEnabled || !("speechSynthesis" in window) || !text) {
        return;
      }
      if (text === lastSpokenInstruction) {
        return;
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.04;
      utterance.pitch = 1.0;
      utterance.lang = "en-US";
      window.speechSynthesis.speak(utterance);
      lastSpokenInstruction = text;
    }

    function render(data) {
      const command = data.command || "maintain_speed";
      hero.className = "hero command-" + command;
      instructionText.textContent = data.instruction_text || "Waiting for guidance...";
      commandValue.textContent = formatCommand(command);
      riskValue.textContent = typeof data.risk_score === "number" ? data.risk_score.toFixed(2) : "-";
      headingValue.textContent = formatHeading(data.recommended_heading_px);
      updatedValue.textContent = new Date().toLocaleTimeString();
      statusText.textContent = "Live";
      updateReasons(data.reasons || []);

      if (typeof data.frame_index !== "undefined") {
        const frameKey = String(data.frame_index);
        if (frameKey !== lastFrameKey) {
          frameImage.src = "/frame.jpg?frame=" + encodeURIComponent(frameKey) + "&t=" + Date.now();
          lastFrameKey = frameKey;
        }
      }

      if (data.voice_instruction) {
        speak(data.voice_instruction);
      }
    }

    async function poll() {
      try {
        const response = await fetch("/latest", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        const data = await response.json();
        render(data);
      } catch (error) {
        statusText.textContent = "Reconnecting";
      }
    }

    frameImage.addEventListener("load", () => {
      frameImage.hidden = false;
      frameEmpty.hidden = true;
    });

    voiceButton.addEventListener("click", () => {
      voiceEnabled = true;
      lastSpokenInstruction = "";
      voiceStatus.textContent = "Voice guidance is on.";
      if ("speechSynthesis" in window) {
        const testUtterance = new SpeechSynthesisUtterance("Voice guidance enabled.");
        testUtterance.rate = 1.0;
        testUtterance.lang = "en-US";
        window.speechSynthesis.speak(testUtterance);
      } else {
        voiceStatus.textContent = "This browser does not support speech synthesis.";
      }
    });

    poll();
    window.setInterval(poll, 700);
  </script>
</body>
</html>
"""

PHONE_MANIFEST = {
    "name": "Scooter Guidance Live",
    "short_name": "Guidance",
    "display": "standalone",
    "background_color": "#08111f",
    "theme_color": "#0f172a",
    "start_url": "/",
}


class GuidancePublisher:
    """Tiny polling API for clients such as a Streamlit phone app."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._payload: dict = {"status": "starting"}
        self._frame_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def update(self, payload: dict, frame_jpeg: bytes | None = None) -> None:
        with self._lock:
            self._payload = payload
            if frame_jpeg is not None:
                self._frame_jpeg = frame_jpeg

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._payload)

    def frame_snapshot(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def start(self) -> None:
        if self._server is not None:
            return
        publisher = self
        dashboard_bytes = PHONE_DASHBOARD_HTML.encode("utf-8")
        manifest_bytes = json.dumps(PHONE_MANIFEST).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                route = self.path.split("?", 1)[0]
                if route == "/latest":
                    body = json.dumps(publisher.snapshot()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if route == "/frame.jpg":
                    frame_jpeg = publisher.frame_snapshot()
                    if frame_jpeg is None:
                        self.send_error(503, "Frame not ready")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(frame_jpeg)
                    return
                if route == "/manifest.webmanifest":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/manifest+json")
                    self.send_header("Content-Length", str(len(manifest_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(manifest_bytes)
                    return
                if route in {"/", "/index.html"}:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(dashboard_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(dashboard_bytes)
                    return
                self.send_error(404, "Not found")

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
