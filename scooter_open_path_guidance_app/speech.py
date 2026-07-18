from __future__ import annotations

import threading
import time


class InstructionSpeaker:
    """Background text-to-speech helper that keeps audio off the inference thread."""

    def __init__(self, rate: int = 185, min_repeat_interval_s: float = 1.8) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "Local speech requires pyttsx3. Install it with "
                "`pip install -r scooter_open_path_guidance_app/requirements.txt`."
            ) from exc

        self._pyttsx3 = pyttsx3
        self.rate = rate
        self.min_repeat_interval_s = min_repeat_interval_s
        self._condition = threading.Condition()
        self._pending_text: str | None = None
        self._stop_requested = False
        self._last_spoken_text = ""
        self._last_spoken_at = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, text: str) -> None:
        now_s = time.time()
        normalized = text.strip()
        if not normalized:
            return
        with self._condition:
            if (
                normalized == self._last_spoken_text
                and now_s - self._last_spoken_at < self.min_repeat_interval_s
            ):
                return
            self._pending_text = normalized
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        engine = self._pyttsx3.init()
        engine.setProperty("rate", self.rate)

        while True:
            with self._condition:
                while not self._stop_requested and self._pending_text is None:
                    self._condition.wait()
                if self._stop_requested:
                    break
                text = self._pending_text
                self._pending_text = None

            if text is None:
                continue

            engine.say(text)
            engine.runAndWait()
            self._last_spoken_text = text
            self._last_spoken_at = time.time()

        engine.stop()
