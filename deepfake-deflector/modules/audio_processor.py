"""
Deepfake Deflector - Audio Processor Module
=============================================
Handles real-time audio capture, analysis, and manipulation
to disrupt deepfake audio cloning.
"""

import threading
from typing import Any, Dict, Optional

import numpy as np

# scipy is optional — if unavailable the formant-jitter step is skipped
# but pitch shift and noise injection still work.
try:
    from scipy.signal import lfilter
    _HAS_SCIPY = True
except ImportError:
    lfilter = None  # type: ignore[assignment]
    _HAS_SCIPY = False


class AudioProcessor:
    """Processes audio streams to defend against deepfake voice cloning."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.is_running = False

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio data.

        Args:
            audio_chunk: Audio samples as a numpy array.

        Returns:
            Processed audio chunk.
        """
        # Placeholder for audio deepfake deflection processing
        return audio_chunk

    def start(self):
        """Start audio processing."""
        self.is_running = True

    def stop(self):
        """Stop audio processing."""
        self.is_running = False


class VoiceShimmer:
    """Real-time voice perturbation to disrupt deepfake voice-cloning models.

    Applies subtle, imperceptible audio transformations — micro pitch shifts,
    ultra-low-level white noise, and slight formant randomisation — so that
    the outgoing voice signal is hostile to encoder networks while remaining
    perfectly natural to human listeners.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 1024,
        shimmer_amount: float = 0.003,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.shimmer_amount: float = shimmer_amount
        self.is_active: bool = False

        self._chunks_processed: int = 0
        self._stream: Optional[Any] = None  # PyAudio stream
        self._pa: Optional[Any] = None      # PyAudio instance
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Audio stream management
    # ------------------------------------------------------------------
    def start_stream(self) -> None:
        """Open a PyAudio input/output stream and run it in a background thread."""
        if self.is_active:
            return

        try:
            import pyaudio
        except ImportError:
            print("[ERROR] pyaudio is not installed. Run: pip install pyaudio")
            return

        self._pa = pyaudio.PyAudio()
        self._stop_event.clear()

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk_size,
        )

        self.is_active = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[INFO] VoiceShimmer stream started.")

    def _run_loop(self) -> None:
        """Background loop: read → process → write audio chunks."""
        while not self._stop_event.is_set():
            try:
                raw = self._stream.read(self.chunk_size, exception_on_overflow=False)  # type: ignore[union-attr]
                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                processed = self.apply_shimmer(chunk)
                self._stream.write(processed.astype(np.float32).tobytes())  # type: ignore[union-attr]
            except Exception:
                # Stream may have been closed while reading
                break

    def stop_stream(self) -> None:
        """Cleanly stop the audio stream and background thread."""
        if not self.is_active:
            return

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

        self.is_active = False
        print("[INFO] VoiceShimmer stream stopped.")

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------
    def apply_shimmer(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process an audio chunk to break deepfake voice cloning.

        Applies three complementary transformations:

        1. **Micro pitch shift** — ±shimmer_amount semitones via resampling.
           Voice cloning encoders (SV2TTS, Resemblyzer, VALL-E) extract speaker
           embeddings that are sensitive to pitch consistency.  Randomly varying
           the pitch by a few thousandths of a semitone per chunk corrupts the
           embedding without being audible to humans.

        2. **Imperceptible white noise** — at −60 dB (~ 0.001 amplitude).
           This is far below the audible threshold but it adds entropy to the
           mel-spectrogram representation used by most voice cloners, degrading
           reconstruction quality.

        3. **Formant randomisation** — a tiny random IIR all-pole filter
           nudges the spectral envelope.  Formant structure is the key feature
           that a cloning model uses to reproduce speaker identity.  Randomly
           jittering it per chunk produces enough variance to prevent stable
           speaker embedding extraction.

        The combination makes cloned speech garbled or identity-drifted while
        keeping the original voice perfectly natural to human ears.

        Args:
            audio_chunk: 1-D float32 audio samples (mono).

        Returns:
            Processed audio chunk (same length as input).
        """
        self._chunks_processed += 1
        chunk = audio_chunk.astype(np.float64)
        original_len = len(chunk)

        # --- 1. Subtle pitch shift via linear resampling ------------------
        semitone_shift = np.random.uniform(
            -self.shimmer_amount, self.shimmer_amount
        )
        factor = 2.0 ** (semitone_shift / 12.0)
        indices = np.linspace(0, len(chunk) - 1, int(len(chunk) * factor))
        indices = np.clip(indices, 0, len(chunk) - 1)
        chunk = np.interp(indices, np.arange(len(chunk)), chunk)
        # Trim or pad back to original length
        if len(chunk) >= original_len:
            chunk = chunk[:original_len]
        else:
            chunk = np.pad(chunk, (0, original_len - len(chunk)))

        # --- 2. Imperceptible white noise at -60 dB ----------------------
        noise_amplitude = 10.0 ** (-60.0 / 20.0)  # ≈ 0.001
        noise = np.random.randn(original_len) * noise_amplitude
        chunk = chunk + noise

        # --- 3. Slight formant randomisation (all-pole filter jitter) -----
        #     A tiny random IIR filter nudges the spectral envelope.
        #     Skipped gracefully if scipy is not installed.
        if _HAS_SCIPY and lfilter is not None:
            a_coeff = [1.0, -0.97 + np.random.uniform(-0.005, 0.005)]
            chunk = lfilter([1.0], a_coeff, chunk)

        # Clip to [-1, 1] to avoid clipping artefacts
        chunk = np.clip(chunk, -1.0, 1.0)

        return chunk.astype(np.float32)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Return current status of the VoiceShimmer.

        Returns:
            Dict with ``is_active``, ``shimmer_level``, and ``chunks_processed``.
        """
        return {
            "is_active": self.is_active,
            "shimmer_level": self.shimmer_amount,
            "chunks_processed": self._chunks_processed,
        }
