"""
STT v3 - Real-time Speech-to-Text with dual capture (mic + speakers).
"""

import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

try:
    import soundcard as sc
except ImportError:
    sc = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None

# ── Configuration ───────────────────────────────────────
RATE = 16000
CHUNK_SEC = 0.3
BUFFER_SEC = 2
SILENCE_RMS = 0.005
MODEL_SIZE = "base"
LANGUAGES = {"en": "English", "es": "Spanish"}
PREFS_FILE = Path(__file__).parent / "stt_prefs.json"
# ────────────────────────────────────────────────────────

audio_q: queue.Queue = queue.Queue()


def _load_prefs():
    defaults = {"language": "en", "mode": "1", "mic_idx": None, "spk_idx": None}
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                saved = json.load(f)
            defaults.update({k: v for k, v in saved.items() if k in defaults})
        except Exception:
            pass
    return defaults


def _save_prefs(prefs):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass


class State:
    def __init__(self):
        self.alive = True
        self.paused = False
        self.new_meeting = threading.Event()
        self.lock = threading.Lock()
        self.out_path = None
        prefs = _load_prefs()
        self.language = prefs["language"]
        self.prefs = prefs

    def save(self):
        self.prefs["language"] = self.language
        _save_prefs(self.prefs)

state = State()


class AudioChunk:
    __slots__ = ("data", "src")
    def __init__(self, data, src):
        self.data = data
        self.src = src


def resample(audio, src_rate, dst_rate):
    if src_rate == dst_rate:
        return audio
    ratio = dst_rate / src_rate
    n_out = int(len(audio) * ratio)
    return np.interp(np.arange(n_out) / ratio,
                     np.arange(len(audio)), audio).astype(np.float32)


def detect_compute():
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def list_devices():
    devs = []
    # Loopback via soundcard microphone API
    if sc:
        try:
            for m in sc.all_microphones(include_loopback=True):
                if m.isloopback:
                    devs.append(("sc_loop", m, m.name, "loopback"))
        except Exception:
            pass
    # Loopback via pyaudiowpatch (WASAPI)
    if pyaudio and not devs:
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                if d.get("isLoopbackDevice"):
                    devs.append(("pa_loop", i, d["name"], "loopback"))
            p.terminate()
        except Exception:
            pass
    # Regular input devices via sounddevice
    if sd:
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    devs.append(("sd", i, d["name"], "input"))
        except Exception:
            pass
    return devs


def choose_devices():
    devs = list_devices()
    if not devs:
        print("ERROR: No audio devices found.", flush=True)
        sys.exit(1)

    prefs = state.prefs
    saved_mode = prefs.get("mode", "1")
    saved_mic = prefs.get("mic_idx")
    saved_spk = prefs.get("spk_idx")
    saved_lang = prefs.get("language", "en")

    print("\nAvailable devices:", flush=True)
    for i, (_, _, name, kind) in enumerate(devs):
        print(f"  [{i}] ({kind}) {name}")

    # Language
    lang_list = list(LANGUAGES.items())
    print(f"\n  Language:")
    for i, (code, name) in enumerate(lang_list):
        marker = " *" if code == saved_lang else ""
        print(f"    {i+1} = {name} ({code}){marker}")
    try:
        sel = input(f"  Language [Enter={LANGUAGES.get(saved_lang, saved_lang)}]: ").strip()
    except EOFError:
        sel = ""
    if sel.isdigit() and 1 <= int(sel) <= len(lang_list):
        state.language = lang_list[int(sel)-1][0]
    else:
        state.language = saved_lang
    print(f"  -> {LANGUAGES[state.language]}")

    # Mode
    print(f"\n  Mode:")
    print(f"    1 = Mic only")
    print(f"    2 = Speakers only (loopback)")
    print(f"    3 = Mic + Speakers (both)")
    try:
        mode = input(f"  Mode [Enter={saved_mode}]: ").strip()
    except EOFError:
        mode = ""
    if mode not in ("1", "2", "3"):
        mode = saved_mode

    mic_dev = spk_dev = None

    if mode in ("1", "3"):
        inputs = [(i, d) for i, d in enumerate(devs) if d[3] == "input"]
        if inputs:
            print("\n  Mic devices:")
            for idx, d in inputs:
                marker = " *" if idx == saved_mic else ""
                print(f"    [{idx}] {d[2]}{marker}")
            default_hint = f"Enter={saved_mic}" if saved_mic is not None else "Enter=first"
            try:
                sel = input(f"  Mic # [{default_hint}]: ").strip()
            except EOFError:
                sel = ""
            if sel.isdigit() and 0 <= int(sel) < len(devs):
                mic_dev = devs[int(sel)]
            elif saved_mic is not None and 0 <= saved_mic < len(devs):
                mic_dev = devs[saved_mic]
            else:
                mic_dev = inputs[0][1]
            print(f"  -> Mic: {mic_dev[2]}")
        else:
            print("  WARNING: No input devices found.")

    if mode in ("2", "3"):
        loops = [(i, d) for i, d in enumerate(devs) if d[3] == "loopback"]
        all_opts = list(loops)
        stereo = [(i, d) for i, d in enumerate(devs)
                  if d[3] == "input" and "stereo mix" in d[2].lower()]
        all_opts.extend(stereo)

        if all_opts:
            print("\n  Speaker capture devices:")
            for idx, d in all_opts:
                tag = "loopback" if d[3] == "loopback" else "stereo mix"
                marker = " *" if idx == saved_spk else ""
                print(f"    [{idx}] ({tag}) {d[2]}{marker}")
            default_hint = f"Enter={saved_spk}" if saved_spk is not None else "Enter=first"
            try:
                sel = input(f"  Speaker # [{default_hint}]: ").strip()
            except EOFError:
                sel = ""
            if sel.isdigit() and 0 <= int(sel) < len(devs):
                spk_dev = devs[int(sel)]
            elif saved_spk is not None and 0 <= saved_spk < len(devs):
                spk_dev = devs[saved_spk]
            else:
                spk_dev = all_opts[0][1]
            print(f"  -> Speakers: {spk_dev[2]}")
        else:
            print("  WARNING: No loopback or Stereo Mix device found.")

    # Save preferences
    prefs["language"] = state.language
    prefs["mode"] = mode
    # Find the index in the devs list for saving
    for i, d in enumerate(devs):
        if mic_dev is not None and d is mic_dev:
            prefs["mic_idx"] = i
        if spk_dev is not None and d is spk_dev:
            prefs["spk_idx"] = i
    _save_prefs(prefs)

    return mic_dev, spk_dev


# ── Subprocess probe for soundcard loopback ─────────────

def _probe_sc_loopback(mic_name):
    """Test soundcard loopback in a subprocess to detect native crashes."""
    code = (
        "import soundcard as sc, numpy as np\n"
        f"name = {mic_name!r}\n"
        "for m in sc.all_microphones(include_loopback=True):\n"
        "    if m.isloopback and m.name == name:\n"
        "        with m.recorder(samplerate=16000) as r:\n"
        "            d = r.record(1600)\n"
        "            print(f'OK {d.shape}')\n"
        "            break\n"
        "else:\n"
        "    print('NOTFOUND')\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=10)
        ok = r.returncode == 0 and "OK" in r.stdout
        if not ok:
            err = r.stderr.strip()[-200:] if r.stderr else ""
            print(f"    Probe failed (exit={r.returncode}). {err}", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        print("    Probe timed out.", flush=True)
        return False
    except Exception as e:
        print(f"    Probe error: {e}", flush=True)
        return False


# ── Capture backends ────────────────────────────────────

def _capture_sc_loopback(mic_obj, label):
    frames = int(RATE * CHUNK_SEC)
    try:
        with mic_obj.recorder(samplerate=RATE) as rec:
            print(f"  [{label}] Loopback: {mic_obj.name}", flush=True)
            while state.alive:
                pcm = rec.record(frames).astype(np.float32)
                mono = pcm.mean(axis=1) if pcm.ndim == 2 else pcm
                if not state.paused:
                    audio_q.put(AudioChunk(mono, label))
        return True
    except Exception as e:
        print(f"  [{label}] soundcard loopback failed: {e}", flush=True)
        return False


def _capture_pa_loopback(device_idx, label):
    """Capture loopback audio via PyAudioWPatch WASAPI."""
    if pyaudio is None:
        return False
    p = pyaudio.PyAudio()
    try:
        info = p.get_device_info_by_index(device_idx)
    except Exception as e:
        print(f"  [{label}] PyAudio device error: {e}", flush=True)
        p.terminate()
        return False

    native_rate = int(info["defaultSampleRate"])
    ch = max(info["maxInputChannels"], 1)
    chunk_frames = int(native_rate * CHUNK_SEC)

    def callback(in_data, frame_count, time_info, status):
        if state.paused:
            return (None, pyaudio.paContinue)
        audio = np.frombuffer(in_data, dtype=np.float32)
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)
        if native_rate != RATE:
            audio = resample(audio, native_rate, RATE)
        audio_q.put(AudioChunk(audio, label))
        return (None, pyaudio.paContinue)

    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=ch,
            rate=native_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=chunk_frames,
            stream_callback=callback,
        )
        print(f"  [{label}] PyAudio loopback: {info['name']} ({native_rate}Hz)", flush=True)
        stream.start_stream()
        while state.alive and stream.is_active():
            time.sleep(CHUNK_SEC)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
    except Exception as e:
        print(f"  [{label}] PyAudio loopback failed: {e}", flush=True)
        p.terminate()
        return False


def _capture_sd(device_idx, label):
    try:
        info = sd.query_devices(device_idx)
    except Exception as e:
        print(f"  [{label}] ERROR: Can't query device {device_idx}: {e}", flush=True)
        return False
    ch = min(info["max_input_channels"], 2)
    native_rate = int(info["default_samplerate"])

    # Try RATE first; if unsupported, use native rate + resample
    use_rate = RATE
    need_resample = False
    try:
        sd.check_input_settings(device=device_idx, samplerate=RATE, channels=ch)
    except Exception:
        use_rate = native_rate
        need_resample = (native_rate != RATE)

    frames = int(use_rate * CHUNK_SEC)

    def cb(data, n, t, status):
        if state.paused:
            return
        if status:
            print(f"  !! [{label}] {status}", flush=True)
        mono = data[:, 0] if ch == 1 else data.mean(axis=1)
        chunk = mono.copy().astype(np.float32)
        if need_resample:
            chunk = resample(chunk, use_rate, RATE)
        audio_q.put(AudioChunk(chunk, label))

    try:
        with sd.InputStream(samplerate=use_rate, device=device_idx, channels=ch,
                            dtype="float32", blocksize=frames, callback=cb):
            extra = f" (resampling {use_rate}->{RATE}Hz)" if need_resample else ""
            print(f"  [{label}] Capturing: {info['name']}{extra}", flush=True)
            while state.alive:
                sd.sleep(int(CHUNK_SEC * 1000))
        return True
    except Exception as e:
        print(f"  [{label}] ERROR: {e}", flush=True)
        return False


def capture_stream(dev, label):
    if dev is None:
        return
    backend = dev[0]

    if backend == "sc_loop":
        print(f"  [{label}] Probing loopback '{dev[2]}'...", flush=True)
        if _probe_sc_loopback(dev[2]):
            print(f"  [{label}] Probe OK.", flush=True)
            if _capture_sc_loopback(dev[1], label):
                return
        else:
            print(f"  [{label}] soundcard loopback not usable.", flush=True)

        # Fallback 1: pyaudiowpatch
        if pyaudio:
            print(f"  [{label}] Trying PyAudioWPatch...", flush=True)
            try:
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    d = p.get_device_info_by_index(i)
                    if d.get("isLoopbackDevice") and dev[2].lower()[:20] in d["name"].lower():
                        p.terminate()
                        if _capture_pa_loopback(i, label):
                            return
                        break
                else:
                    p.terminate()
            except Exception:
                pass

        # Fallback 2: Stereo Mix
        if sd:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and "stereo mix" in d["name"].lower():
                    print(f"  [{label}] Falling back to Stereo Mix ({d['name']})...", flush=True)
                    print(f"  [{label}] NOTE: Stereo Mix only captures Realtek audio, not USB.", flush=True)
                    if _capture_sd(i, label):
                        return
                    break

        print(f"  [{label}] All loopback methods failed.", flush=True)
        if not pyaudio:
            print(f"  [{label}] TIP: pip install pyaudiowpatch  (for WASAPI loopback)", flush=True)

    elif backend == "pa_loop":
        if _capture_pa_loopback(dev[1], label):
            return

    elif backend == "sd":
        _capture_sd(dev[1], label)


# ── Transcription ───────────────────────────────────────

def _do_transcribe(model, audio, f):
    if state.paused:
        return
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_RMS:
        return
    segments, _ = model.transcribe(
        audio,
        language=state.language,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        log_prob_threshold=-0.8,
    )
    for seg in segments:
        text = seg.text.strip()
        if text:
            print(text, flush=True)
            f.write(text + "\n")
            f.flush()


def _new_file():
    p = Path(f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt")
    state.out_path = p
    return p


def _get_audio_chunk(timeout=None):
    """Get next audio chunk, returning None on poison pill or new-meeting signal."""
    try:
        item = audio_q.get(timeout=timeout)
    except queue.Empty:
        return "empty"
    if item is None:
        return None
    if state.new_meeting.is_set():
        return "new_meeting"
    return item


def transcribe_loop(model, out_path, dual):
    target = int(BUFFER_SEC * RATE)

    try:
        while state.alive:
            state.new_meeting.clear()
            cur_path = state.out_path or out_path
            print(f"  Recording to: {cur_path}", flush=True)

            with open(cur_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===\n\n")

                if dual:
                    switch = _transcribe_dual_loop(model, f, target)
                else:
                    switch = _transcribe_single_loop(model, f, target)

            if not switch:
                break

    except Exception as e:
        print(f"  ERROR in transcription: {e}", flush=True)
        import traceback
        traceback.print_exc()


def _transcribe_single_loop(model, f, target):
    buf, buf_len = [], 0
    while state.alive:
        item = _get_audio_chunk(timeout=0.5)
        if item is None:
            return False
        if item == "empty":
            continue
        if item == "new_meeting":
            return True
        buf.append(item.data)
        buf_len += len(item.data)
        if buf_len < target:
            continue
        audio = np.concatenate(buf)[:target]
        buf.clear()
        buf_len = 0
        _do_transcribe(model, audio, f)
    return False


def _transcribe_dual_loop(model, f, target):
    while state.alive:
        if state.new_meeting.is_set():
            return True
        bufs = {}
        deadline = time.monotonic() + BUFFER_SEC
        got_data = False

        while time.monotonic() < deadline:
            item = _get_audio_chunk(timeout=0.05)
            if item is None:
                return False
            if item == "empty":
                continue
            if item == "new_meeting":
                return True
            bufs.setdefault(item.src, []).append(item.data)
            got_data = True

        if not got_data:
            continue

        arrays = []
        for chunks in bufs.values():
            arrays.append(np.concatenate(chunks))

        if not arrays:
            continue

        min_len = min(len(a) for a in arrays)
        min_len = min(min_len, target)

        if len(arrays) == 1:
            mixed = arrays[0][:min_len]
        else:
            mixed = np.sum([a[:min_len] for a in arrays], axis=0).astype(np.float32)
            mixed = np.clip(mixed, -1.0, 1.0)

        _do_transcribe(model, mixed, f)
    return False


# ── Command listener ────────────────────────────────────

def _print_commands():
    status = "PAUSED" if state.paused else "RECORDING"
    lang = LANGUAGES.get(state.language, state.language)
    print(f"\n  [{status} | {lang}]  P=pause  N=new meeting  L=language  Q=quit", flush=True)


def command_listener():
    """Read single-key commands from stdin."""
    import msvcrt
    _print_commands()
    while state.alive:
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            if ch == "p":
                state.paused = not state.paused
                tag = "PAUSED" if state.paused else "RESUMED"
                print(f"\n  >> {tag}", flush=True)
                _print_commands()
            elif ch == "l":
                codes = list(LANGUAGES.keys())
                idx = (codes.index(state.language) + 1) % len(codes)
                state.language = codes[idx]
                state.save()
                print(f"\n  >> Language: {LANGUAGES[state.language]}", flush=True)
                _print_commands()
            elif ch == "n":
                new_path = _new_file()
                state.new_meeting.set()
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break
                state.paused = False
                print(f"\n  >> NEW MEETING -> {new_path}", flush=True)
                _print_commands()
            elif ch == "q":
                print("\n  >> QUIT", flush=True)
                state.alive = False
                audio_q.put(None)
                return
        else:
            time.sleep(0.1)


# ── Main ────────────────────────────────────────────────

def main():
    print("=" * 40)
    print("  STT v3 - Speech-to-Text")
    print("=" * 40)

    mic_dev, spk_dev = choose_devices()
    if not mic_dev and not spk_dev:
        print("  ERROR: No devices selected.")
        input("  Press Enter to exit...")
        return

    hw, ct = detect_compute()
    print(f"\n  Compute: {hw} ({ct})", flush=True)

    model = None
    for mname in [MODEL_SIZE, "tiny"]:
        print(f"  Probing '{mname}'...", flush=True)
        probe = subprocess.run(
            [sys.executable, "-c",
             f"from faster_whisper import WhisperModel; "
             f"WhisperModel('{mname}', device='{hw}', compute_type='{ct}'); "
             f"print('OK')"],
            capture_output=True, text=True, timeout=120,
        )
        if probe.returncode == 0 and "OK" in probe.stdout:
            print(f"  Loading '{mname}'...", flush=True)
            try:
                model = WhisperModel(mname, device=hw, compute_type=ct)
                print(f"  Model: {mname}", flush=True)
                break
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
        else:
            err = probe.stderr.strip()[-300:] if probe.stderr else ""
            print(f"  '{mname}' failed (exit={probe.returncode}).", flush=True)
            if err:
                print(f"    {err}", flush=True)

    if model is None:
        print("\n  No model loaded. Try: pip install pip-system-certs")
        input("  Press Enter to exit...")
        return

    out = Path(f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt")
    state.out_path = out
    print(f"  Output: {out}\n", flush=True)

    dual = mic_dev is not None and spk_dev is not None
    threads = []
    if mic_dev:
        t = threading.Thread(target=capture_stream, args=(mic_dev, "MIC"), daemon=True)
        threads.append(("MIC", t))
    if spk_dev:
        t = threading.Thread(target=capture_stream, args=(spk_dev, "SPK"), daemon=True)
        threads.append(("SPK", t))

    t_stt = threading.Thread(target=transcribe_loop, args=(model, out, dual), daemon=True)
    for _, t in threads:
        t.start()
    t_stt.start()

    time.sleep(2)
    for lbl, t in threads:
        if not t.is_alive():
            print(f"  WARNING: {lbl} thread died.", flush=True)
    if not t_stt.is_alive():
        print("  ERROR: Transcription thread died.")
        input("  Press Enter to exit...")
        return

    active = [n for n, t in threads if t.is_alive()]
    if not active:
        print("  ERROR: No capture threads running.")
        if not pyaudio:
            print("  TIP: pip install pyaudiowpatch  (adds WASAPI loopback support)")
        input("  Press Enter to exit...")
        return

    print(f"  Listening ({' + '.join(active)})...\n", flush=True)

    try:
        command_listener()
    except KeyboardInterrupt:
        pass
    finally:
        state.alive = False
        audio_q.put(None)
        t_stt.join(timeout=5)
        for _, t in threads:
            t.join(timeout=2)
        print(f"\n  Transcript saved: {state.out_path}")
        input("  Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        input(f"  SystemExit({e.code}). Press Enter...")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n  FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        input("  Press Enter to exit...")