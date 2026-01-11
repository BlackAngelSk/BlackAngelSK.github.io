import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
# try importing optional heavy dependencies; allow installer to add them if missing
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except Exception:
    WhisperModel = None
    HAS_FASTER_WHISPER = False

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HAS_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    HAS_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

import warnings
import json
from pathlib import Path
import time
from datetime import datetime
import sys
import subprocess
import shutil
import contextlib

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
WHISPER_MODEL = "large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = os.cpu_count() or 1

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_CODES = {
    "sk": "slk_Latn",
    "cs": "ces_Latn",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "pl": "pol_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "hu": "hun_Latn",
    "uk": "ukr_Cyrl",
    "tr": "tur_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
}

SUPPORTED_LANGUAGES = {
    "sk": "Slovenčina",
    "cs": "Čeština",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "pl": "Polski",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
    "ru": "Русский",
    "hu": "Magyar",
    "uk": "Українська",
    "tr": "Türkçe",
    "ja": "日本語",
    "zh": "中文",
}
# ============================================
# Theme config
THEME_FILE = str(Path(__file__).parent / ".translator_theme.json")
DEFAULT_THEME = "light"
SUPPORTED_THEMES = ("light", "dark")
# ============================================
# Dependency config
REQUIRED_PACKAGES = ["faster-whisper", "transformers", "torch", "sentencepiece"]
# ffmpeg is required by faster-whisper for many inputs — check availability
def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None
# ============================================


def transcribe_audio(model, audio_file):
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        vad_filter=True
    )
    return list(segments), info


def save_files(base, original_segments, translated_segments, target_code):
    with open(base + "_original.txt", "w", encoding="utf-8") as f:
        for s in original_segments:
            f.write(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text.strip()}\n")

    with open(base + f"_{target_code}.txt", "w", encoding="utf-8") as f:
        for line in translated_segments:
            f.write(line + "\n")


def load_nllb(device: str = "cpu"):
    """Load NLLB model. If device == 'cuda' and a GPU is available, load with half precision and device_map auto."""
    if not HAS_TRANSFORMERS or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError("transformers library is not available; run 'Install deps' first")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)

    # prefer GPU with float16 if available (only pass torch_dtype when torch is present)
    try:
        if device == "cuda" and HAS_TORCH and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
            td = getattr(torch, "float16", None)
            if td is not None:
                kwargs["torch_dtype"] = td
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, **kwargs)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        model.eval()
    except Exception:
        # fallback to safe cpu load on error
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        model.eval()
    return tokenizer, model


def translate_segments(tokenizer, model, segments, src_lang, tgt_lang, progress_callback=None):
    """Translate a list of segments. Optionally call progress_callback(index, total)
    after each segment to report progress."""
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translated = []
    total = len(segments)
    for idx, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if not text:
            if progress_callback:
                try:
                    progress_callback(idx, total)
                except Exception:
                    pass
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # use torch.no_grad() when torch is available; otherwise use nullcontext() so linters don't flag this
        _no_grad = torch.no_grad() if (HAS_TORCH and torch is not None) else contextlib.nullcontext()
        with _no_grad:
            output = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4,
                no_repeat_ngram_size=3
            )

        result = tokenizer.decode(output[0], skip_special_tokens=True)

        translated.append(
            f"[{seg.start:.2f}s → {seg.end:.2f}s] {result}"
        )

        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    return translated


class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper + NLLB Translator 3")
        self.root.geometry("700x650")

        self.model = None
        self.tokenizer = None
        self.nllb_model = None
        self.selected_files = []
        # processing state for ETA
        self._processing = False
        self._timings = []
        self._current_file_start = None
        self._current_index = 0
        # per-segment timing (initialized here so linters recognize the attributes)
        self._segment_timings = []
        self._last_segment_time = None

        self.setup_ui()
        # load and apply saved theme (dark/light)
        self.load_theme()
        self.apply_theme(self.theme)
        self.load_whisper()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Button(frame, text="Vybrať súbory", command=self.select_files).pack()
        self.file_label = ttk.Label(frame, text="Žiadne súbory")
        self.file_label.pack(pady=5)

        # theme toggle (created here and updated in apply_theme)
        self.theme_btn = ttk.Button(frame, text="", command=self.toggle_theme)
        self.theme_btn.pack(pady=5)

        # install dependencies button
        self.install_btn = ttk.Button(frame, text="Install deps", command=self.install_dependencies)
        self.install_btn.pack(pady=5)

        # ETA display
        self.eta_label = ttk.Label(frame, text="ETA: —")
        self.eta_label.pack(pady=5)

        # device selection (Auto / GPU / CPU)
        ttk.Label(frame, text="Device").pack()
        self.device_var = tk.StringVar(value=("Auto"))
        ttk.Combobox(
            frame,
            textvariable=self.device_var,
            values=["Auto", "GPU", "CPU"],
            state="readonly",
            width=12
        ).pack()

        ttk.Label(frame, text="Cieľový jazyk").pack()
        # show friendly entries like "en - English" but store code when parsing
        lang_options = [f"{code} - {name}" for code, name in SUPPORTED_LANGUAGES.items()]
        self.lang_var = tk.StringVar(value=f"sk - {SUPPORTED_LANGUAGES['sk']}")
        ttk.Combobox(
            frame,
            textvariable=self.lang_var,
            values=lang_options,
            state="readonly",
            width=30
        ).pack()

        self.start_btn = ttk.Button(
            frame,
            text="Spustiť",
            command=self.start,
            state="disabled"
        )
        self.start_btn.pack(pady=10)

        self.log = scrolledtext.ScrolledText(frame, height=25)
        self.log.pack(fill="both", expand=True)

    def write(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.root.update_idletasks()

    def load_theme(self):
        """Load saved theme from local file, fallback to default."""
        try:
            if os.path.exists(THEME_FILE):
                with open(THEME_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    t = data.get("theme")
                    if t in SUPPORTED_THEMES:
                        self.theme = t
                        return
        except Exception:
            pass
        # fallback
        self.theme = DEFAULT_THEME

    def save_theme(self):
        try:
            with open(THEME_FILE, "w", encoding="utf-8") as f:
                json.dump({"theme": self.theme}, f)
        except Exception:
            pass

    def apply_theme(self, theme=None):
        """Apply theme colors to widgets and ttk styles with high-contrast text."""
        if theme:
            self.theme = theme
        dark = self.theme == "dark"
        if dark:
            bg = "#0b0b0b"; text = "#0400ff"; panel = "#222222"; accent = "#e40a0a"
        else:
            bg = "#ffffff"; text = "#000000"; panel = "#f1f1f1"; accent = "#0077ff"

        # configure root background
        try:
            self.root.configure(bg=bg)
        except Exception:
            pass

        # configure ttk styles where possible
        style = ttk.Style()
        try:
            style.configure("TFrame", background=bg)
            style.configure("TLabel", background=bg, foreground=text)
            style.configure("TButton", background=panel, foreground=text)
            style.configure("TCombobox", fieldbackground=panel, background=panel, foreground=text)
        except Exception:
            pass

        # text/log area colors
        try:
            self.log.configure(bg=panel, fg=text, insertbackground=text)
        except Exception:
            pass
        try:
            self.file_label.configure(background=bg, foreground=text)
        except Exception:
            # ttk Labels may not accept bg/fg on some platforms
            try:
                self.file_label.config(foreground=text)
            except Exception:
                pass

        # update toggle label and style
        if hasattr(self, "theme_btn"):
            try:
                self.theme_btn.configure(style="TButton")
            except Exception:
                pass
            self.update_toggle_label()

    def toggle_theme(self):
        new = "light" if self.theme == "dark" else "dark"
        self.apply_theme(new)
        self.save_theme()

    def update_toggle_label(self):
        if hasattr(self, "theme_btn"):
            self.theme_btn.config(text=("Light mode" if self.theme == "dark" else "Dark mode"))

    def _format_seconds(self, s: float) -> str:
        s = int(round(s))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"

    def _missing_packages(self):
        missing = []
        if not HAS_FASTER_WHISPER:
            missing.append("faster-whisper")
        if not HAS_TRANSFORMERS:
            missing.append("transformers")
        if not HAS_TORCH:
            missing.append("torch")
        # sentencepiece is only needed for some tokenizers
        try:
            import sentencepiece  # noqa: F401
        except Exception:
            missing.append("sentencepiece")
        return missing

    def install_dependencies(self):
        missing = self._missing_packages()
        if not missing and _ffmpeg_available():
            messagebox.showinfo("Deps", "All required dependencies are already installed.")
            return

        if missing:
            if not messagebox.askyesno("Install", f"Missing packages: {', '.join(missing)}. Install now?"):
                return

        if not _ffmpeg_available():
            if not messagebox.askyesno("ffmpeg missing", "ffmpeg is not found in PATH. Some files may fail. Continue installing Python packages?"):
                return

        # disable install button while running
        try:
            self.install_btn.config(state="disabled")
        except Exception:
            pass
        threading.Thread(target=self._run_installer, args=(missing,), daemon=True).start()

    def _run_installer(self, missing):
        # Install missing packages via pip and stream output back to the UI log
        try:
            to_install = missing if missing else []
            # ensure base packages are included
            for pkg in REQUIRED_PACKAGES:
                if pkg not in to_install:
                    to_install.append(pkg)

            for pkg in to_install:
                self.write(f"Installing {pkg} ...")
                cmd = [sys.executable, "-m", "pip", "install", pkg]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                # iterate over proc.stdout directly (safer for linters; avoids calling readline)
                if proc.stdout is not None:
                    for line in proc.stdout:
                        if line:
                            self.write(line.strip())
                rc = proc.wait()
                if rc != 0:
                    self.write(f"Installation of {pkg} failed (rc={rc}). See output above.")
                    messagebox.showerror("Install error", f"Failed to install {pkg}. Check log.")
                    self.install_btn.config(state="normal")
                    return

            # attempt to re-import
            self.write("Re-importing installed packages...")
            self._try_imports()

            missing_after = self._missing_packages()
            if missing_after:
                self.write(f"Still missing: {', '.join(missing_after)}")
                messagebox.showwarning("Partial", f"Some packages are still missing: {', '.join(missing_after)}")
            else:
                self.write("All required packages installed.")
                messagebox.showinfo("Done", "Dependencies installed.")
        except Exception as e:
            self.write(f"Installer error: {e}")
            messagebox.showerror("Install error", str(e))
        finally:
            try:
                self.install_btn.config(state="normal")
            except Exception:
                pass

    def _try_imports(self):
        # best-effort re-imports and flag updates
        global WhisperModel, AutoTokenizer, AutoModelForSeq2SeqLM, torch, HAS_FASTER_WHISPER, HAS_TRANSFORMERS, HAS_TORCH
        try:
            from faster_whisper import WhisperModel
            HAS_FASTER_WHISPER = True
            self.write("faster_whisper available")
        except Exception:
            HAS_FASTER_WHISPER = False
            self.write("faster_whisper not available")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            HAS_TRANSFORMERS = True
            self.write("transformers available")
        except Exception:
            HAS_TRANSFORMERS = False
            self.write("transformers not available")
        try:
            import torch
            HAS_TORCH = True
            self.write(f"torch available: {torch.__version__}")
        except Exception:
            HAS_TORCH = False
            self.write("torch not available")


    def _update_eta_label(self):
        # runs periodically during processing to update ETA
        if not self._processing:
            return
        now = time.time()
        total = len(self.selected_files)
        i = self._current_index
        elapsed_current = 0.0
        if self._current_file_start:
            elapsed_current = now - self._current_file_start
        average = None
        if self._timings:
            average = sum(self._timings) / len(self._timings)

        expected_total = None
        if average is not None:
            expected_total = average
        elif elapsed_current > 0:
            # Heuristic: estimate total time for current file if no prior data
            expected_total = max(10.0, elapsed_current * 1.5)

        if expected_total is None:
            text = "ETA: calculating..."
        else:
            remaining_files = max(0, total - i)
            # estimate remaining = remaining files * expected_total + remaining part of current file
            rem = remaining_files * expected_total + max(0.0, expected_total - elapsed_current)
            if rem < 1:
                text = "ETA: <1s"
            else:
                finish = datetime.fromtimestamp(now + rem).strftime("%H:%M:%S")
                text = f"ETA: {self._format_seconds(rem)} (finish at {finish})"

        try:
            self.eta_label.config(text=text)
        except Exception:
            pass

        try:
            self.root.after(1000, self._update_eta_label)
        except Exception:
            pass

    def _resolve_whisper_device(self):
        choice = (self.device_var.get() or "Auto").lower()
        if choice == "gpu":
            if HAS_TORCH and torch is not None and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        if choice == "cpu":
            return "cpu"
        # Auto: prefer CUDA if available
        return "cuda" if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"

    def _resolve_torch_device(self):
        if not HAS_TORCH or torch is None:
            return torch.device("cpu") if torch is not None else None
        return torch.device("cuda" if (self.device_var.get() or "Auto").lower() in ("auto", "gpu") and torch.cuda.is_available() else "cpu")

    def load_whisper(self):
        def task():
            self.write("Načítavam Whisper...")
            if not HAS_FASTER_WHISPER or WhisperModel is None:
                self.write("! faster-whisper is not installed. Click 'Install deps' to install required packages.")
                try:
                    self.install_btn.config(state="normal")
                    self.start_btn.config(state="disabled")
                except Exception:
                    pass
                return

            device = self._resolve_whisper_device()
            # choose compute type for device
            compute_type = "int8" if device == "cpu" else "float16"
            try:
                self.model = WhisperModel(
                    WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=CPU_THREADS
                )
                self.write(f"✓ Whisper pripravený (device={device}, compute={compute_type})")
            except Exception as e:
                # fallback to CPU
                self.write(f"! Whisper load failed on {device}: {e}. Falling back to CPU")
                try:
                    self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
                    self.write("✓ Whisper pripravený (device=cpu, compute=int8)")
                except Exception as e2:
                    self.write(f"Whisper could not be initialized: {e2}")
                    messagebox.showerror("Whisper error", "Whisper could not be initialized. Please install faster-whisper or check installation.")
                    try:
                        self.start_btn.config(state="disabled")
                    except Exception:
                        pass
                    return

            try:
                self.start_btn.config(state="normal")
            except Exception:
                pass

        threading.Thread(target=task, daemon=True).start()

    def select_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[
                ("Audio/Video", ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg", "*.opus", "*.aac", "*.wma", "*.mp4", "*.mkv", "*.webm", "*.mov", "*.mpg", "*.mpeg", "*.avi", "*.ogv")),
            ]
        )
        if files:
            self.selected_files = list(files)
            self.file_label.config(text=f"{len(files)} súborov")

    def start(self):
        if not self.selected_files:
            return

        def task():
            raw = self.lang_var.get()
            if " - " in raw:
                target_code = raw.split(" - ", 1)[0]
            else:
                target_code = raw

            if target_code not in NLLB_CODES:
                messagebox.showerror("Chyba", f"Nepodporovaný jazyk: {target_code}")
                return

            target_nllb = NLLB_CODES[target_code]

            total_files = len(self.selected_files)
            if total_files == 0:
                messagebox.showinfo("Info", "Žiadne súbory na spracovanie")
                return

            # reset timing state and start ETA updates
            self._timings = []
            self._processing = True
            self._current_index = 0
            self._current_file_start = None
            try:
                self.root.after(1000, self._update_eta_label)
            except Exception:
                pass

            self.write(f"Preklad do: {target_code} ({SUPPORTED_LANGUAGES.get(target_code, 'unknown')})")
            self.write("Načítavam NLLB...")
            if not HAS_TRANSFORMERS:
                self.write("! transformers is not installed. Click 'Install deps' to install required packages.")
                messagebox.showerror("Chyba", "transformers library is missing. Please install via 'Install deps' first.")
                return
            # load NLLB onto the selected device
            nllb_device = "cuda" if (self.device_var.get() or "Auto").lower() in ("auto", "gpu") and (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"
            if (self.device_var.get() or "Auto").lower() == "gpu" and not (HAS_TORCH and torch is not None and torch.cuda.is_available()):
                self.write("! GPU requested for NLLB but no CUDA available — using CPU")
            try:
                self.tokenizer, self.nllb_model = load_nllb(device=nllb_device)
                self.write(f"✓ NLLB pripravený (device={nllb_device})")
            except Exception as e:
                self.write(f"NLLB load failed: {e}")
                messagebox.showerror("NLLB error", f"Failed to load NLLB model: {e}")
                return

            for i, file in enumerate(self.selected_files, 1):
                self._current_index = i
                self._current_file_start = time.time()

                self.write(f"\n[{i}/{total_files}] {os.path.basename(file)}")

                segments, info = transcribe_audio(self.model, file)
                detected = info.language or "en"
                src_nllb = NLLB_CODES.get(detected, "eng_Latn")

                self.write(f"Jazyk: {detected}")

                # prepare per-segment timing tracking for better ETA
                self._segment_timings = []
                self._last_segment_time = None
                def _seg_progress(idx, total):
                    now = time.time()
                    if self._last_segment_time is None:
                        # first segment start
                        self._last_segment_time = now
                    else:
                        seg_elapsed = now - self._last_segment_time
                        self._segment_timings.append(seg_elapsed)
                        self._last_segment_time = now

                    # compute avg segment time when possible
                    avg_seg = (sum(self._segment_timings) / len(self._segment_timings)) if self._segment_timings else None
                    # remaining segments for this file
                    rem_segs = max(0, total - idx)

                    # estimate remaining for current file
                    if avg_seg is not None:
                        rem_current = rem_segs * avg_seg
                    else:
                        # fallback heuristic based on per-file average or fixed small estimate
                        rem_current = max(3.0, (time.time() - (self._current_file_start or now)) * (rem_segs + 1))

                    # estimate remaining across remaining files using per-file average if known
                    per_file_avg = (sum(self._timings) / len(self._timings)) if self._timings else None
                    remaining_files = max(0, total_files - i)
                    rem_other = remaining_files * (per_file_avg if per_file_avg is not None else max(5.0, rem_current))

                    rem_total = rem_current + rem_other
                    finish = datetime.fromtimestamp(time.time() + rem_total).strftime("%H:%M:%S")
                    txt = f"ETA: {self._format_seconds(rem_total)} (finish at {finish}) — seg {idx}/{total}"
                    try:
                        self.root.after(0, lambda t=txt: self.eta_label.config(text=t))
                    except Exception:
                        pass

                translated = translate_segments(
                    self.tokenizer,
                    self.nllb_model,
                    segments,
                    src_nllb,
                    target_nllb,
                    progress_callback=_seg_progress
                )

                save_files(
                    os.path.splitext(file)[0],
                    segments,
                    translated,
                    target_code
                )

                file_elapsed = time.time() - self._current_file_start
                self._timings.append(file_elapsed)
                self._current_file_start = None

                # estimate remaining time based on average or heuristic if needed
                avg = (sum(self._timings) / len(self._timings)) if self._timings else None
                if avg is not None:
                    expected_total = avg
                else:
                    expected_total = max(10.0, file_elapsed * 1.5)

                remaining_files = max(0, total_files - i)
                rem = remaining_files * expected_total + max(0.0, expected_total - 0.0)

                # ensure 'finish' is always defined for linters (will be updated when rem >= 1)
                finish = 'now'
                if rem < 1:
                    eta_text = "ETA: <1s"
                else:
                    finish = datetime.fromtimestamp(time.time() + rem).strftime("%H:%M:%S")
                    eta_text = f"ETA: {self._format_seconds(rem)} (finish at {finish})"

                self.root.after(0, lambda txt=eta_text: self.eta_label.config(text=txt))

                self.write(f"✓ Hotovo — trvalo {self._format_seconds(file_elapsed)}; zostáva {self._format_seconds(rem)} (dokoncí o {finish if rem>=1 else 'now'})")

            self._processing = False
            finish_all = datetime.fromtimestamp(time.time()).strftime("%H:%M:%S")
            try:
                self.root.after(0, lambda f=finish_all: self.eta_label.config(text=f"Done — finished at {f}"))
            except Exception:
                pass

            messagebox.showinfo("Hotovo", "Všetko spracované")

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    TranslatorGUI(root)
    root.mainloop()
