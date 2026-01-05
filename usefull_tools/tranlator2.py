import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from faster_whisper import WhisperModel
from transformers import pipeline

# Skrytie otravného FutureWarning z Transformers
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# --------------------- KONFIGURÁCIA A KONŠTANTY ---------------------
model_name = "large-v3"
compute_type = "int8"
device = "cpu"
cpu_threads = os.cpu_count()

# Správne NLLB jazykové kódy (ISO 639-3 + script)
NLLB_CODES = {
    "sk": "slk_Latn",   # slovenčina
    "en": "eng_Latn",   # angličtina
    "de": "deu_Latn",   # nemčina
    "fr": "fra_Latn",   # francúzština
    "cs": "ces_Latn",   # čeština
    "pl": "pol_Latn",   # poľština
    "es": "spa_Latn",   # španielčina
    "it": "ita_Latn",   # taliančina
    "ru": "rus_Cyrl",   # ruština
    "hu": "hun_Latn",   # maďarčina
    "uk": "ukr_Cyrl",   # ukrajinčina
    "pt": "por_Latn",   # portugalčina
    "nl": "nld_Latn",   # holandčina
    "ja": "jpn_Jpan",   # japončina
    "zh": "zho_Hans",   # čínština zjednodušená
}

SUPPORTED_LANGUAGES = {
    "sk": "Slovenčina", "en": "Angličtina", "de": "Nemčina", "fr": "Francúzština",
    "cs": "Čeština", "pl": "Poľština", "es": "Španielčina", "it": "Taliančina",
    "ru": "Ruština", "hu": "Maďarčina", "uk": "Ukrajinčina", "pt": "Portugalčina",
    "nl": "Holandčina", "ja": "Japončina", "zh": "Čínština (zjednodušená)",
}

preferred_translation_model = "facebook/nllb-200-3.3B"
fallback_translation_model = "facebook/nllb-200-distilled-1.3B"
# -----------------------------------------------------------------

# --------------------- FUNKCIE PRE LOGIKU A SPRACOVANIE ---------------------
def transcribe_audio(model, audio_file):
    """Spraví transkripciu z audio súboru (prvý krok)."""
    segments, info = model.transcribe(audio_file, language=None, beam_size=5, vad_filter=True)
    return list(segments), info

def prepare_original_text(segment_list):
    """Pripraví originálny transkript s časmi."""
    original_lines = [f"[{s.start:.2f}s → {s.end:.2f}s] {s.text.strip()}" for s in segment_list]
    full_original = " ".join([s.text.strip() for s in segment_list]).strip()
    return original_lines, full_original

def translate_text(translator, full_original, src_nllb, target_nllb):
    """Preloží celý text do cieľového jazyka (druhý krok)."""
    full_translated = full_original
    error = None
    if translator:
        try:
            result = translator(full_original, src_lang=src_nllb, tgt_lang=target_nllb)
            full_translated = result[0]['translation_text'].strip()
        except Exception as e:
            error = e
    return full_translated, error

def split_translated_text(full_translated):
    """Rozdelí preložený text na vety."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', full_translated)
    return [s.strip() for s in sentences if s.strip()]

def assign_translations_to_segments(segment_list, sentences, full_translated, full_original):
    """Priradí preložené vety k časovým segmentom bez opakovaní."""
    translated_lines = []
    sentence_count = len(sentences)
    if sentence_count == 0:
        sentences = [full_translated if full_translated else full_original]

    seg_idx = 0
    for sent_idx in range(sentence_count):
        if seg_idx < len(segment_list):
            seg = segment_list[seg_idx]
            translated_lines.append(f"[{seg.start:.2f}s → {seg.end:.2f}s] {sentences[sent_idx]}")
            seg_idx += 1

    while seg_idx < len(segment_list):
        seg = segment_list[seg_idx]
        translated_lines.append(f"[{seg.start:.2f}s → {seg.end:.2f}s] {sentences[-1]}")
        seg_idx += 1

    return translated_lines

def save_files(base, full_original, original_lines, translated_lines, target_code):
    """Uloží originálny a preložený transkript."""
    with open(base + "_original.txt", "w", encoding="utf-8") as f:
        f.write(full_original + "\n\n# S časmi:\n" + "\n".join(original_lines))
    with open(base + f"_{target_code}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))
# -----------------------------------------------------------------

class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transkribér + Prekladač (large-v3)")
        self.root.geometry("950x700")
        self.dark_mode = True
        self.model = None
        self.translator = None

        self.setup_ui()
        self.apply_theme()
        self.load_whisper_model()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Theme toggle
        self.theme_btn = ttk.Button(main_frame, text="🌙 Dark Mode", command=self.toggle_theme)
        self.theme_btn.grid(row=0, column=2, sticky="ne", pady=(0,10))

        # Výber súborov
        ttk.Label(main_frame, text="Vyber súbory alebo priečinok:", font=("Segoe UI", 11, "bold")).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0,10))
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0,20))
        self.file_label = ttk.Label(file_frame, text="Žiadne súbory vybrané", foreground="#888888")
        self.file_label.grid(row=0, column=0, sticky="w")
        ttk.Button(file_frame, text="Vybrať súbory", command=self.select_files).grid(row=0, column=1, padx=(20,10))
        ttk.Button(file_frame, text="Vybrať priečinok", command=self.select_folder).grid(row=0, column=2)

        # Cieľový jazyk
        ttk.Label(main_frame, text="Cieľový jazyk:", font=("Segoe UI", 11, "bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0,10))
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0,25))
        self.lang_var = tk.StringVar(value="sk")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly", width=10)
        lang_combo.grid(row=0, column=0)
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self.update_lang_name())

        self.lang_name_label = ttk.Label(lang_frame, text=SUPPORTED_LANGUAGES["sk"], font=("Segoe UI", 14, "bold"))
        self.lang_name_label.grid(row=0, column=1, padx=(20,0))

        # Spustiť
        self.start_btn = ttk.Button(main_frame, text="Spustiť transkripciu a preklad", command=self.start_processing)
        self.start_btn.grid(row=5, column=0, columnspan=3, pady=30)
        self.start_btn.state(["disabled"])

        # Progress
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0,15))

        # Log
        ttk.Label(main_frame, text="Log výstup:", font=("Segoe UI", 11, "bold")).grid(row=7, column=0, columnspan=3, sticky="nw", pady=(10,5))
        self.log_text = scrolledtext.ScrolledText(main_frame, height=20)
        self.log_text.grid(row=8, column=0, columnspan=3, sticky="nsew")
        main_frame.rowconfigure(8, weight=1)

        self.selected_files = []

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.theme_btn.config(text="🌙 Dark Mode" if self.dark_mode else "☀ Light Mode")

    def apply_theme(self):
        if self.dark_mode:
            bg = "#121212"
            fg = "#e0e0e0"
            entry_bg = "#1e1e1e"
            progress_color = "#4caf50"
            button_bg = "#333333"
        else:
            bg = "#f0f0f0"
            fg = "#000000"
            entry_bg = "#ffffff"
            progress_color = "#0078d7"
            button_bg = "#e0e0e0"

        self.root.configure(bg=bg)
        style = ttk.Style()
        style.configure(".", background=bg, foreground=fg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=button_bg, foreground=fg)
        style.configure("TCombobox", fieldbackground=entry_bg, foreground=fg)
        style.configure("Horizontal.TProgressbar", background=progress_color)
        self.log_text.config(bg=entry_bg, fg=fg, insertbackground=fg)
        self.file_label.config(foreground="#888888" if self.dark_mode else "#666666")
        self.lang_name_label.config(foreground="#00bfff" if self.dark_mode else "#0066cc")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_lang_name(self):
        code = self.lang_var.get()
        self.lang_name_label.config(text=SUPPORTED_LANGUAGES.get(code, code.upper()))

    def load_whisper_model(self):
        def load():
            self.log("Načítavam Whisper model...")
            try:
                self.model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
                self.log("✓ Whisper načítaný!")
                self.start_btn.state(["!disabled"])
            except Exception as e:
                self.log(f"✗ Chyba Whisper: {e}")
                messagebox.showerror("Chyba", str(e))

        threading.Thread(target=load, daemon=True).start()

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio/Video", "*.mp3 *.wav *.m4a *.mp4 *.mkv *.avi *.mov *.webm")])
        if files:
            self.selected_files = list(files)
            self.file_label.config(text=f"Vybraných {len(files)} súborov", foreground="#e0e0e0" if self.dark_mode else "#000000")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            exts = (".mp3",".wav",".m4a",".mp4",".mkv",".avi",".mov",".webm")
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
            if files:
                self.selected_files = files
                self.file_label.config(text=f"Nájdených {len(files)} súborov", foreground="#e0e0e0" if self.dark_mode else "#000000")
            else:
                messagebox.showinfo("Info", "Žiadne podporované súbory.")

    def start_processing(self):
        if not self.selected_files:
            messagebox.showwarning("Pozor", "Vyber súbory!")
            return

        target_code = self.lang_var.get()
        target_nllb = NLLB_CODES.get(target_code, "slk_Latn")
        target_name = SUPPORTED_LANGUAGES.get(target_code, target_code.upper())

        def process():
            self.progress.start()
            self.start_btn.state(["disabled"])

            translator = None
            self.log("Načítavam NLLB prekladač...")
            try:
                translator = pipeline("translation", model=fallback_translation_model, device=-1)
                self.log("✓ Distilled-1.3B model načítaný (stabilnejší).")
            except Exception as e:
                self.log(f"✗ Prekladač zlyhal: {e}")

            total = len(self.selected_files)
            for idx, audio_file in enumerate(self.selected_files, 1):
                self.log(f"\n[{idx}/{total}] {os.path.basename(audio_file)}")
                try:
                    segment_list, info = transcribe_audio(self.model, audio_file)
                    detected_lang = info.language.lower() if info.language else "en"
                    src_nllb = NLLB_CODES.get(detected_lang, "eng_Latn")

                    self.log(f"  Jazyk: {detected_lang.upper()} | Segmentov: {len(segment_list)}")
                    if len(segment_list) == 0:
                        self.log("  Žiadna reč.")
                        continue

                    original_lines, full_original = prepare_original_text(segment_list)

                    translated_lines = original_lines  # fallback

                    if detected_lang == target_code:
                        self.log(f"  Už v {target_name} – preklad nie je potrebný.")
                    else:
                        self.log(f"  Prekladám celý text do {target_name}...")
                        full_translated, error = translate_text(translator, full_original, src_nllb, target_nllb)
                        if error:
                            self.log(f"  ✗ Preklad zlyhal ({error}) – používam originál.")
                            full_translated = full_original

                        sentences = split_translated_text(full_translated)
                        translated_lines = assign_translations_to_segments(segment_list, sentences, full_translated, full_original)

                    save_files(os.path.splitext(audio_file)[0], full_original, original_lines, translated_lines, target_code)
                    self.log(f"  ✓ Uložené!")

                except Exception as e:
                    self.log(f"  ✗ Chyba: {e}")

            self.log("\n=== HOTOVO ===")
            self.progress.stop()
            self.start_btn.state(["!disabled"])
            messagebox.showinfo("Hotovo", "Spracované!")

        threading.Thread(target=process, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()