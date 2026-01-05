import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from faster_whisper import WhisperModel
from transformers import pipeline

# --------------------- NASTAVENIA ---------------------
model_name = "large-v3"
compute_type = "int8"
device = "cpu"
cpu_threads = os.cpu_count()

SUPPORTED_LANGUAGES = {
    "sk": "Slovenčina", "en": "Angličtina", "de": "Nemčina", "fr": "Francúzština",
    "cs": "Čeština", "pl": "Poľština", "es": "Španielčina", "it": "Taliančina",
    "ru": "Ruština", "hu": "Maďarčina", "uk": "Ukrajinčina", "pt": "Portugalčina",
    "nl": "Holandčina", "ja": "Japončina", "zh": "Čínština (zjednodušená)",
}

preferred_translation_model = "facebook/nllb-200-3.3B"
fallback_translation_model = "facebook/nllb-200-distilled-1.3B"
# ---------------------------------------------------

class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transkribér + Prekladač (large-v3) – Dark Mode")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e1e1e")  # Dark background

        # Dark mode štýl
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff")
        style.configure("TButton", background="#333333", foreground="#ffffff")
        style.map("TButton", background=[("active", "#444444")])
        style.configure("TCombobox", fieldbackground="#333333", background="#333333", foreground="#ffffff")
        style.configure("Treeview", background="#2d2d2d", fieldbackground="#2d2d2d", foreground="#ffffff")
        style.configure("Vertical.TScrollbar", background="#333333", troughcolor="#1e1e1e")

        self.model = None
        self.translator = None
        self.translation_model_used = "Žiadny"

        self.setup_ui()
        self.load_whisper_model()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Výber súborov
        ttk.Label(main_frame, text="Vyber súbory alebo priečinok:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0,8))
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0,15))
        self.file_label = ttk.Label(file_frame, text="Žiadne súbory vybrané", foreground="#aaaaaa")
        self.file_label.grid(row=0, column=0, sticky="w")
        ttk.Button(file_frame, text="Vybrať súbory", command=self.select_files).grid(row=0, column=1, padx=(15,8))
        ttk.Button(file_frame, text="Vybrať priečinok", command=self.select_folder).grid(row=0, column=2)

        # Cieľový jazyk
        ttk.Label(main_frame, text="Cieľový jazyk:", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(10,5))
        self.lang_var = tk.StringVar(value="sk")
        lang_combo = ttk.Combobox(main_frame, textvariable=self.lang_var, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly", width=15)
        lang_combo.grid(row=3, column=0, sticky="w", pady=(0,15))
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self.update_lang_name())

        self.lang_name_label = ttk.Label(main_frame, text=SUPPORTED_LANGUAGES["sk"], foreground="#00ddff")
        self.lang_name_label.grid(row=3, column=1, sticky="w", padx=(20,0))

        # Spustiť
        self.start_btn = ttk.Button(main_frame, text="Spustiť transkripciu a preklad", command=self.start_processing)
        self.start_btn.grid(row=4, column=0, columnspan=3, pady=20)
        self.start_btn.state(["disabled"])

        # Progress
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0,10))

        # Log
        ttk.Label(main_frame, text="Log výstup:", font=("Segoe UI", 10, "bold")).grid(row=6, column=0, sticky="nw", pady=(10,5))
        self.log_text = scrolledtext.ScrolledText(main_frame, height=18, bg="#2d2d2d", fg="#ffffff", insertbackground="#ffffff")
        self.log_text.grid(row=7, column=0, columnspan=3, sticky="nsew")
        main_frame.rowconfigure(7, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.selected_files = []

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_lang_name(self):
        code = self.lang_var.get()
        self.lang_name_label.config(text=SUPPORTED_LANGUAGES.get(code, code.upper()))

    def load_whisper_model(self):
        def load():
            self.log("Načítavam Whisper large-v3 model...")
            try:
                self.model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
                self.log("✓ Whisper model načítaný úspešne!")
                self.start_btn.state(["!disabled"])
            except Exception as e:
                self.log(f"✗ Chyba: {e}")
                messagebox.showerror("Chyba", f"Neopodarilo sa načítať Whisper:\n{e}")

        threading.Thread(target=load, daemon=True).start()

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio/Video", "*.mp3 *.wav *.m4a *.mp4 *.mkv *.avi *.mov *.webm")])
        if files:
            self.selected_files = list(files)
            self.file_label.config(text=f"Vybraných {len(files)} súborov", foreground="#ffffff")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            exts = (".mp3",".wav",".m4a",".mp4",".mkv",".avi",".mov",".webm")
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
            if files:
                self.selected_files = files
                self.file_label.config(text=f"Nájdených {len(files)} súborov", foreground="#ffffff")
            else:
                messagebox.showinfo("Info", "Žiadne podporované súbory.")

    def start_processing(self):
        if not self.selected_files:
            messagebox.showwarning("Pozor", "Vyber najprv súbory!")
            return

        target_code = self.lang_var.get()
        target_name = SUPPORTED_LANGUAGES.get(target_code, target_code.upper())

        def process():
            self.progress.start()
            self.start_btn.state(["disabled"])

           # Načítanie prekladača
            translator = None
            self.log("Načítavam NLLB prekladač...")
            try:
                translator = pipeline("translation", model=preferred_translation_model, device=-1, torch_dtype="auto")
                self.log("✓ Veľký 3.3B model načítaný!")
            except Exception as e:
                self.log(f"✗ 3.3B zlyhal ({e}), skúšam menší...")
                try:
                    translator = pipeline("translation", model=fallback_translation_model, device=-1, torch_dtype="auto")
                    self.log("✓ Distilled-1.3B načítaný.")
                except Exception as e2:
                    self.log(f"✗ Prekladač sa nepodarilo načítať: {e2}")

            total = len(self.selected_files)
            for idx, audio_file in enumerate(self.selected_files, 1):
                self.log(f"\n[{idx}/{total}] {os.path.basename(audio_file)}")
                try:
                    segments, info = self.model.transcribe(audio_file, language=None, beam_size=5, vad_filter=True)
                    segment_list = list(segments)
                    detected_lang = info.language.lower() if info.language else "en"

                    self.log(f"  Jazyk: {detected_lang.upper()} | Segmentov: {len(segment_list)}")
                    if len(segment_list) == 0:
                        self.log("  Žiadna reč – preskakujem.")
                        continue

                    # Pôvodná transkripcia s presnými časmi
                    original_lines = [f"[{s.start:.2f}s → {s.end:.2f}s] {s.text.strip()}" for s in segment_list]
                    full_original = " ".join([s.text.strip() for s in segment_list]).strip()

                    translated_lines = []

                    if detected_lang == target_code:
                        self.log(f"  Už v {target_name} – preklad nie je potrebný.")
                        translated_lines = original_lines
                    else:
                        self.log(f"  Prekladám CELÝ text do {target_name} naraz (najlepšia kvalita)...")
                        full_translated = full_original  # fallback

                        if translator:
                            try:
                                src_lang = f"{detected_lang}_Latn"
                                tgt_lang = f"{target_code}_Latn"

                                # Opravené volanie – bez max_length alebo s veľkým číslom
                                result = translator(
                                    full_original,
                                    src_lang=src_lang,
                                    tgt_lang=tgt_lang
                                )
                                full_translated = result[0]['translation_text'].strip()
                                self.log("  ✓ Preklad úspešný!")
                            except Exception as e:
                                self.log(f"  ✗ Preklad zlyhal ({e}) – použijem originál.")
                                full_translated = full_original
                        else:
                            self.log("  Prekladač nedostupný – použijem originál.")
                            full_translated = full_original

                        # Inteligentné rozdelenie preloženého textu na vety
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', full_translated)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        if not sentences:
                            sentences = [full_translated]

                        # Priradenie časov podľa pomeru pozície v texte
                        total_length = len(full_original)
                        char_pos = 0
                        sent_idx = 0

                        for seg in segment_list:
                            seg_text = seg.text.strip()
                            seg_len = len(seg_text)
                            if total_length > 0:
                                char_pos += seg_len
                                ratio = char_pos / total_length
                                sent_idx = min(int(ratio * len(sentences)), len(sentences) - 1)
                            translated_sentence = sentences[sent_idx] if sent_idx < len(sentences) else sentences[-1]
                            translated_lines.append(f"[{seg.start:.2f}s → {seg.end:.2f}s] {translated_sentence}")

                    # Uloženie súborov
                    base = os.path.splitext(audio_file)[0]
                    orig_file = base + "_original.txt"
                    trans_file = base + f"_{target_code}.txt"

                    with open(orig_file, "w", encoding="utf-8") as f:
                        f.write(full_original + "\n\n# S časmi:\n" + "\n".join(original_lines))

                    with open(trans_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(translated_lines))

                    self.log(f"  ✓ Hotovo! Uložené: {os.path.basename(orig_file)} a {os.path.basename(trans_file)}")

                except Exception as e:
                    self.log(f"  ✗ Nečakaná chyba: {e}")

            self.log("\n=== VŠETKO DOKONČENÉ ===")
            self.progress.stop()
            self.start_btn.state(["!disabled"])
            messagebox.showinfo("Hotovo", "Všetky súbory boli úspešne spracované!")

        threading.Thread(target=process, daemon=True).start()
if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()