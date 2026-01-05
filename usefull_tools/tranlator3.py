import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import warnings

warnings.filterwarnings("ignore")

# ================== KONFIG ==================
WHISPER_MODEL = "large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = os.cpu_count()

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_CODES = {
    "sk": "slk_Latn",
    "cs": "ces_Latn",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "pl": "pol_Latn",
}

SUPPORTED_LANGUAGES = {
    "sk": "Slovenčina",
    "cs": "Čeština",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "pl": "Polski",
}
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


def load_nllb():
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    return tokenizer, model


def translate_segments(tokenizer, model, segments, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translated = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
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

    return translated


class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper + NLLB Translator (FIXED)")
        self.root.geometry("900x650")

        self.model = None
        self.tokenizer = None
        self.nllb_model = None
        self.selected_files = []

        self.setup_ui()
        self.load_whisper()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Button(frame, text="Vybrať súbory", command=self.select_files).pack()
        self.file_label = ttk.Label(frame, text="Žiadne súbory")
        self.file_label.pack(pady=5)

        ttk.Label(frame, text="Cieľový jazyk").pack()
        self.lang_var = tk.StringVar(value="sk")
        ttk.Combobox(
            frame,
            textvariable=self.lang_var,
            values=list(SUPPORTED_LANGUAGES.keys()),
            state="readonly"
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

    def load_whisper(self):
        def task():
            self.write("Načítavam Whisper...")
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS
            )
            self.write("✓ Whisper pripravený")
            self.start_btn.config(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def select_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Audio/Video", "*.mp3 *.wav *.m4a *.mp4 *.mkv *.avi")]
        )
        if files:
            self.selected_files = list(files)
            self.file_label.config(text=f"{len(files)} súborov")

    def start(self):
        if not self.selected_files:
            return

        def task():
            target_code = self.lang_var.get()
            target_nllb = NLLB_CODES[target_code]

            self.write("Načítavam NLLB...")
            self.tokenizer, self.nllb_model = load_nllb()
            self.write("✓ NLLB pripravený")

            for i, file in enumerate(self.selected_files, 1):
                self.write(f"\n[{i}/{len(self.selected_files)}] {os.path.basename(file)}")

                segments, info = transcribe_audio(self.model, file)
                detected = info.language or "en"
                src_nllb = NLLB_CODES.get(detected, "eng_Latn")

                self.write(f"Jazyk: {detected}")

                translated = translate_segments(
                    self.tokenizer,
                    self.nllb_model,
                    segments,
                    src_nllb,
                    target_nllb
                )

                save_files(
                    os.path.splitext(file)[0],
                    segments,
                    translated,
                    target_code
                )

                self.write("✓ Hotovo")

            messagebox.showinfo("Hotovo", "Všetko spracované")

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    TranslatorGUI(root)
    root.mainloop()
