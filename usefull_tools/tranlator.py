import whisper
import sys
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Načítanie modelu – vyber podľa svojho hardvéru:
# "tiny" alebo "base" = rýchly, menej presný (dobrý na testovanie)
# "small" = dobrý kompromis
# "medium" alebo "large" = veľmi presný, ale pomalší a viac pamäte (ideálne s GPU)
model_name = "large"  # Zmeň na "medium" alebo "large" pre lepšiu presnosť (aj pre slovenčinu)

print("Načítavam model Whisper...")
model = whisper.load_model(model_name)

# Cesta k audio súboru – buď ako parameter, alebo zadaj tu
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = input("Zadaj cestu k audio súboru (napr. nahraj.mp3): ")

print("Prepisujem audio na text...")
result = model.transcribe(audio_file)  # "sk" pre slovenčinu – model ju automaticky deteguje, ale toto zlepší presnosť

text = result["text"].strip()

print("\nTranskripcia:")
print(text)
from transformers import pipeline

translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
translated = translator(text, src_lang="eng_Latn", tgt_lang="slk_Latn")
print("\nPreklad do slovenčiny:")
print(translated[0]['translation_text'])

# Uloženie do súboru
output_file = audio_file.rsplit(".", 1)[0] + ".txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)



print(f"\nTranskripcia uložená do: {output_file}")