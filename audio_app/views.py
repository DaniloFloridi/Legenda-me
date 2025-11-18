import tempfile
import os

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch


whisper = WhisperModel("medium", device="cpu", compute_type="int8")

marian_model_name = r"E:\Projeto Atualizado - Django Trial\models\opus-mt-en-pt"
tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
model = MarianMTModel.from_pretrained(marian_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def translate_text(text):

    batch = tokenizer([text], return_tensors="pt", padding=True).to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0]


def index(request):
    return render(request, "index.html")

@csrf_exempt
def transcribe_audio(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    print("📥 Received POST request…")

    audio_data = request.FILES.get("audio")
    if not audio_data:
        print("❌ No audio received!")
        return JsonResponse({"error": "No audio received"}, status=400)

    print("🎤 Audio received:", audio_data.size, "bytes")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        for chunk in audio_data.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    print("📁 Saved temp file:", tmp_path)

    segments, info = whisper.transcribe(tmp_path, language="en")
    transcript = " ".join([s.text for s in segments])
    print("📝 Transcript:", transcript)

    translation = translate_text(transcript)
    print("🌎 Translation:", translation)

    os.remove(tmp_path)

    return JsonResponse({
        "transcript": transcript,
        "translation": translation
    })
