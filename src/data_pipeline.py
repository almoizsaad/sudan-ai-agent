"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   خط أنابيب تجهيز بيانات اللهجة السودانية - نسخة آمنة قانونياً                 ║
║   Google Colab GPU/CPU Pipeline                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   ملاحظة مهمة:                                                                 ║
║   هذا السكربت لا يقوم بتحميل قنوات يوتيوب بالجملة (bulk channel scraping)،    ║
║   لأن ذلك يخالف حقوق الملكية الفكرية لأصحاب القنوات حتى لو كانت "عامة".        ║
║   بدلاً من ذلك، مصادر البيانات هنا هي:                                         ║
║     1) مجموعات بيانات مفتوحة ومرخّصة أصلاً للاستخدام البحثي/التطويري          ║
║        (Lisan-Sudanese, Sudanese Dialect Speech Corpus عبر HuggingFace)       ║
║     2) مقاطعك الخاصة أو أي مقطع حصلت على إذن صريح لاستخدامه — ضعها في         ║
║        مجلد manual_uploads/ وسيعالجها نفس الـ pipeline بنفس الجودة.           ║
║     3) (اختياري) قائمة روابط يوتيوب محددة بترخيص Creative Commons صراحة،      ║
║        تضيفها أنت يدوياً بعد التحقق من ترخيص كل فيديو بنفسك.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

المتطلبات:
    pip install datasets huggingface_hub librosa soundfile numpy tqdm torch torchaudio transformers onnxruntime yt-dlp

الاستخدام في Colab:
    1. ضع مقاطعك الخاصة/المرخّصة في مجلد manual_uploads/ (يُنشأ تلقائياً عند أول تشغيل)
    2. عدّل OPEN_DATASETS و(اختياري) CC_LICENSED_URLS أدناه حسب حاجتك
    3. غيّر MODE إلى "full" أو "fetch_open_only" أو "process_only"
    4. شغّل الخلية
"""

import os
import sys
import shutil
import json
import glob
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

# ─────────────────────────────────────────────────────────────
# التحقق من البيئة والأجهزة المتاحة
# ─────────────────────────────────────────────────────────────

def detect_device():
    """اكتشاف أفضل جهاز متاح: GPU > CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[Device] ✅ GPU detected: {torch.cuda.get_device_name(0)}")
            return "gpu"
    except ImportError:
        pass
    print("[Device] ⚠️ CPU only (consider switching to GPU runtime in Colab)")
    return "cpu"

DEVICE_TYPE = detect_device()

# ─────────────────────────────────────────────────────────────
# 1) CONFIGURATION
# ─────────────────────────────────────────────────────────────

# مجموعات بيانات مفتوحة مرخّصة على HuggingFace — عدّل/أضف حسب توفّرها فعلياً
# تحقق من اسم المجموعة الدقيق على huggingface.co/datasets قبل التشغيل، فالأسماء قد تتغير
OPEN_DATASETS = [
    # {"repo_id": "<org>/lisan-sudanese", "config": None, "split": "train"},
    # {"repo_id": "<org>/sudanese-dialect-speech-corpus", "config": None, "split": "train"},
]

# (اختياري) روابط يوتيوب فردية تحققت بنفسك أنها مرخّصة Creative Collections صراحة
# لا تضع رابط قناة كاملة هنا — فقط فيديوهات محددة تحققت من ترخيصها يدوياً
CC_LICENSED_URLS = [
    # "https://www.youtube.com/watch?v=XXXXXXXXXXX",
]

# مجلد ترفع فيه تسجيلاتك الخاصة أو أي مقطع حصلت على إذن كتابي لاستخدامه
MANUAL_UPLOADS_DIR = "/content/manual_uploads"

# مسارات الإخراج
RAW_DIR = "/content/01_Raw_Collected"
VAD_DIR = "/content/02_VAD_Filtered"
DIALECT_DIR = "/content/03_Dialect_Cleaned"
FINAL_DIR = "/content/04_Final_Dataset"

# إعدادات VAD
VAD_SAMPLE_RATE = 16000
VAD_MIN_SPEECH_DURATION = 0.5
VAD_MIN_SILENCE_DURATION = 0.3

# إعدادات تصنيف اللهجة
MSA_CONFIDENCE_THRESHOLD = 0.6
MIN_AUDIO_DURATION_SEC = 2.0

# إعدادات التوازي
MAX_WORKERS = min(mp.cpu_count(), 8)
BATCH_SIZE = 16


# ─────────────────────────────────────────────────────────────
# 2) COLLECTION PIPELINE — مصادر قانونية فقط
# ─────────────────────────────────────────────────────────────

def fetch_open_datasets(datasets_config=OPEN_DATASETS, output_dir=RAW_DIR):
    \"\"\"
    يجلب مجموعات بيانات صوتية عربية/سودانية مفتوحة من HuggingFace Hub
    ويحفظها محلياً كملفات wav جاهزة للمعالجة.
    \"\"\"
    if not datasets_config:
        print("[Open Datasets] ⚠️ لم تُحدَّد أي مجموعة بيانات في OPEN_DATASETS — تخطّي هذه الخطوة.")
        print("                 أضف repo_id الصحيح بعد التحقق منه على huggingface.co/datasets")
        return output_dir

    from datasets import load_dataset
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)

    for ds_cfg in datasets_config:
        repo_id = ds_cfg["repo_id"]
        config = ds_cfg.get("config")
        split = ds_cfg.get("split", "train")

        print(f"\\n[Open Datasets] 📥 تحميل {repo_id} (split={split})...")
        try:
            ds = load_dataset(repo_id, config, split=split, streaming=False)
        except Exception as err:
            print(f"[!] فشل تحميل {repo_id}: {err}")
            continue

        safe_name = repo_id.replace("/", "__")
        subdir = os.path.join(output_dir, safe_name)
        os.makedirs(subdir, exist_ok=True)

        saved = 0
        for i, row in enumerate(ds):
            audio_field = row.get("audio")
            if audio_field is None:
                continue
            try:
                array = audio_field["array"]
                sr = audio_field["sampling_rate"]
                out_path = os.path.join(subdir, f"{safe_name}_{i:06d}.wav")
                sf.write(out_path, array, sr)
                # احفظ أي نص مرافق إن وُجد، مفيد لاحقاً لتدريب/تقييم ASR أو كمرجع نصي
                text_field = row.get("text") or row.get("transcription") or row.get("sentence")
                if text_field:
                    with open(out_path.replace(".wav", ".txt"), "w", encoding="utf-8") as f:
                        f.write(str(text_field))
                saved += 1
            except Exception as err:
                print(f"[!] تخطّي عنصر {i} في {repo_id}: {err}")

        print(f"[Open Datasets] ✅ {repo_id}: {saved} ملف محفوظ في {subdir}")

    return output_dir


def collect_manual_uploads(input_dir=MANUAL_UPLOADS_DIR, output_dir=RAW_DIR):
    \"\"\"
    ينسخ أي ملفات صوتية وضعتها بنفسك في manual_uploads/
    (تسجيلاتك الخاصة، أو مقاطع حصلت على إذن كتابي صريح لاستخدامها)
    \"\"\"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
        audio_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not audio_files:
        print(f"[Manual Uploads] ⚠️ لا يوجد ملفات في {input_dir}")
        print("                  ضع فيه تسجيلاتك الخاصة أو مقاطع لديك إذن كتابي لاستخدامها.")
        return output_dir

    dest_subdir = os.path.join(output_dir, "manual_uploads")
    os.makedirs(dest_subdir, exist_ok=True)

    for f in audio_files:
        shutil.copy2(f, os.path.join(dest_subdir, os.path.basename(f)))

    print(f"[Manual Uploads] ✅ نُسخ {len(audio_files)} ملف من مصادرك الخاصة/المرخّصة")
    return output_dir


def fetch_cc_licensed_clips(urls=CC_LICENSED_URLS, output_dir=RAW_DIR):
    \"\"\"
    يحمّل فيديوهات فردية محددة فقط (وليس قنوات كاملة) — يجب أن تكون قد
    تحققت بنفسك مسبقاً أن كل رابط مرخّص Creative Commons أو لديك إذن صريح له.
    هذه الدالة تتعمّد عدم قبول رابط قناة كاملة — أدخل رابط فيديو واحد في كل مرة.
    \"\"\"
    if not urls:
        print("[CC Clips] ⚠️ لا توجد روابط في CC_LICENSED_URLS — تخطّي.")
        return output_dir

    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    dest_subdir = os.path.join(output_dir, "cc_licensed_clips")
    os.makedirs(dest_subdir, exist_ok=True)

    for url in urls:
        if "/channel/" in url or "/@" in url and "watch?v=" not in url:
            print(f"[!] تخطّي {url} — هذه الدالة تقبل روابط فيديو فردية فقط، وليس قنوات كاملة.")
            continue

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "outtmpl": f"{dest_subdir}/%(id)s.%(ext)s",
            "noplaylist": True,   # يمنع تحميل قائمة تشغيل كاملة بالخطأ
        }
        print(f"[CC Clips] 📥 {url}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as err:
            print(f"[!] خطأ في تحميل {url}: {err}")

    return output_dir


def run_collection(output_dir=RAW_DIR):
    \"\"\"يجمع من كل المصادر القانونية معاً\"\"\"
    print("=" * 60)
    print("[الخطوة 1] جمع البيانات من مصادر مرخّصة فقط")
    print("=" * 60)
    fetch_open_datasets(output_dir=output_dir)
    collect_manual_uploads(output_dir=output_dir)
    fetch_cc_licensed_clips(output_dir=output_dir)
    return output_dir


# ─────────────────────────────────────────────────────────────
# 3) VAD PIPELINE
# ─────────────────────────────────────────────────────────────

class FastVADProcessor:
    \"\"\"معالج VAD: يحمّل النموذج مرة واحدة، معالجة متوازية عبر ProcessPoolExecutor\"\"\"

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, use_onnx=True):
        if self._initialized:
            return
        self._initialized = True
        self.use_onnx = use_onnx
        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self):
        if self.use_onnx:
            try:
                import onnxruntime as ort
                model_path = self._download_onnx_model()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if DEVICE_TYPE == "gpu" else ["CPUExecutionProvider"]
                )
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.model_type = "onnx"
                print("[VAD] ✅ ONNX model loaded (fastest)")
                return
            except Exception as e:
                print(f"[!] ONNX failed: {e}, falling back to PyTorch")

        import torch
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            force_reload=False, onnx=False
        )
        self.model_type = "pytorch"
        print("[VAD] ✅ PyTorch model loaded")

    def _download_onnx_model(self):
        model_path = "silero_vad.onnx"
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
            print("[VAD] Downloading ONNX model...")
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def process_single(self, audio_path, output_path):
        import librosa
        import soundfile as sf
        try:
            wav, sr = librosa.load(audio_path, sr=VAD_SAMPLE_RATE, mono=True)

            segments = self._onnx_detect(wav, sr) if self.model_type == "onnx" else self._pytorch_detect(wav, sr)
            if not segments:
                return False, 0.0, audio_path

            speech_parts = [wav[int(s * sr):int(e * sr)] for s, e in segments]
            gap = np.zeros(int(0.15 * sr))
            combined = []
            for i, part in enumerate(speech_parts):
                combined.append(part)
                if i < len(speech_parts) - 1:
                    combined.append(gap)

            final_audio = np.concatenate(combined)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, final_audio, sr)
            return True, len(final_audio) / sr, audio_path
        except Exception as e:
            print(f"[!] Error processing {audio_path}: {e}")
            return False, 0.0, audio_path

    def _onnx_detect(self, wav, sr):
        wav = wav.astype(np.float32)
        window_size_samples = 512 if sr == 16000 else 256
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)

        speeches = []
        current_speech = None
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i:i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = np.pad(chunk, (0, window_size_samples - len(chunk)))
            inputs = {
                "input": chunk.reshape(1, -1), "h": h, "c": c,
                "sr": np.array([sr], dtype=np.int64),
            }
            out = self.session.run(None, inputs)
            prob, h, c = out[0][0][0], out[1], out[2]
            if prob > 0.5:
                if current_speech is None:
                    current_speech = {"start": i / sr}
            else:
                if current_speech is not None:
                    current_speech["end"] = i / sr
                    if current_speech["end"] - current_speech["start"] > VAD_MIN_SPEECH_DURATION:
                        speeches.append((current_speech["start"], current_speech["end"]))
                    current_speech = None
        if current_speech is not None:
            current_speech["end"] = len(wav) / sr
            speeches.append((current_speech["start"], current_speech["end"]))
        return speeches

    def _pytorch_detect(self, wav, sr):
        import torch
        wav_tensor = torch.tensor(wav, dtype=torch.float32)
        get_speech_ts = self.utils[0]
        timestamps = get_speech_ts(
            wav_tensor, self.model, sampling_rate=sr, threshold=0.5,
            min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
            return_seconds=True,
        )
        return [(ts["start"], ts["end"]) for ts in timestamps]


def _vad_worker(audio_path, output_path):
    processor = FastVADProcessor(use_onnx=True)
    return processor.process_single(audio_path, output_path)


def run_vad_parallel(input_dir=RAW_DIR, output_dir=VAD_DIR, max_workers=MAX_WORKERS):
    print("\\n" + "=" * 50)
    print("[الخطوة 2] VAD متوازي")
    print("=" * 50)

    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
        audio_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not audio_files:
        print("[!] لا يوجد ملفات — تأكد من تشغيل خطوة الجمع أولاً")
        return output_dir

    print(f"[*] {len(audio_files)} ملف | {max_workers} عمال متوازيين")

    tasks = []
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, input_dir)
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + "_vad.wav")
        tasks.append((audio_path, out_path))

    start_time = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_vad_worker, a, o): (a, o) for a, o in tasks
        }
        for future in as_completed(future_to_file):
            success, duration, orig_path = future.result()
            results.append((success, duration, orig_path))
            if success:
                print(f"  ✓ {os.path.basename(orig_path)} ({duration:.1f}s)")

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r[0])
    total_dur = sum(r[1] for r in results if r[0])
    print(f"\\n[*] ⏱️ VAD: {elapsed:.1f}s | {successful}/{len(audio_files)} نجح")
    print(f"[*] إجمالي مدة الكلام: {total_dur/3600:.2f} ساعة")
    return output_dir


# ─────────────────────────────────────────────────────────────
# 4) DIALECT CLASSIFICATION
# ─────────────────────────────────────────────────────────────

class FastDialectClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id="badrex/mms-300m-arabic-dialect-identifier"):
        if self._initialized:
            return
        self._initialized = True
        from transformers import pipeline
        self.classifier = pipeline(
            "audio-classification", model=model_id,
            device=0 if DEVICE_TYPE == "gpu" else -1,
        )
        print(f"[Dialect] ✅ Classifier loaded on {DEVICE_TYPE}")

    def classify_batch(self, audio_paths):
        results = []
        for path in audio_paths:
            try:
                preds = self.classifier(path)
                scores = {p["label"]: p["score"] for p in preds}
                msa_score = scores.get("MSA", 0.0)
                results.append({
                    "path": path,
                    "is_msa": msa_score >= MSA_CONFIDENCE_THRESHOLD,
                    "msa_confidence": msa_score,
                    "top_dialect": max(scores, key=scores.get),
                    "all_scores": scores,
                })
            except Exception as e:
                print(f"[!] Error classifying {path}: {e}")
                results.append(None)
        return results


def run_dialect_filter_parallel(input_dir=VAD_DIR, output_dir=DIALECT_DIR, batch_size=BATCH_SIZE):
    print("\\n" + "=" * 50)
    print("[الخطوة 3] تصفية اللهجة - معالجة دفعات")
    print("=" * 50)

    import librosa

    os.makedirs(output_dir, exist_ok=True)
    for sub in ("accepted", "rejected_msa", "rejected_other"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    if not audio_files:
        print("[!] لا يوجد ملفات")
        return output_dir

    print(f"[*] {len(audio_files)} ملف | حجم الدفعة: {batch_size}")
    classifier = FastDialectClassifier()

    accepted = rejected_msa = rejected_other = 0
    start_time = time.time()

    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        print(f"\\n  الدفعة {i // batch_size + 1}/{(len(audio_files) - 1) // batch_size + 1}")
        results = classifier.classify_batch(batch)

        for audio_path, result in zip(batch, results):
            if result is None:
                continue
            duration = librosa.get_duration(path=audio_path)
            if duration < MIN_AUDIO_DURATION_SEC:
                continue

            base_name = os.path.splitext(os.path.basename(audio_path))[0]

            if result["is_msa"]:
                out = os.path.join(output_dir, "rejected_msa", f"{base_name}_msa{result['msa_confidence']:.2f}.wav")
                shutil.copy2(audio_path, out)
                rejected_msa += 1
                print(f"    ✗ MSA: {base_name} ({result['msa_confidence']:.2f})")
            elif result["top_dialect"] in ("Egyptian", "Gulf", "Levantine", "Maghrebi") and \\
                    result["all_scores"].get(result["top_dialect"], 0) > 0.7:
                out = os.path.join(output_dir, "rejected_other", f"{base_name}_{result['top_dialect']}.wav")
                shutil.copy2(audio_path, out)
                rejected_other += 1
                print(f"    ✗ {result['top_dialect']}: {base_name}")
            else:
                rel = os.path.relpath(audio_path, input_dir)
                out = os.path.join(output_dir, "accepted", rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(audio_path, out)
                accepted += 1
                print(f"    ✓ ACCEPT: {base_name}")

    elapsed = time.time() - start_time
    print(f"\\n[*] ⏱️ تصفية: {elapsed:.1f}s")
    print(f"    ✓ مقبول: {accepted} | ✗ فصحى: {rejected_msa} | ✗ أخرى: {rejected_other}")
    return output_dir


# ─────────────────────────────────────────────────────────────
# 5) FINAL CLEANUP
# ─────────────────────────────────────────────────────────────

def finalize_dataset(input_dir=DIALECT_DIR, output_dir=FINAL_DIR):
    print("\\n" + "=" * 50)
    print("[الخطوة 4] التنظيف النهائي")
    print("=" * 50)

    import librosa

    accepted_dir = os.path.join(input_dir, "accepted")
    if not os.path.exists(accepted_dir):
        print("[!] لا يوجد ملفات مقبولة")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(accepted_dir, "**", "*.wav"), recursive=True)

    total_duration = 0.0
    file_stats = []

    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, accepted_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy2(audio_path, out_path)

        # انسخ أي نص مرافق (من مجموعات البيانات المفتوحة) إن وُجد
        txt_src = audio_path.replace("_vad.wav", ".txt")
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, out_path.replace(".wav", ".txt"))

        duration = librosa.get_duration(path=audio_path)
        total_duration += duration
        file_stats.append({
            "file": rel_path,
            "duration_sec": duration,
            "duration_min": duration / 60,
        })

    report = {
        "total_files": len(audio_files),
        "total_duration_hours": total_duration / 3600,
        "sources_note": "بيانات مجمّعة من مصادر مفتوحة مرخّصة، تسجيلات ذاتية، و/أو مقاطع بإذن صريح فقط",
        "files": file_stats,
    }

    with open(os.path.join(output_dir, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[*] ✅ النهائي: {len(audio_files)} ملف | {total_duration/3600:.2f} ساعة")
    return output_dir


# ─────────────────────────────────────────────────────────────
# 6) MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_full_pipeline():
    total_start = time.time()
    print("=" * 60)
    print("🚀 خط أنابيب آمن - Sudanese Dialect Dataset")
    print(f"⚡ Device: {DEVICE_TYPE.upper()} | Workers: {MAX_WORKERS}")
    print("=" * 60)

    run_collection()
    run_vad_parallel()
    run_dialect_filter_parallel()
    final_dir = finalize_dataset()

    total_elapsed = time.time() - total_start
    print("\\n" + "=" * 60)
    print(f"✅ اكتملت جميع الخطوات في {total_elapsed/60:.1f} دقيقة!")
    print(f"📁 المجلد النهائي: {final_dir}")
    print("=" * 60)


def run_process_only():
    \"\"\"يعالج فقط ما هو موجود بالفعل في RAW_DIR (بدون إعادة الجمع)\"\"\"
    if not os.path.exists(RAW_DIR):
        print(f"[!] {RAW_DIR} غير موجود — شغّل خطوة الجمع أولاً")
        return
    run_vad_parallel()
    run_dialect_filter_parallel()
    finalize_dataset()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODE = "full"  # ← غيّر هذا: "full" | "fetch_open_only" | "process_only"

    if MODE == "full":
        run_full_pipeline()
    elif MODE == "fetch_open_only":
        run_collection()
    elif MODE == "process_only":
        run_process_only()
    else:
        print(f"[!] وضع غير معروف: {MODE}")
