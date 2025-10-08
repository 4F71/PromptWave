# 🎧 PromptWave — AI Audio Generation via Prompt Engineering & Vibe Coding

**PromptWave** is a Python-based system that generates audio from text-based AI prompts — blending **prompt engineering** precision with the creative flow of **vibe coding**.  
The goal is not just to create sound, but to *capture emotion, atmosphere, and intent* through code.

---

## 🌐 TR Açıklama

### Genel Bakış
PromptWave, **prompt engineering** yaklaşımını **vibe coding** felsefesiyle birleştiren bir **Python ses üretim sistemidir**.  
Amaç yalnızca ses üretmek değil, her sesin arkasındaki duyguyu, enerjiyi ve atmosferi yakalayarak “yaratıcı frekanslar” oluşturabilmektir.  

Bu yapı, kullanıcıların kendi tarzlarını ve enerjilerini “prompt”lar aracılığıyla sese dönüştürmelerine olanak tanır.  
Proje, modüler mimarisiyle kolayca genişletilebilir ve deneysel ses üretimi için idealdir.

---

### Kullanılan Kütüphaneler
- `numpy` — Sayısal hesaplamalar ve sinyal işlemleri  
- `scipy` — DSP ve WAV işleme  
- `soundfile` — Ses dosyalarını okuma/yazma  
- `matplotlib` — Görselleştirme (opsiyonel, analiz için)  

### Tasarım ve Kullanıcı Kolaylığı
- **Modüler Yapı:** `sampler.py` ana çalışma dosyası, `prompts/` klasörü prompt yönetimi için ayrılmıştır.  
- **Otomatik Dosya Kaydı:** `output/` klasörü altında timestamp’e göre WAV dosyaları otomatik oluşturulur.  
- **Kolay Genişletme:** Yeni promptlar veya ses modülleri kolayca eklenebilir.  
- **Minimal Bağımlılık:** Sadece gerekli Python kütüphaneleri ile çalışır.  

### Klasör Yapısı (Özet)
- `sampler.py` → Ana Python kodu  
- `README.md` → Proje açıklaması  
- `requirements.txt` → Kullanılan kütüphaneler  
- `prompts/` → Ana ve draft promptlar (`main.txt`, `1_draft.txt`, `2_draft.txt`, `3_draft.txt`)  
- `output/` → Üretilen WAV dosyaları  

---

## 🌐 EN Description

### Overview
PromptWave merges **prompt engineering** with **vibe coding** — a creative approach to transforming written intent into expressive, atmospheric sound.  
This project goes beyond traditional code-based generation, focusing on emotional flow and sonic storytelling through prompts.

It provides a modular, user-friendly Python system for creative audio generation and experimentation.

---

### Libraries Used
- `numpy` — Numerical computation and signal processing  
- `scipy` — DSP and WAV file handling  
- `soundfile` — Reading/writing audio files  
- `matplotlib` — Visualization (optional, for analysis)  

### Design & Usability
- **Modular Structure:** `sampler.py` handles audio generation, `prompts/` manages all prompt files.  
- **Automatic File Saving:** Outputs are saved in the `output/` folder with timestamped filenames.  
- **Easy Extension:** New prompts or audio modules can be added easily.  
- **Minimal Dependencies:** Runs with only essential Python libraries.  

### Folder Structure (Summary)
- `sampler.py` → Main Python script  
- `README.md` → Project description  
- `requirements.txt` → Python dependencies  
- `prompts/` → Main and draft prompts (`main.txt`, `1_draft.txt`, `2_draft.txt`, `3_draft.txt`)  
- `output/` → Generated WAV files  

---

## 🚀 How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
