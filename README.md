# 🎧 PromptWave — AI Audio Generation via Prompt Engineering

**PromptWave** is a Python-based project that generates audio from text-based AI prompts. It leverages **prompt engineering** to create high-quality, creative audio outputs while keeping user experience simple and intuitive.

---

## 🌐 TR Açıklama

### Genel Bakış
PromptWave, farklı taslak ve ana promptlardan ses üretmek için tasarlanmış bir **Python ses sampleri ve prompt yönetim sistemidir**. Projenin temel amacı, kullanıcıların metin tabanlı promptlar aracılığıyla yüksek kaliteli ses dosyaları üretebilmesini sağlamaktır.

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
PromptWave generates audio from text-based AI prompts using **prompt engineering**. It provides a modular, user-friendly Python system for creative audio generation.

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
