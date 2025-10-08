#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Audio DSP & AI-Assisted Sound Generator
Ultra-comprehensive noise, natural sound, brainwave, and mixing system
"""

import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import os  # <-- export_audio için eklendi

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: KULLANICI KONTROL PANELİ
# ═══════════════════════════════════════════════════════════════════════════

"""
MASTER TOGGLES TABLOSU
══════════════════════════════════════════════════════════════════════════════
Parametre                  | Açıklama                                    | Referans           | Örnek | Etki
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ENABLE_NOISE_GENERATOR     | Teknik gürültü üreticisini aktif eder       | True/False         | True  | Tüm noise türlerini üretir
ENABLE_NATURAL_SOUNDS      | Doğal ses üreticisini aktif eder            | True/False         | True  | Yağmur, rüzgar, okyanus vb.
ENABLE_MIXING_SYSTEM       | Çoklu katman karıştırıcıyı aktif eder       | True/False         | True  | Blogları birleştirir
ENABLE_FREQUENCY_FILTERS   | Frekans bazlı işlemleri aktif eder          | True/False         | True  | Boost, notch, bandpass
ENABLE_VISUALIZER          | Dalga formu ve spektrum görselini gösterir  | True/False         | True  | Matplotlib grafikleri
ENABLE_FILE_EXPORT         | WAV dosya çıktısı oluşturur                 | True/False         | True  | Timestamp'li WAV kaydı
"""

ENABLE_NOISE_GENERATOR = True
ENABLE_NATURAL_SOUNDS = True
ENABLE_MIXING_SYSTEM = True
ENABLE_FREQUENCY_FILTERS = False
ENABLE_VISUALIZER = True
ENABLE_FILE_EXPORT = True

"""
GENEL AYARLAR TABLOSU
══════════════════════════════════════════════════════════════════════════════
Parametre        | Açıklama                              | Referans Aralık    | Örnek  | Etki
──────────────────────────────────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE      | Örnekleme frekansı Hz cinsinden       | 8000-192000 Hz     | 44100  | Ses kalitesi ve frekans üst sınırı
DURATION         | Toplam ses süresi saniye cinsinden    | 1-3600 saniye      | 30     | Üretilen sesin uzunluğu
STEREO_MODE      | Stereo çıkış modu                     | True/False         | True   | Stereo veya mono çıkış
MASTER_AMPLITUDE | Ana çıkış ses seviyesi                | 0.0-1.0            | 0.7    | Genel ses yüksekliği
"""

SAMPLE_RATE = 44100
DURATION = 30
STEREO_MODE = True
MASTER_AMPLITUDE = 0.7

"""
NOISE TÜRÜ AKTIVASYON TABLOSU
══════════════════════════════════════════════════════════════════════════════
Tür      | Açıklama                                  | Frekans Davranışı     | Aktif | Kullanım
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
white    | Beyaz gürültü tüm frekanslarda eşit      | Düz spektrum          | True  | Maske, test sinyali
pink     | Pembe gürültü düşük frekans ağırlıklı    | 1/f azalma            | True  | Doğal ambiyans
brown    | Kahverengi gürültü çok düşük frekans     | 1/f² azalma           | True  | Derin bas, okyanus
blue     | Mavi gürültü yüksek frekans ağırlıklı    | f artışı              | False | Tiz vurgulu maske
violet   | Mor gürültü çok yüksek frekans           | f² artışı             | False | Ultra tiz enerji
gray     | Gri gürültü psiko-akustik düzleştirilmiş | Algısal düz           | True  | İnsan kulağına düz
green    | Yeşil gürültü 500Hz merkez ağırlıklı     | Orta frekans tepe     | False | Konuşma bandı maske
"""

noise_types = {
    "white": False,
    "pink": False,
    "brown": True,
    "blue": False,
    "violet": False,
    "gray": False,
    "green": True
}

"""
DOĞAL SES MIX KATMANLARI TABLOSU
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Katman    | Aktif | Ağırlık | Frekans Aralığı Hz   | Naturalness | Açıklama                           | Etki
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
rain      | T/F   | 0.0-1.0 | 400-2500             | 0.0-1.0     | Yağmur damlası doku                | Yüksek frekans taneler
thunder   | T/F   | 0.0-1.0 | 20-120               | 0.0-1.0     | Gök gürültüsü düşük frekans bas    | Dramatik bas vuruşları
wind      | T/F   | 0.0-1.0 | 100-800              | 0.0-1.0     | Rüzgar esintisi sürekli modülasyon | Orta-düşük süpürme
ocean     | T/F   | 0.0-1.0 | 30-500               | 0.0-1.0     | Okyanus dalgası düşük frekanslı    | Ritmik bas dalgalanma
fire      | T/F   | 0.0-1.0 | 800-5000             | 0.0-1.0     | Ateş çıtırtısı keskin transienter  | Yüksek frekans patlamalar
crickets  | T/F   | 0.0-1.0 | 3000-8000            | 0.0-1.0     | Cırcır böceği yüksek frekans ton   | Tiz periyodik sesler
car       | T/F   | 0.0-1.0 | 80-400               | 0.0-1.0     | Araba motor düşük frekans hum      | Monoton bas titreşim
train     | T/F   | 0.0-1.0 | 60-300               | 0.0-1.0     | Tren hareketi ritmik düşük frekans | Periyodik bas vuruş
vinyl     | T/F   | 0.0-1.0 | 200-4000             | 0.0-1.0     | Vinil çıtırtı geniş bantlı         | Retro analog doku
"""

noise_mix = {
    "rain": {
        "enabled": False,
        "weight": 0.6,
        "freq_range": (400, 2500),
        "naturalness": 0.2
    },
    "thunder": {
        "enabled": False,
        "weight": 0.8,
        "freq_range": (50, 500),
        "naturalness": 0.9
    },
    "wind": {
        "enabled": False,
        "weight": 0.5,
        "freq_range": (100, 800),
        "naturalness": 0.6
    },
    "ocean": {
        "enabled": False,
        "weight": 0.4,
        "freq_range": (100,500),
        "naturalness": 1.0
    },
    "fire": {
        "enabled": False,
        "weight": 0.3,
        "freq_range": (800, 5000),
        "naturalness": 0.75
    },
    "crickets": {
        "enabled": False,
        "weight": 0.4,
        "freq_range": (3000, 8000),
        "naturalness": 0.85
    },
    "car": {
        "enabled": False,
        "weight": 0.5,
        "freq_range": (80, 400),
        "naturalness": 0.5
    },
    "train": {
        "enabled": False,
        "weight": 0.6,
        "freq_range": (60, 300),
        "naturalness": 0.6
    },
    "vinyl": {
        "enabled": False,
        "weight": 0.3,
        "freq_range": (200, 4000),
        "naturalness": 1.0
    }
}

"""
NATURALNESS SİSTEMİ ULTRA GENİŞ PARAMETRE TABLOSU
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Parametre              | Açıklama                                      | Referans Aralık  | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
naturalness            | Ana gerçekçilik seviyesi                      | 0.0-1.0          | 0.7   | Tüm alt parametreleri ölçekler
randomness_amount      | Rastgele genlik varyasyonu miktarı            | 0.0-1.0          | 0.3   | Genlik sapması yüzdesi
freq_mod_depth         | Frekans modülasyon derinliği Hz               | 0.0-100.0 Hz     | 10.0  | Frekans kaydırma miktarı
freq_mod_rate          | Frekans modülasyon hızı Hz                    | 0.01-20.0 Hz     | 0.5   | Modülasyon dalgalanma hızı
amp_variation_amount   | Genlik varyasyon yüzdesi                      | 0.0-0.5          | 0.15  | Genlik değişim aralığı
grain_size             | Granüler sentez tane boyutu ms                | 5-200 ms         | 50    | Mikro ses parçacık boyutu
micro_timing_jitter    | Mikro zamanlama sapma miktarı ms              | 0.0-50.0 ms      | 5.0   | Temporal varyasyon
texture_layers         | Ek doku katman sayısı                         | 0-5              | 2     | Karmaşıklık katmanları
perlin_octaves         | Perlin gürültü oktav sayısı                   | 1-8              | 4     | Fraktal detay seviyesi
spectral_tilt          | Spektral eğim derecesi dB/oktav               | -12.0-12.0       | -3.0  | Frekans dengesizlik eğimi
"""

naturalness_params = {
    "randomness_amount": 0.3,
    "freq_mod_depth": 10.0,
    "freq_mod_rate": 0.5,
    "amp_variation_amount": 0.15,
    "grain_size": 50,
    "micro_timing_jitter": 5.0,
    "texture_layers": 2,
    "perlin_octaves": 4,
    "spectral_tilt": -3.0
}

"""
SPESIFIK FREKANS İŞLEMLERI TABLOSU
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Frekans Hz | İşlem Tipi  | Açıklama                           | Q Faktörü | Kazanç dB | Etki
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
452        | boost       | Frekans yükseltme                  | 0.5-10.0  | -24 - +24 | Belirli frekansı güçlendirir
1000       | notch       | Frekans kesme                      | 0.5-10.0  | -60 - 0   | Belirli frekansı zayıflatır
4320       | bandpass    | Bant geçiren filtre                | 0.5-10.0  | 0 - +12   | Sadece bu bandı geçirir
528        | synth_tone  | Sentetik ton ekleme                | N/A       | 0.0-1.0   | Saf sinüs tonu ekler
7830       | additive    | Harmonik sinüs ekleme              | N/A       | 0.0-1.0   | Ek harmonik katman
"""

specific_frequencies = [
    {"freq": 452, "operation": "boost", "q_factor": 2.0, "gain_db": 6.0},
    {"freq": 1000, "operation": "notch", "q_factor": 5.0, "gain_db": -20.0},
    {"freq": 4320, "operation": "bandpass", "q_factor": 1.5, "gain_db": 3.0},
    {"freq": 528, "operation": "synth_tone", "amplitude": 0.1},
    {"freq": 7830, "operation": "additive", "amplitude": 0.05}
]

"""
BRAINWAVE DALGALARI AKTIVASYON TABLOSU
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Dalga  | Aktif | Frekans Aralığı Hz | Merkez Hz | Açıklama                          | Psikolojik Etki
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
delta  | T/F   | 0.5-4.0            | 2.0       | Derin uyku delta dalgası          | Derin dinlenme restorasyon
theta  | T/F   | 4.0-8.0            | 6.0       | Meditasyon theta dalgası          | Yaratıcılık rüya hali
alpha  | T/F   | 8.0-13.0           | 10.0      | Rahatlatıcı alpha dalgası         | Sakin uyanıklık rahatlama
beta   | T/F   | 13.0-30.0          | 20.0      | Aktif düşünce beta dalgası        | Konsantrasyon dikkat
gamma  | T/F   | 30.0-100.0         | 40.0      | Yüksek bilişsel gamma dalgası     | Yüksek farkındalık bilgi işleme
"""

brainwave_config = {
    "delta": {"enabled": False, "freq_range": (0.5, 4.0), "center_freq": 2.0, "amplitude": 0.3, "mode": "tone"},
    "theta": {"enabled": False, "freq_range": (4.0, 8.0), "center_freq": 6.0, "amplitude": 0.3, "mode": "tone"},
    "alpha": {"enabled": False, "freq_range": (8.0, 13.0), "center_freq": 10.0, "amplitude": 0.4, "mode": "tone"},
    "beta": {"enabled": False, "freq_range": (13.0, 30.0), "center_freq": 20.0, "amplitude": 0.3, "mode": "tone"},
    "gamma": {"enabled": False, "freq_range": (30.0, 100.0), "center_freq": 40.0, "amplitude": 0.2, "mode": "tone"}
}

"""
MIX BLOG KONFIGÜRASYONU TABLOSU
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Blog Adı       | Ağırlık | Frekans Mapping           | Açıklama                              | Önerilen Kombinasyon
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
noise_white    | 0.0-1.0 | 20-20000 Hz (full)        | Beyaz gürültü tabanlı                 | Maske + doğal sesler
natural_rain   | 0.0-1.0 | 400-2500 Hz (high)        | Yağmur katmanı                        | Rain + thunder + alpha
natural_thunder| 0.0-1.0 | 20-120 Hz (low)           | Gök gürültüsü katmanı                 | Thunder + ocean + delta
brainwave_alpha| 0.0-1.0 | 8-13 Hz (sub-bass)        | Alpha dalgası katmanı                 | Alpha + rain + pink
spectral_low   | 0.0-1.0 | 20-250 Hz                 | Düşük frekans bölgesi                 | Thunder + brown + delta
spectral_mid   | 0.0-1.0 | 250-2000 Hz               | Orta frekans bölgesi                  | Rain + wind + alpha
spectral_high  | 0.0-1.0 | 2000-20000 Hz             | Yüksek frekans bölgesi                | Crickets + fire + beta
"""

mix_blog_config = {
    "layers": [
        {"blog": "rain", "weight": 0.6, "freq_map": "high"},
        {"blog": "thunder", "weight": 0.4, "freq_map": "low"},
        {"blog": "alpha", "weight": 0.3, "freq_map": "sub"}
    ],
    "spectral_mapping": {
        "low": (20, 250),
        "mid": (250, 2000),
        "high": (2000, 20000),
        "sub": (0.5, 100)
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: KATALOG (Noise ve Ses Tipleri Kataloğu)
# ═══════════════════════════════════════════════════════════════════════════

"""
KAPSAMLI SES VE GÜRÜLTÜ KATALOĞU
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Tür          | Kategori  | Teknik Tanım                                    | Frekans Aralığı Hz | Naturalness | Genlik
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
white        | Technical | Düz güç spektral yoğunluk                       | 20-20000           | 0.0-0.3     | 0.3-0.8
pink         | Technical | 1/f güç spektrumu azalma                        | 20-20000           | 0.2-0.5     | 0.3-0.8
brown        | Technical | 1/f² güç spektrumu dik azalma                   | 20-10000           | 0.3-0.6     | 0.3-0.8
blue         | Technical | f güç spektrumu artış yüksek frekans            | 100-20000          | 0.0-0.3     | 0.2-0.7
violet       | Technical | f² güç spektrumu dik artış ultra yüksek         | 500-20000          | 0.0-0.3     | 0.2-0.6
gray         | Technical | Psiko-akustik düzleştirilmiş algısal düz        | 20-20000           | 0.1-0.4     | 0.3-0.8
green        | Technical | 500Hz merkez gaussian bant boost                | 200-2000           | 0.1-0.4     | 0.3-0.7
rain         | Natural   | Yağmur damlası yüksek frekans stokastik tane    | 400-2500           | 0.5-1.0     | 0.4-0.9
thunder      | Natural   | Gök gürültüsü düşük frekans düzensiz patlama    | 20-120             | 0.7-1.0     | 0.5-1.0
wind         | Natural   | Rüzgar esintisi orta frekans sürekli modülasyon | 100-800            | 0.5-0.9     | 0.3-0.8
ocean        | Natural   | Okyanus dalgası düşük frekans ritmik süpürme    | 30-500             | 0.6-1.0     | 0.4-0.9
fire         | Natural   | Ateş çıtırtı yüksek frekans keskin transient    | 800-5000           | 0.6-0.9     | 0.3-0.7
crickets     | Natural   | Cırcır böceği tiz frekans periyodik ton         | 3000-8000          | 0.7-1.0     | 0.2-0.6
car          | Hybrid    | Araba motor düşük frekans harmonik hum          | 80-400             | 0.4-0.7     | 0.3-0.7
train        | Hybrid    | Tren hareketi ritmik düşük frekans vuruş        | 60-300             | 0.5-0.8     | 0.4-0.8
vinyl        | Hybrid    | Vinil çıtırtı geniş bant analog doku            | 200-4000           | 0.5-0.8     | 0.2-0.5
perlin       | Technical | Perlin fraktal gürültü çok oktav detay          | 20-10000           | 0.6-1.0     | 0.3-0.8
granular     | Technical | Granüler sentez mikro tane doku overlay         | 100-8000           | 0.7-1.0     | 0.2-0.6
delta        | Brainwave | Delta dalgası derin uyku restorasyon            | 0.5-4.0            | 0.0-0.5     | 0.2-0.5
theta        | Brainwave | Theta dalgası meditasyon yaratıcılık            | 4.0-8.0            | 0.0-0.5     | 0.2-0.5
alpha        | Brainwave | Alpha dalgası rahatlatma sakin uyanıklık        | 8.0-13.0           | 0.0-0.5     | 0.2-0.6
beta         | Brainwave | Beta dalgası konsantrasyon aktif düşünce        | 13.0-30.0          | 0.0-0.5     | 0.2-0.5
gamma        | Brainwave | Gamma dalgası yüksek bilişsel farkındalık       | 30.0-100.0         | 0.0-0.5     | 0.1-0.4
"""

CATALOG = {
    "white": {
        "category": "technical",
        "description": "Düz güç spektral yoğunluk tüm frekanslarda eşit enerji",
        "freq_range": (20, 20000),
        "naturalness_default": 0.1,
        "amplitude_range": (0.3, 0.8)
    },
    "pink": {
        "category": "technical",
        "description": "1/f güç spektrumu düşük frekans ağırlıklı doğal dağılım",
        "freq_range": (20, 20000),
        "naturalness_default": 0.3,
        "amplitude_range": (0.3, 0.8)
    },
    "brown": {
        "category": "technical",
        "description": "1/f² güç spektrumu çok düşük frekans dominant dik azalma",
        "freq_range": (20, 10000),
        "naturalness_default": 0.4,
        "amplitude_range": (0.3, 0.8)
    },
    "blue": {
        "category": "technical",
        "description": "f güç spektrumu yüksek frekans ağırlıklı artış",
        "freq_range": (100, 20000),
        "naturalness_default": 0.1,
        "amplitude_range": (0.2, 0.7)
    },
    "violet": {
        "category": "technical",
        "description": "f² güç spektrumu ultra yüksek frekans dominant dik artış",
        "freq_range": (500, 20000),
        "naturalness_default": 0.1,
        "amplitude_range": (0.2, 0.6)
    },
    "gray": {
        "category": "technical",
        "description": "Psiko-akustik düzleştirilmiş insan algısına düz spektrum",
        "freq_range": (20, 20000),
        "naturalness_default": 0.2,
        "amplitude_range": (0.3, 0.8)
    },
    "green": {
        "category": "technical",
        "description": "500Hz merkez gaussian bant boost konuşma bandı maske",
        "freq_range": (200, 2000),
        "naturalness_default": 0.2,
        "amplitude_range": (0.3, 0.7)
    },
    "rain": {
        "category": "natural",
        "description": "Yağmur damlası yüksek frekans stokastik tane doku",
        "freq_range": (400, 2500),
        "naturalness_default": 0.7,
        "amplitude_range": (0.4, 0.9)
    },
    "thunder": {
        "category": "natural",
        "description": "Gök gürültüsü düşük frekans düzensiz dramatik patlama",
        "freq_range": (20, 120),
        "naturalness_default": 0.9,
        "amplitude_range": (0.5, 1.0)
    },
    "wind": {
        "category": "natural",
        "description": "Rüzgar esintisi orta frekans sürekli modülasyon süpürme",
        "freq_range": (100, 800),
        "naturalness_default": 0.6,
        "amplitude_range": (0.3, 0.8)
    },
    "ocean": {
        "category": "natural",
        "description": "Okyanus dalgası düşük frekans ritmik süpürme dalgalanma",
        "freq_range": (30, 500),
        "naturalness_default": 0.8,
        "amplitude_range": (0.4, 0.9)
    },
    "fire": {
        "category": "natural",
        "description": "Ateş çıtırtı yüksek frekans keskin transient patlama",
        "freq_range": (800, 5000),
        "naturalness_default": 0.75,
        "amplitude_range": (0.3, 0.7)
    },
    "crickets": {
        "category": "natural",
        "description": "Cırcır böceği tiz frekans periyodik ton chirp",
        "freq_range": (3000, 8000),
        "naturalness_default": 0.85,
        "amplitude_range": (0.2, 0.6)
    },
    "car": {
        "category": "hybrid",
        "description": "Araba motor düşük frekans harmonik hum titreşim",
        "freq_range": (80, 400),
        "naturalness_default": 0.5,
        "amplitude_range": (0.3, 0.7)
    },
    "train": {
        "category": "hybrid",
        "description": "Tren hareketi ritmik düşük frekans periyodik vuruş",
        "freq_range": (60, 300),
        "naturalness_default": 0.6,
        "amplitude_range": (0.4, 0.8)
    },
    "vinyl": {
        "category": "hybrid",
        "description": "Vinil çıtırtı geniş bant analog retro doku",
        "freq_range": (200, 4000),
        "naturalness_default": 0.7,
        "amplitude_range": (0.2, 0.5)
    },
    "perlin": {
        "category": "technical",
        "description": "Perlin fraktal gürültü çok oktav detaylı doku",
        "freq_range": (20, 10000),
        "naturalness_default": 0.8,
        "amplitude_range": (0.3, 0.8)
    },
    "granular": {
        "category": "technical",
        "description": "Granüler sentez mikro tane doku overlay parçacık",
        "freq_range": (100, 8000),
        "naturalness_default": 0.8,
        "amplitude_range": (0.2, 0.6)
    },
    "delta": {
        "category": "brainwave",
        "description": "Delta dalgası derin uyku restorasyon 0.5-4Hz",
        "freq_range": (0.5, 4.0),
        "naturalness_default": 0.0,
        "amplitude_range": (0.2, 0.5)
    },
    "theta": {
        "category": "brainwave",
        "description": "Theta dalgası meditasyon yaratıcılık rüya 4-8Hz",
        "freq_range": (4.0, 8.0),
        "naturalness_default": 0.0,
        "amplitude_range": (0.2, 0.5)
    },
    "alpha": {
        "category": "brainwave",
        "description": "Alpha dalgası rahatlatma sakin uyanıklık 8-13Hz","freq_range": (8.0, 13.0),
        "naturalness_default": 0.0,
        "amplitude_range": (0.2, 0.6)
    },
    "beta": {
        "category": "brainwave",
        "description": "Beta dalgası konsantrasyon aktif düşünce 13-30Hz",
        "freq_range": (13.0, 30.0),
        "naturalness_default": 0.0,
        "amplitude_range": (0.2, 0.5)
    },
    "gamma": {
        "category": "brainwave",
        "description": "Gamma dalgası yüksek bilişsel farkındalık 30-100Hz",
        "freq_range": (30.0, 100.0),
        "naturalness_default": 0.0,
        "amplitude_range": (0.1, 0.4)
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def normalize_signal(sig, target_amplitude=1.0):
    """Sinyali normalize et ve clipping önle"""
    if np.max(np.abs(sig)) > 0:
        sig = sig / np.max(np.abs(sig)) * target_amplitude
    return np.clip(sig, -1.0, 1.0)


def apply_naturalness(sig, sr, naturalness, nat_params):
    """
    NATURALNESS PARAMETRELERİNİ UYGULA
    ══════════════════════════════════════════════════════════════════════════════
    Naturalness seviyesi 0.0-1.0 arası, tüm alt parametreleri ölçekler:
    
    0.0   -> Tamamen sentetik, hiç modülasyon yok
    0.25  -> Hafif rastgelelik, minimal freq/amp varyasyon
    0.5   -> Orta seviye, freq mod küçük, amp vary orta
    0.75  -> Yüksek seviye, freq mod orta, amp vary yüksek, grain eklenir
    1.0   -> Maksimum gerçekçilik, çoklu modülasyon, mikro jitter, granular
    """
    if naturalness <= 0.0:
        return sig
    
    result = sig.copy()
    n_samples = len(sig)
    t = np.arange(n_samples) / sr
    
    # Rastgele genlik varyasyonu
    if nat_params["randomness_amount"] > 0:
        random_amp = 1.0 + (np.random.randn(n_samples) * nat_params["randomness_amount"] * naturalness * 0.1)
        result *= random_amp
    
    # Frekans modülasyonu (pitch wobble)
    if nat_params["freq_mod_depth"] > 0 and nat_params["freq_mod_rate"] > 0:
        mod_signal = np.sin(2 * np.pi * nat_params["freq_mod_rate"] * t)
        freq_shift = mod_signal * nat_params["freq_mod_depth"] * naturalness
        phase_mod = np.cumsum(freq_shift) / sr
        result *= (1.0 + 0.01 * np.sin(2 * np.pi * phase_mod))
    
    # Genlik varyasyon envelope
    if nat_params["amp_variation_amount"] > 0:
        env_freq = 0.1 + np.random.rand() * 0.5
        envelope = 1.0 + nat_params["amp_variation_amount"] * naturalness * np.sin(2 * np.pi * env_freq * t)
        result *= envelope
    
    # Granüler doku overlay (yüksek naturalness'ta)
    if naturalness > 0.5 and nat_params["grain_size"] > 0:
        grain_samples = int(nat_params["grain_size"] * sr / 1000)
        n_grains = max(1, int(n_samples / (grain_samples / 2)))
        grain_layer = np.zeros(n_samples)
        
        for _ in range(min(n_grains, 100)):
            pos = np.random.randint(0, max(1, n_samples - grain_samples))
            grain_env = np.hanning(grain_samples)
            grain_noise = np.random.randn(grain_samples) * 0.1 * (naturalness - 0.5) * 2
            grain_layer[pos:pos+grain_samples] += grain_env * grain_noise
        
        result += grain_layer * 0.3
    
    # Mikro timing jitter (temporal varyasyon)
    if nat_params["micro_timing_jitter"] > 0 and naturalness > 0.6:
        jitter_amount = nat_params["micro_timing_jitter"] * naturalness / 1000.0
        jitter_samples = int(jitter_amount * sr)
        if jitter_samples > 0:
            shift = np.random.randint(-jitter_samples, jitter_samples + 1)
            result = np.roll(result, shift)
    
    # Perlin noise overlay (fraktal doku)
    if naturalness > 0.7 and nat_params["perlin_octaves"] > 0:
        perlin_layer = generate_perlin_noise(n_samples, nat_params["perlin_octaves"])
        result += perlin_layer * 0.05 * (naturalness - 0.7) * 3.33
    
    # Spektral tilt (frekans dengesi)
    if nat_params["spectral_tilt"] != 0.0:
        nyquist = sr / 2
        tilt_filter = sps.butter(4, [100 / nyquist, 0.95], btype='band', output='sos')
        tilted = sps.sosfilt(tilt_filter, result)
        tilt_factor = nat_params["spectral_tilt"] / 12.0 * naturalness
        result = result * (1 - abs(tilt_factor) * 0.3) + tilted * tilt_factor * 0.3
    
    return normalize_signal(result, np.max(np.abs(sig)) * 1.1)


def generate_perlin_noise(n_samples, octaves=4):
    """Basit Perlin-benzeri fraktal noise üretimi"""
    result = np.zeros(n_samples)
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(octaves):
        noise_len = max(2, int(n_samples / frequency))
        noise = np.random.randn(noise_len)
        noise_interp = np.interp(
            np.linspace(0, noise_len - 1, n_samples),
            np.arange(noise_len),
            noise
        )
        result += noise_interp * amplitude
        amplitude *= 0.5
        frequency *= 2.0
    
    return normalize_signal(result, 0.5)


def apply_bandpass_filter(sig, sr, freq_range):
    """Band-pass filtre uygula"""
    low, high = freq_range
    nyquist = sr / 2
    
    low_norm = max(0.001, min(low / nyquist, 0.999))
    high_norm = max(0.001, min(high / nyquist, 0.999))
    
    if low_norm >= high_norm:
        return sig
    
    sos = sps.butter(4, [low_norm, high_norm], btype='band', output='sos')
    return sps.sosfilt(sos, sig)


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: GÜRÜLTÜ ÜRETİCİ FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def generate_white_noise(duration, sr, amplitude=0.5):
    """Beyaz gürültü: düz spektrum, tüm frekanslarda eşit güç"""
    n_samples = int(duration * sr)
    noise = np.random.randn(n_samples)
    return normalize_signal(noise, amplitude)


def generate_pink_noise(duration, sr, amplitude=0.5):
    """Pembe gürültü: 1/f spektrum, düşük frekans ağırlıklı"""
    n_samples = int(duration * sr)
    white = np.random.randn(n_samples)
    
    # FFT tabanlı 1/f şekillendirme
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/sr)
    freqs[0] = 1.0
    
    pink_filter = 1.0 / np.sqrt(freqs)
    pink_filter[0] = pink_filter[1]
    
    pink_fft = fft * pink_filter
    pink = np.fft.irfft(pink_fft, n=n_samples)
    
    return normalize_signal(pink, amplitude)


def generate_brown_noise(duration, sr, amplitude=0.5):
    """Kahverengi gürültü: 1/f² spektrum, çok düşük frekans dominant"""
    n_samples = int(duration * sr)
    white = np.random.randn(n_samples)
    
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/sr)
    freqs[0] = 1.0
    
    brown_filter = 1.0 / freqs
    brown_filter[0] = brown_filter[1]
    
    brown_fft = fft * brown_filter
    brown = np.fft.irfft(brown_fft, n=n_samples)
    
    return normalize_signal(brown, amplitude)


def generate_blue_noise(duration, sr, amplitude=0.5):
    """Mavi gürültü: f spektrum, yüksek frekans ağırlıklı"""
    n_samples = int(duration * sr)
    white = np.random.randn(n_samples)
    
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/sr)
    freqs[0] = 1.0
    
    blue_filter = np.sqrt(freqs)
    blue_fft = fft * blue_filter
    blue = np.fft.irfft(blue_fft, n=n_samples)
    
    return normalize_signal(blue, amplitude)


def generate_violet_noise(duration, sr, amplitude=0.5):
    """Mor gürültü: f² spektrum, ultra yüksek frekans dominant"""
    n_samples = int(duration * sr)
    white = np.random.randn(n_samples)
    
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/sr)
    freqs[0] = 1.0
    
    violet_filter = freqs
    violet_fft = fft * violet_filter
    violet = np.fft.irfft(violet_fft, n=n_samples)
    
    return normalize_signal(violet, amplitude)


def generate_gray_noise(duration, sr, amplitude=0.5):
    """Gri gürültü: psiko-akustik düzleştirilmiş, insan algısına düz"""
    n_samples = int(duration * sr)
    pink = generate_pink_noise(duration, sr, 1.0)
    
    # Equal-loudness kontur yaklaşımı (basitleştirilmiş)
    nyquist = sr / 2
    sos = sps.butter(2, [0.1, 0.9], btype='band', output='sos')
    gray = sps.sosfilt(sos, pink)
    
    return normalize_signal(gray, amplitude)


def generate_green_noise(duration, sr, amplitude=0.5):
    """Yeşil gürültü: 500Hz merkez gaussian boost"""
    n_samples = int(duration * sr)
    white = np.random.randn(n_samples)
    
    # 500Hz civarında gaussian boost
    nyquist = sr / 2
    center_norm = 500 / nyquist
    
    if center_norm < 0.999:
        sos = sps.butter(4, [max(0.001, center_norm - 0.3), min(0.999, center_norm + 0.3)], btype='band', output='sos')
        green = sps.sosfilt(sos, white)
        return normalize_signal(green, amplitude)
    
    return normalize_signal(white, amplitude)


def generate_noise(noise_type, duration, sr, amplitude=0.5):
    """Ana gürültü üretim fonksiyonu"""
    generators = {
        "white": generate_white_noise,
        "pink": generate_pink_noise,
        "brown": generate_brown_noise,
        "blue": generate_blue_noise,
        "violet": generate_violet_noise,
        "gray": generate_gray_noise,
        "green": generate_green_noise
    }
    
    if noise_type in generators:
        return generators[noise_type](duration, sr, amplitude)
    else:
        return generate_white_noise(duration, sr, amplitude)


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: SES BLOKLARI (SOUND BLOGS)
# ═══════════════════════════════════════════════════════════════════════════

"""
RAIN BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
density            | Yağmur damlası yoğunluğu           | 0.1-1.0         | 0.7   | Daha fazla tane/damla
drop_freq_center   | Damla merkez frekansı Hz           | 400-2500        | 1200  | Ortalama damla tiz seviyesi
drop_freq_variance | Damla frekans dağılımı Hz          | 50-500          | 300   | Frekans çeşitliliği
impact_sharpness   | Damla vuruş keskinliği             | 0.0-1.0         | 0.6   | Transient sertliği
"""

def sound_blog_rain(duration, sr, amplitude=0.5, naturalness=0.7, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    rain = np.zeros(n_samples)
    
    # Yağmur parametreleri
    density = 0.7
    drop_freq_center = 1200
    drop_freq_variance = 300
    impact_sharpness = 0.6
    
    # Yağmur damlaları oluştur
    n_drops = int(density * duration * 200)
    
    for _ in range(n_drops):
        pos = np.random.randint(0, n_samples)
        drop_freq = drop_freq_center + np.random.randn() * drop_freq_variance
        drop_freq = np.clip(drop_freq, 400, 2500)
        
        # Damla envelope
        drop_len = int(sr * 0.02 * (1.0 + np.random.rand()))
        drop_len = min(drop_len, n_samples - pos)
        
        if drop_len > 0:
            t_drop = np.arange(drop_len) / sr
            decay = np.exp(-t_drop * (20 + impact_sharpness * 30))
            drop_tone = np.sin(2 * np.pi * drop_freq * t_drop) * decay
            rain[pos:pos+drop_len] += drop_tone * np.random.rand()
    
    # Arka plan gürültü katmanı
    background = generate_pink_noise(duration, sr, amplitude * 0.3)
    background = apply_bandpass_filter(background, sr, (400, 2500))
    
    rain = rain + background
    rain = normalize_signal(rain, amplitude)
    
    # Naturalness uygula
    rain = apply_naturalness(rain, sr, naturalness, nat_params)
    
    return rain


"""
THUNDER BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
rumble_freq        | Gürültü merkez frekansı Hz         | 20-120          | 60    | Bas gürültü tonu
strike_intensity   | Şimşek vuruş yoğunluğu             | 0.0-1.0         | 0.8   | Dramatik patlama gücü
decay_time         | Gürültü azalma süresi saniye       | 1.0-5.0         | 2.5   | Uzun kuyruk süresi
rumble_variation   | Gürültü frekans varyasyonu         | 0.0-1.0         | 0.5   | Pitch değişimi
"""

def sound_blog_thunder(duration, sr, amplitude=0.7, naturalness=0.9, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    thunder = np.zeros(n_samples)
    
    # Thunder parametreleri
    rumble_freq = 60
    strike_intensity = 0.8
    decay_time = 2.5
    rumble_variation = 0.5
    
    # Birkaç thunder strike oluştur
    n_strikes = int(duration / 5) + 1
    
    for _ in range(n_strikes):
        strike_pos = np.random.randint(0, max(1, n_samples - int(decay_time * sr)))
        
        # Strike uzunluğu
        strike_len = int(decay_time * sr)
        strike_len = min(strike_len, n_samples - strike_pos)
        
        t_strike = np.arange(strike_len) / sr
        
        # Frekans modülasyonu
        freq_mod = rumble_freq * (1.0 + rumble_variation * np.sin(2 * np.pi * 0.5 * t_strike))
        phase = np.cumsum(freq_mod) / sr
        
        # Exponential decay envelope
        envelope = np.exp(-t_strike / decay_time) * strike_intensity
        
        # Bas ton + gürültü
        tone = np.sin(2 * np.pi * phase) * envelope
        noise = np.random.randn(strike_len) * envelope * 0.3
        
        strike_signal = tone + noise
        thunder[strike_pos:strike_pos+strike_len] += strike_signal
    
    # Düşük frekans filtreleme
    thunder = apply_bandpass_filter(thunder, sr, (20, 120))
    thunder = normalize_signal(thunder, amplitude)
    
    # Naturalness uygula
    thunder = apply_naturalness(thunder, sr, naturalness, nat_params)
    
    return thunder


"""
WIND BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
gust_frequency     | Rüzgar patlaması frekansı Hz       | 0.05-0.5        | 0.15  | Rüzgar vuruş hızı
wind_intensity     | Rüzgar şiddeti                     | 0.0-1.0         | 0.6   | Genel güç seviyesi
freq_sweep_range   | Frekans süpürme aralığı Hz         | 100-800         | (100,800) | Pitch değişim bandı
modulation_depth   | Modülasyon derinliği               | 0.0-1.0         | 0.7   | Pitch dalgalanma
"""

def sound_blog_wind(duration, sr, amplitude=0.6, naturalness=0.6, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Wind parametreleri
    gust_frequency = 0.15
    wind_intensity = 0.6
    modulation_depth = 0.7
    
    # Temel gürültü
    wind = generate_pink_noise(duration, sr, wind_intensity)
    
    # Frekans bandı
    wind = apply_bandpass_filter(wind, sr, (100, 800))
    
    # Gust modülasyonu (rüzgar patlamaları)
    gust_lfo = (1.0 + np.sin(2 * np.pi * gust_frequency * t)) / 2.0
    gust_env = 0.5 + gust_lfo * modulation_depth * 0.5
    
    wind *= gust_env
    wind = normalize_signal(wind, amplitude)
    
    # Naturalness uygula
    wind = apply_naturalness(wind, sr, naturalness, nat_params)
    
    return wind


"""
OCEAN BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
wave_frequency     | Dalga frekansı Hz                  | 0.05-0.3        | 0.12  | Dalga tekrar hızı
wave_depth         | Dalga derinliği                    | 0.0-1.0         | 0.8   | Dalga amplitüd gücü
foam_amount        | Köpük miktarı                      | 0.0-1.0         | 0.4   | Yüksek frekans köpük
tide_variation     | Gel-git varyasyonu                 | 0.0-1.0         | 0.3   | Uzun dönemli değişim
"""

def sound_blog_ocean(duration, sr, amplitude=0.7, naturalness=0.8, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Ocean parametreleri
    wave_frequency = 0.12
    wave_depth = 0.8
    foam_amount = 0.4
    tide_variation = 0.3
    
    # Temel dalga gürültüsü
    ocean = generate_brown_noise(duration, sr, wave_depth)
    ocean = apply_bandpass_filter(ocean, sr, (30, 500))
    
    # Dalga envelope (ritmik dalgalanma)
    wave_envelope = (1.0 + np.sin(2 * np.pi * wave_frequency * t)) / 2.0
    wave_envelope = 0.6 + wave_envelope * 0.4
    
    # Gel-git modulasyonu (çok yavaş)
    tide_lfo = np.sin(2 * np.pi * 0.02 * t) * tide_variation
    wave_envelope *= (1.0 + tide_lfo)
    
    ocean *= wave_envelope
    
    # Köpük katmanı (yüksek frekans)
    if foam_amount > 0:
        foam = generate_white_noise(duration, sr, foam_amount * 0.3)
        foam = apply_bandpass_filter(foam, sr, (800, 3000))
        foam *= wave_envelope ** 2
        ocean += foam
    
    ocean = normalize_signal(ocean, amplitude)
    
    # Naturalness uygula
    ocean = apply_naturalness(ocean, sr, naturalness, nat_params)
    
    return ocean


"""
FIRE BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
crackle_density    | Çıtırtı yoğunluğu                  | 0.1-1.0         | 0.6   | Çıtırtı sayısı
pop_intensity      | Patlama şiddeti                    | 0.0-1.0         | 0.7   | Keskin transient gücü
flame_roar         | Alev uğultusu miktarı              | 0.0-1.0         | 0.4   | Arka plan uğultu
ember_glow         | Kor parıltı düşük frekans          | 0.0-1.0         | 0.3   | Düşük frekans vurgu
"""

def sound_blog_fire(duration, sr, amplitude=0.6, naturalness=0.75, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    fire = np.zeros(n_samples)
    
    # Fire parametreleri
    crackle_density = 0.6
    pop_intensity = 0.7
    flame_roar = 0.4
    
    # Crackle/pop olayları
    n_crackles = int(crackle_density * duration * 30)
    
    for _ in range(n_crackles):
        pos = np.random.randint(0, n_samples)
        
        # Crackle uzunluğu
        crackle_len = int(sr * (0.01 + np.random.rand() * 0.05))
        crackle_len = min(crackle_len, n_samples - pos)
        
        if crackle_len > 0:
            t_crackle = np.arange(crackle_len) / sr
            
            # Keskin decay
            envelope = np.exp(-t_crackle * (30 + np.random.rand() * 50))
            
            # Yüksek frekans burst
            freq = 1500 + np.random.rand() * 3000
            crackle = np.random.randn(crackle_len) * envelope * pop_intensity
            
            fire[pos:pos+crackle_len] += crackle
    
    # Alev uğultusu arka planı
    if flame_roar > 0:
        roar = generate_pink_noise(duration, sr, flame_roar * 0.5)
        roar = apply_bandpass_filter(roar, sr, (200, 2000))
        fire += roar
    
    # Yüksek frekans filtreleme
    fire = apply_bandpass_filter(fire, sr, (800, 5000))
    fire = normalize_signal(fire, amplitude)
    
    # Naturalness uygula
    fire = apply_naturalness(fire, sr, naturalness, nat_params)
    
    return fire


"""
CRICKETS BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
chirp_rate         | Cırcır frekansı Hz                 | 1.0-10.0        | 3.0   | Cırcır tekrar hızı
cricket_count      | Cırcır böceği sayısı               | 1-20            | 8     | Eşzamanlı cırcır sayısı
pitch_center       | Ton merkez frekansı Hz             | 3000-8000       | 5000  | Ortalama cırcır tiz tonu
pitch_variation    | Ton varyasyonu Hz                  | 100-1000        | 500   | Cırcır arası fark
"""

def sound_blog_crickets(duration, sr, amplitude=0.5, naturalness=0.85, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    crickets = np.zeros(n_samples)
    
    # Cricket parametreleri
    chirp_rate = 3.0
    cricket_count = 8
    pitch_center = 5000
    pitch_variation = 500
    
    # Her cırcır böceği için
    for _ in range(cricket_count):
        cricket_pitch = pitch_center + (np.random.rand() - 0.5) * pitch_variation * 2
        chirp_period = sr / chirp_rate * (0.8 + np.random.rand() * 0.4)
        
        n_chirps = int(duration * chirp_rate * (0.8 + np.random.rand() * 0.4))
        
        for _ in range(n_chirps):
            chirp_pos = np.random.randint(0, max(1, n_samples - int(chirp_period)))
            chirp_len = int(0.05 * sr)
            chirp_len = min(chirp_len, n_samples - chirp_pos)
            
            if chirp_len > 0:
                t_chirp = np.arange(chirp_len) / sr
                envelope = np.sin(np.pi * t_chirp / (chirp_len / sr)) ** 2
                chirp_tone = np.sin(2 * np.pi * cricket_pitch * t_chirp) * envelope
                crickets[chirp_pos:chirp_pos+chirp_len] += chirp_tone * 0.3
    
    crickets = normalize_signal(crickets, amplitude)
    
    # Naturalness uygula
    crickets = apply_naturalness(crickets, sr, naturalness, nat_params)
    
    return crickets


"""
CAR BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
engine_rpm         | Motor devir hızı RPM               | 800-3000        | 1500  | Temel motor frekansı
harmonic_count     | Harmonik sayısı                    | 3-10            | 5     | Motor ton karmaşıklığı
vibration_amount   | Titreşim miktarı                   | 0.0-1.0         | 0.5   | Düzensiz titreşim
road_noise         | Yol gürültüsü seviyesi             | 0.0-1.0         | 0.3   | Arka plan yol sesi
"""

def sound_blog_car(duration, sr, amplitude=0.6, naturalness=0.5, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Car parametreleri
    engine_rpm = 1500
    harmonic_count = 5
    vibration_amount = 0.5
    road_noise = 0.3
    
    # Motor temel frekansı (RPM'den Hz'e)
    base_freq = engine_rpm / 60.0
    
    car = np.zeros(n_samples)
    
    # Harmonikler
    for h in range(1, harmonic_count + 1):
        harmonic_freq = base_freq * h
        harmonic_amp = 1.0 / h
        car += np.sin(2 * np.pi * harmonic_freq * t) * harmonic_amp
    
    # Vibrasyon modülasyonu
    if vibration_amount > 0:
        vib_lfo = np.sin(2 * np.pi * 5.0 * t) * vibration_amount * 0.1
        car *= (1.0 + vib_lfo)
    
    # Yol gürültüsü
    if road_noise > 0:
        road = generate_pink_noise(duration, sr, road_noise * 0.4)
        road = apply_bandpass_filter(road, sr, (100, 500))
        car += road
    
    # Frekans bandı
    car = apply_bandpass_filter(car, sr, (80, 400))
    car = normalize_signal(car, amplitude)
    
    # Naturalness uygula
    car = apply_naturalness(car, sr, naturalness, nat_params)
    
    return car


"""
TRAIN BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
wheel_rhythm       | Tekerlek vuruş ritmi Hz            | 1.0-4.0         | 2.5   | Clickety-clack hızı
rail_rumble        | Ray uğultusu seviyesi              | 0.0-1.0         | 0.7   | Düşük frekans titreşim
mechanical_clank   | Mekanik şakırtı miktarı            | 0.0-1.0         | 0.5   | Metalik vuruş sesi
speed_variation    | Hız varyasyonu                     | 0.0-0.3         | 0.1   | Tempo değişimi
"""

def sound_blog_train(duration, sr, amplitude=0.7, naturalness=0.6, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    train = np.zeros(n_samples)
    
    # Train parametreleri
    wheel_rhythm = 2.5
    rail_rumble = 0.7
    mechanical_clank = 0.5
    
    # Tekerlek vuruşları (ritmik)
    click_period = sr / wheel_rhythm
    n_clicks = int(duration * wheel_rhythm)
    
    for i in range(n_clicks):
        click_pos = int(i * click_period)
        if click_pos < n_samples:
            click_len = int(0.05 * sr)
            click_len = min(click_len, n_samples - click_pos)
            
            if click_len > 0:
                t_click = np.arange(click_len) / sr
                envelope = np.exp(-t_click * 40)
                
                # Metalik ses
                click = np.random.randn(click_len) * envelope * mechanical_clank
                train[click_pos:click_pos+click_len] += click
    
    # Ray uğultusu (düşük frekans sürekli)
    if rail_rumble > 0:
        rumble = generate_brown_noise(duration, sr, rail_rumble * 0.6)
        rumble = apply_bandpass_filter(rumble, sr, (60, 300))
        train += rumble
    
    train = normalize_signal(train, amplitude)
    
    # Naturalness uygula
    train = apply_naturalness(train, sr, naturalness, nat_params)
    
    return train


"""
VINYL BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
crackle_density    | Çıtırtı yoğunluğu                  | 0.1-1.0         | 0.5   | Analog çıtırtı miktarı
pop_frequency      | Pop frekansı saniye başına         | 0.1-5.0         | 1.0   | Büyük pop sayısı
dust_noise         | Toz gürültü seviyesi               | 0.0-1.0         | 0.3   | Sürekli arka plan
warmth_amount      | Analog sıcaklık miktarı            | 0.0-1.0         | 0.6   | Düşük frekans vurgu
"""

def sound_blog_vinyl(duration, sr, amplitude=0.4, naturalness=0.7, nat_params=None):
    if nat_params is None:
        nat_params = naturalness_params
    
    n_samples = int(duration * sr)
    vinyl = np.zeros(n_samples)
    
    # Vinyl parametreleri
    crackle_density = 0.5
    pop_frequency = 1.0
    dust_noise = 0.3
    
    # Küçük crackle'lar (sürekli)
    n_crackles = int(crackle_density * duration * 100)
    
    for _ in range(n_crackles):
        pos = np.random.randint(0, n_samples)
        crackle_len = int(sr * 0.002 * (1 + np.random.rand()))
        crackle_len = min(crackle_len, n_samples - pos)
        
        if crackle_len > 0:
            crackle = np.random.randn(crackle_len) * 0.3
            vinyl[pos:pos+crackle_len] += crackle
    
    # Büyük pop'lar (seyrek)
    n_pops = int(pop_frequency * duration)
    
    for _ in range(n_pops):
        pos = np.random.randint(0, n_samples)
        pop_len = int(sr * 0.01)
        pop_len = min(pop_len, n_samples - pos)
        
        if pop_len > 0:
            t_pop = np.arange(pop_len) / sr
            envelope = np.exp(-t_pop * 100)
            pop = np.random.randn(pop_len) * envelope * 2.0
            vinyl[pos:pos+pop_len] += pop
    
    # Toz gürültüsü (sürekli düşük seviye)
    if dust_noise > 0:
        dust = generate_pink_noise(duration, sr, dust_noise * 0.2)
        vinyl += dust
    
    # Frekans bandı
    vinyl = apply_bandpass_filter(vinyl, sr, (200, 4000))
    vinyl = normalize_signal(vinyl, amplitude)
    
    # Naturalness uygula
    vinyl = apply_naturalness(vinyl, sr, naturalness, nat_params)
    
    return vinyl


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: BRAINWAVE BLOGS
# ═══════════════════════════════════════════════════════════════════════════

"""
DELTA WAVE BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
center_frequency   | Merkez frekans Hz (0.5-4Hz)        | 0.5-4.0         | 2.0   | Delta dalga frekansı
modulation_depth   | Modülasyon derinliği               | 0.0-1.0         | 0.3   | Frekans dalgalanma
beat_frequency     | Binaural beat frekansı Hz          | 0.0-4.0         | 1.5   | Stereo beat offset
mode               | Mod: tone veya boost               | tone/boost      | tone  | Ton üret veya filtre
"""

def brainwave_blog_delta(duration, sr, amplitude=0.3, mode="tone"):
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    center_frequency = 2.0
    modulation_depth = 0.3
    beat_frequency = 1.5
    
    if mode == "tone":
        # Saf delta ton üretimi
        delta = np.sin(2 * np.pi * center_frequency * t)
        
        # Hafif modülasyon
        if modulation_depth > 0:
            mod = np.sin(2 * np.pi * 0.1 * t) * modulation_depth
            delta *= (1.0 + mod)
        
        delta = normalize_signal(delta, amplitude)
        return delta
    
    elif mode == "boost":
        # Delta bandını boost et (mevcut sinyale uygulanır)
        noise = generate_pink_noise(duration, sr, amplitude)
        delta_filtered = apply_bandpass_filter(noise, sr, (0.5, 4.0))
        return delta_filtered
    
    return np.zeros(n_samples)


"""
THETA WAVE BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
center_frequency   | Merkez frekans Hz (4-8Hz)          | 4.0-8.0         | 6.0   | Theta dalga frekansı
modulation_depth   | Modülasyon derinliği               | 0.0-1.0         | 0.3   | Frekans dalgalanma
beat_frequency     | Binaural beat frekansı Hz          | 0.0-4.0         | 2.0   | Stereo beat offset
mode               | Mod: tone veya boost               | tone/boost      | tone  | Ton üret veya filtre
"""

def brainwave_blog_theta(duration, sr, amplitude=0.3, mode="tone"):
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    center_frequency = 6.0
    modulation_depth = 0.3
    
    if mode == "tone":
        theta = np.sin(2 * np.pi * center_frequency * t)
        
        if modulation_depth > 0:
            mod = np.sin(2 * np.pi * 0.15 * t) * modulation_depth
            theta *= (1.0 + mod)
        
        theta = normalize_signal(theta, amplitude)
        return theta
    
    elif mode == "boost":
        noise = generate_pink_noise(duration, sr, amplitude)
        theta_filtered = apply_bandpass_filter(noise, sr, (4.0, 8.0))
        return theta_filtered
    
    return np.zeros(n_samples)


"""
ALPHA WAVE BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
center_frequency   | Merkez frekans Hz (8-13Hz)         | 8.0-13.0        | 10.0  | Alpha dalga frekansı
modulation_depth   | Modülasyon derinliği               | 0.0-1.0         | 0.3   | Frekans dalgalanma
beat_frequency     | Binaural beat frekansı Hz          | 0.0-4.0         | 2.5   | Stereo beat offset
mode               | Mod: tone veya boost               | tone/boost      | tone  | Ton üret veya filtre
"""

def brainwave_blog_alpha(duration, sr, amplitude=0.4, mode="tone"):
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    center_frequency = 10.0
    modulation_depth = 0.3
    
    if mode == "tone":
        alpha = np.sin(2 * np.pi * center_frequency * t)
        
        if modulation_depth > 0:
            mod = np.sin(2 * np.pi * 0.2 * t) * modulation_depth
            alpha *= (1.0 + mod)
        
        alpha = normalize_signal(alpha, amplitude)
        return alpha
    
    elif mode == "boost":
        noise = generate_pink_noise(duration, sr, amplitude)
        alpha_filtered = apply_bandpass_filter(noise, sr, (8.0, 13.0))
        return alpha_filtered
    
    return np.zeros(n_samples)


"""
BETA WAVE BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
center_frequency   | Merkez frekans Hz (13-30Hz)        | 13.0-30.0       | 20.0  | Beta dalga frekansı
modulation_depth   | Modülasyon derinliği               | 0.0-1.0         | 0.3   | Frekans dalgalanma
beat_frequency     | Binaural beat frekansı Hz          | 0.0-4.0         | 3.0   | Stereo beat offset
mode               | Mod: tone veya boost               | tone/boost      | tone  | Ton üret veya filtre
"""

def brainwave_blog_beta(duration, sr, amplitude=0.3, mode="tone"):
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    center_frequency = 20.0
    modulation_depth = 0.3
    
    if mode == "tone":
        beta = np.sin(2 * np.pi * center_frequency * t)
        
        if modulation_depth > 0:
            mod = np.sin(2 * np.pi * 0.25 * t) * modulation_depth
            beta *= (1.0 + mod)
        
        beta = normalize_signal(beta, amplitude)
        return beta
    
    elif mode == "boost":
        noise = generate_pink_noise(duration, sr, amplitude)
        beta_filtered = apply_bandpass_filter(noise, sr, (13.0, 30.0))
        return beta_filtered
    
    return np.zeros(n_samples)


"""
GAMMA WAVE BLOG KONTROL PANELİ
══════════════════════════════════════════════════════════════════════════════
Parametre          | Açıklama                           | Referans Aralık | Örnek | Etki
──────────────────────────────────────────────────────────────────────────────────────────
center_frequency   | Merkez frekans Hz (30-100Hz)       | 30.0-100.0      | 40.0  | Gamma dalga frekansı
modulation_depth   | Modülasyon derinliği               | 0.0-1.0         | 0.3   | Frekans dalgalanma
beat_frequency     | Binaural beat frekansı Hz          | 0.0-4.0         | 3.5   | Stereo beat offset
mode               | Mod: tone veya boost               | tone/boost      | tone  | Ton üret veya filtre
"""

def brainwave_blog_gamma(duration, sr, amplitude=0.2, mode="tone"):
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    center_frequency = 40.0
    modulation_depth = 0.3
    
    if mode == "tone":
        gamma = np.sin(2 * np.pi * center_frequency * t)
        
        if modulation_depth > 0:
            mod = np.sin(2 * np.pi * 0.3 * t) * modulation_depth
            gamma *= (1.0 + mod)
        
        gamma = normalize_signal(gamma, amplitude)
        return gamma
    
    elif mode == "boost":
        noise = generate_pink_noise(duration, sr, amplitude)
        gamma_filtered = apply_bandpass_filter(noise, sr, (30.0, 100.0))
        return gamma_filtered
    
    return np.zeros(n_samples)


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: FREKANS İŞLEMLERİ
# ═══════════════════════════════════════════════════════════════════════════

def apply_frequency_operations(signal_input, sr, operations):
    """
    Spesifik frekans işlemlerini uygula
    operations: specific_frequencies listesi
    """
    signal_output = signal_input.copy()
    
    for op in operations:
        freq = op["freq"]
        operation = op["operation"]
        
        if operation == "boost":
            q_factor = op.get("q_factor", 2.0)
            gain_db = op.get("gain_db", 6.0)
            
            nyquist = sr / 2
            freq_norm = freq / nyquist
            
            if 0.001 < freq_norm < 0.999:
                bandwidth = freq_norm / q_factor
                low_freq = max(0.001, freq_norm - bandwidth / 2)
                high_freq = min(0.999, freq_norm + bandwidth / 2)
                
                sos = sps.butter(2, [low_freq, high_freq], btype='band', output='sos')
                boosted = sps.sosfilt(sos, signal_output)
                
                gain_linear = 10 ** (gain_db / 20.0)
                signal_output = signal_output + boosted * (gain_linear - 1.0)
        
        elif operation == "notch":
            q_factor = op.get("q_factor", 5.0)
            gain_db = op.get("gain_db", -20.0)
            
            nyquist = sr / 2
            freq_norm = freq / nyquist
            
            if 0.001 < freq_norm < 0.999:
                bandwidth = freq_norm / q_factor
                low_freq = max(0.001, freq_norm - bandwidth / 2)
                high_freq = min(0.999, freq_norm + bandwidth / 2)
                
                sos = sps.butter(2, [low_freq, high_freq], btype='bandstop', output='sos')
                signal_output = sps.sosfilt(sos, signal_output)
        
        elif operation == "bandpass":
            q_factor = op.get("q_factor", 1.5)
            
            nyquist = sr / 2
            freq_norm = freq / nyquist
            
            if 0.001 < freq_norm < 0.999:
                bandwidth = freq_norm / q_factor
                low_freq = max(0.001, freq_norm - bandwidth / 2)
                high_freq = min(0.999, freq_norm + bandwidth / 2)
                
                sos = sps.butter(4, [low_freq, high_freq], btype='band', output='sos')
                signal_output = sps.sosfilt(sos, signal_output)
        
        elif operation == "synth_tone":
            tone_amplitude = op.get("amplitude", 0.1)
            t = np.arange(len(signal_output)) / sr
            tone = np.sin(2 * np.pi * freq * t) * tone_amplitude
            signal_output += tone
        
        elif operation == "additive":
            add_amplitude = op.get("amplitude", 0.05)
            t = np.arange(len(signal_output)) / sr
            additive = np.sin(2 * np.pi * freq * t) * add_amplitude
            signal_output += additive
    
    return normalize_signal(signal_output, np.max(np.abs(signal_input)))


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 8: KARIŞTIRMA SİSTEMİ (MIX BLOG)
# ═══════════════════════════════════════════════════════════════════════════

def mix_blogs(duration, sr, mix_config, noise_mix_config, brainwave_cfg, nat_params):
    """
    Tüm aktif blogları karıştır ve final sinyali oluştur
    """
    mixed_signal = np.zeros(int(duration * sr))
    
    print("=" * 70)
    print("MIX BLOG BAŞLATILIYOR")
    print("=" * 70)
    
    # Natural sounds karıştır
    if ENABLE_NATURAL_SOUNDS:
        for sound_name, config in noise_mix_config.items():
            if config["enabled"]:
                print(f"Üretiliyor: {sound_name} (weight={config['weight']:.2f}, naturalness={config['naturalness']:.2f})")
                
                if sound_name == "rain":
                    blog_signal = sound_blog_rain(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "thunder":
                    blog_signal = sound_blog_thunder(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "wind":
                    blog_signal = sound_blog_wind(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "ocean":
                    blog_signal = sound_blog_ocean(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "fire":
                    blog_signal = sound_blog_fire(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "crickets":
                    blog_signal = sound_blog_crickets(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "car":
                    blog_signal = sound_blog_car(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "train":
                    blog_signal = sound_blog_train(duration, sr, config['weight'], config['naturalness'], nat_params)
                elif sound_name == "vinyl":
                    blog_signal = sound_blog_vinyl(duration, sr, config['weight'], config['naturalness'], nat_params)
                else:
                    continue
                
                # Frekans bandı uygula
                blog_signal = apply_bandpass_filter(blog_signal, sr, config['freq_range'])
                
                mixed_signal += blog_signal * config['weight']
    
    # Technical noise ekle
    if ENABLE_NOISE_GENERATOR:
        for noise_type, enabled in noise_types.items():
            if enabled:
                print(f"Üretiliyor: {noise_type} noise (amplitude=0.3)")
                noise_signal = generate_noise(noise_type, duration, sr, 0.3)
                mixed_signal += noise_signal * 0.2
    
    # Brainwave ekle
    for wave_name, config in brainwave_cfg.items():
        if config["enabled"]:
            print(f"Üretiliyor: {wave_name} brainwave (freq={config['center_freq']}Hz, amp={config['amplitude']:.2f})")
            
            if wave_name == "delta":
                wave_signal = brainwave_blog_delta(duration, sr, config['amplitude'], config['mode'])
            elif wave_name == "theta":
                wave_signal = brainwave_blog_theta(duration, sr, config['amplitude'], config['mode'])
            elif wave_name == "alpha":
                wave_signal = brainwave_blog_alpha(duration, sr, config['amplitude'], config['mode'])
            elif wave_name == "beta":
                wave_signal = brainwave_blog_beta(duration, sr, config['amplitude'], config['mode'])
            elif wave_name == "gamma":
                wave_signal = brainwave_blog_gamma(duration, sr, config['amplitude'], config['mode'])
            else:
                continue
            
            mixed_signal += wave_signal
    
    # Frekans işlemleri uygula
    if ENABLE_FREQUENCY_FILTERS and len(specific_frequencies) > 0:
        print(f"Frekans işlemleri uygulanıyor: {len(specific_frequencies)} işlem")
        mixed_signal = apply_frequency_operations(mixed_signal, sr, specific_frequencies)
    
    # Final normalizasyon
    mixed_signal = normalize_signal(mixed_signal, MASTER_AMPLITUDE)
    
    print("=" * 70)
    print(f"MIX TAMAMLANDI: {duration}s, {sr}Hz")
    print("=" * 70)
    
    return mixed_signal


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 9: GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════

def visualize_signal(signal_data, sr, title="Audio Signal"):
    """Dalga formu ve spektrum görselleştirme"""
    if not ENABLE_VISUALIZER:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    
    # Dalga formu
    time_axis = np.arange(len(signal_data)) / sr
    sample_points = min(len(signal_data), sr * 5)
    
    axes[0].plot(time_axis[:sample_points], signal_data[:sample_points], linewidth=0.5)
    axes[0].set_title(f"{title} - Dalga Formu")
    axes[0].set_xlabel("Zaman (s)")
    axes[0].set_ylabel("Genlik")
    axes[0].grid(True, alpha=0.3)
    
    # Spektrum
    fft_data = np.fft.rfft(signal_data)
    fft_freqs = np.fft.rfftfreq(len(signal_data), 1/sr)
    fft_magnitude = np.abs(fft_data)
    
    axes[1].plot(fft_freqs, 20 * np.log10(fft_magnitude + 1e-10), linewidth=0.5)
    axes[1].set_title(f"{title} - Frekans Spektrumu")
    axes[1].set_xlabel("Frekans (Hz)")
    axes[1].set_ylabel("Güç (dB)")
    axes[1].set_xlim([20, sr/2])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 10: DOSYA ÇIKTISI
# ═══════════════════════════════════════════════════════════════════════════

def export_audio(sig, sr, stereo=True):
    """WAV dosyası olarak output klasörüne kaydet, dosya ismi timestamp'e göre"""
    if not ENABLE_FILE_EXPORT:
        return
    
    # Sadece output klasörü
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dosya ismini timestamp ile oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_dsp_output_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # Stereo çıkış oluştur
    if stereo and sig.ndim == 1:
        stereo_signal = np.stack([sig, sig], axis=1)
    elif not stereo and sig.ndim == 2:
        stereo_signal = np.mean(sig, axis=1)
    else:
        stereo_signal = sig
    
    # Normalize ve float32'ye dönüştür
    stereo_signal = np.clip(stereo_signal, -1.0, 1.0).astype(np.float32)
    
    # Kaydet
    sf.write(filepath, stereo_signal, sr)
    
    print(f"\n{'='*70}")
    print(f"SES DOSYASI KAYDEDILDI: {filepath}")
    print(f"Format: {'Stereo' if stereo else 'Mono'}, {sr}Hz, {len(sig)/sr:.2f}s")
    print(f"{'='*70}\n")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 11: ANA PROGRAM
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Ana üretim fonksiyonu"""
    print("\n" + "=" * 70)
    print("ADVANCED AUDIO DSP & AI-ASSISTED SOUND GENERATOR")
    print("=" * 70)
    print(f"Yapılandırma:")
    print(f"  - Süre: {DURATION}s")
    print(f"  - Örnekleme Hızı: {SAMPLE_RATE}Hz")
    print(f"  - Stereo: {STEREO_MODE}")
    print(f"  - Master Amplitude: {MASTER_AMPLITUDE}")
    print(f"  - Noise Generator: {ENABLE_NOISE_GENERATOR}")
    print(f"  - Natural Sounds: {ENABLE_NATURAL_SOUNDS}")
    print(f"  - Mixing System: {ENABLE_MIXING_SYSTEM}")
    print(f"  - Frequency Filters: {ENABLE_FREQUENCY_FILTERS}")
    print(f"  - Visualizer: {ENABLE_VISUALIZER}")
    print(f"  - File Export: {ENABLE_FILE_EXPORT}")
    print("=" * 70 + "\n")
    
    # Ana karışık sinyal üret
    if ENABLE_MIXING_SYSTEM:
        final_signal = mix_blogs(
            DURATION,
            SAMPLE_RATE,
            mix_blog_config,
            noise_mix,
            brainwave_config,
            naturalness_params
        )
    else:
        # Sadece tek bir test sinyali üret
        print("Mix sistemi devre dışı, test sinyali üretiliyor...")
        final_signal = generate_pink_noise(DURATION, SAMPLE_RATE, MASTER_AMPLITUDE)
    
    # Stereo dönüşümü
    if STEREO_MODE:
        print("\nStereo sinyal oluşturuluyor...")
        
        # Hafif stereo genişlik için sağ kanalı biraz kaydır
        left_channel = final_signal
        right_channel = np.roll(final_signal, int(SAMPLE_RATE * 0.001))  # 1ms shift
        
        # Stereo matris
        stereo_signal = np.stack([left_channel, right_channel], axis=1)
        output_signal = stereo_signal
    else:
        output_signal = final_signal
    
    # Görselleştirme
    if ENABLE_VISUALIZER:
        print("\nGörselleştirme oluşturuluyor...")
        visualize_signal(final_signal, SAMPLE_RATE, "Final Mixed Output")
    
    # Dosya çıktısı
    if ENABLE_FILE_EXPORT:
        export_audio(output_signal, SAMPLE_RATE, STEREO_MODE)
    
    # Özet rapor
    print("\n" + "=" * 70)
    print("ÜRETIM TAMAMLANDI")
    print("=" * 70)
    print("\nAktif Katmanlar:")
    
    layer_count = 0
    
    if ENABLE_NATURAL_SOUNDS:
        for sound_name, config in noise_mix.items():
            if config["enabled"]:
                layer_count += 1
                print(f"  [{layer_count}] {sound_name.upper()}: "
                      f"weight={config['weight']:.2f}, "
                      f"freq={config['freq_range']}, "
                      f"naturalness={config['naturalness']:.2f}")
    
    if ENABLE_NOISE_GENERATOR:
        for noise_type, enabled in noise_types.items():
            if enabled:
                layer_count += 1
                print(f"  [{layer_count}] {noise_type.upper()} NOISE: amplitude=0.3")
    
    for wave_name, config in brainwave_config.items():
        if config["enabled"]:
            layer_count += 1
            print(f"  [{layer_count}] {wave_name.upper()} WAVE: "
                  f"freq={config['center_freq']}Hz, "
                  f"amplitude={config['amplitude']:.2f}")
    
    print(f"\nToplam Aktif Katman: {layer_count}")
    print(f"Toplam Süre: {DURATION}s")
    print(f"Toplam Örnek: {len(final_signal):,}")
    print(f"Örnekleme Hızı: {SAMPLE_RATE}Hz")
    print(f"Bit Derinliği: 32-bit float")
    print(f"Kanal: {'Stereo (2ch)' if STEREO_MODE else 'Mono (1ch)'}")
    
    if ENABLE_FREQUENCY_FILTERS and len(specific_frequencies) > 0:
        print(f"\nFrekans İşlemleri: {len(specific_frequencies)} işlem uygulandı")
        for idx, op in enumerate(specific_frequencies, 1):
            print(f"  [{idx}] {op['freq']}Hz - {op['operation']}")
    
    print("\n" + "=" * 70)
    print("KULLANIM KILAVUZU")
    print("=" * 70)
    print("""
1. KONTROL PANELİNDEN AYAR DEĞİŞTİRME:
   - Kod başındaki ENABLE_* değişkenlerini True/False yapın
   - noise_types dict'inden istediğiniz noise türünü aktif/deaktif edin
   - noise_mix dict'inden her katmanın parametrelerini düzenleyin
   - brainwave_config dict'inden brainwave ayarlarını değiştirin

2. NATURALNESS PARAMETRELERİNİ AYARLAMA:
   - naturalness_params dict'indeki değerleri düzenleyin
   - randomness_amount: Rastgelelik miktarı (0.0-1.0)
   - freq_mod_depth: Frekans modülasyon derinliği (0-100 Hz)
   - freq_mod_rate: Modülasyon hızı (0.01-20 Hz)
   - amp_variation_amount: Genlik varyasyonu (0.0-0.5)
   - grain_size: Granüler tane boyutu (5-200 ms)
   - micro_timing_jitter: Zamanlama sapması (0-50 ms)

3. FREKANS İŞLEMLERİ EKLEME:
   - specific_frequencies listesine yeni dict ekleyin:
     {"freq": 440, "operation": "boost", "q_factor": 2.0, "gain_db": 6.0}
   - İşlem tipleri: boost, notch, bandpass, synth_tone, additive

4. YENİ KATMAN EKLEME:
   - noise_mix dict'ine yeni bir giriş ekleyin:
     "yeni_ses": {"enabled": True, "weight": 0.5, "freq_range": (100, 1000), "naturalness": 0.7}
   - İlgili sound_blog fonksiyonunu oluşturun veya mevcut birini kullanın

5. BRAINWAVE AYARLAMA:
   - brainwave_config dict'inden istediğiniz dalga türünü enabled=True yapın
   - center_freq: Merkez frekansı ayarlayın
   - amplitude: Dalga gücünü ayarlayın
   - mode: "tone" (ton üretir) veya "boost" (bandı güçlendirir)

6. STEREO GENİŞLİK AYARLAMA:
   - main() fonksiyonunda stereo shift miktarını değiştirin:
     right_channel = np.roll(final_signal, int(SAMPLE_RATE * 0.001))
   - 0.001 değerini artırarak daha geniş stereo elde edin

7. ÇIKTI AYARLARI:
   - DURATION: Toplam süreyi saniye olarak ayarlayın
   - SAMPLE_RATE: Örnekleme hızını ayarlayın (44100, 48000, 96000)
   - MASTER_AMPLITUDE: Ana ses seviyesini ayarlayın (0.0-1.0)

8. NATURALNESS SEVİYELERİ REHBERİ:
   - 0.0: Tamamen sentetik, hiç modülasyon yok
   - 0.25: Hafif rastgelelik, minimal varyasyon
   - 0.5: Orta seviye, dengeli modülasyon
   - 0.75: Yüksek seviye, belirgin gerçekçilik
   - 1.0: Maksimum gerçekçilik, tüm efektler maksimum

9. ÖRNEK KOMBINASYONLAR:
   
   a) Rahatlatıcı Yağmur + Alpha:
      - rain: enabled=True, weight=0.7, naturalness=0.8
      - alpha: enabled=True, amplitude=0.4
   
   b) Fırtına Atmosferi:
      - rain: enabled=True, weight=0.6, naturalness=0.7
      - thunder: enabled=True, weight=0.5, naturalness=0.9
      - wind: enabled=True, weight=0.4, naturalness=0.6
   
   c) Okyanus Meditasyonu:
      - ocean: enabled=True, weight=0.8, naturalness=0.9
      - theta: enabled=True, amplitude=0.3
      - pink noise: enabled=True
   
   d) Şömine Ambiyansı:
      - fire: enabled=True, weight=0.7, naturalness=0.8
      - brown noise: enabled=True
   
   e) Gece Doğası:
      - crickets: enabled=True, weight=0.6, naturalness=0.85
      - wind: enabled=True, weight=0.3, naturalness=0.6
      - alpha: enabled=True, amplitude=0.3

10. PERFORMANS İPUÇLARI:
    - Uzun süreler için (>60s) DURATION'ı artırın
    - Daha hızlı işlem için SAMPLE_RATE'i düşürün (22050)
    - ENABLE_VISUALIZER'ı False yaparak render hızını artırın
    - Çok fazla katman karıştırıyorsanız MASTER_AMPLITUDE'u azaltın

11. TEKNİK NOISE FARKLARI:
    - WHITE: Düz spektrum, tüm frekanslar eşit - maske, test
    - PINK: 1/f azalma, doğal dağılım - ambiyans, müzik
    - BROWN: 1/f² azalma, bas dominant - okyanus, bas
    - BLUE: f artış, tiz dominant - yüksek frekans maske
    - VIOLET: f² artış, ultra tiz - çok yüksek frekans enerji
    - GRAY: Psiko-akustik düz - insan kulağına dengeli
    - GREEN: 500Hz merkez - konuşma bandı maske

12. FREKANS İŞLEM TİPLERİ:
    - boost: Belirli frekansı güçlendirir (gain_db: +değer)
    - notch: Belirli frekansı zayıflatır (gain_db: -değer)
    - bandpass: Sadece o bandı geçirir (q_factor: keskinlik)
    - synth_tone: Saf sinüs tonu ekler (amplitude: ses seviyesi)
    - additive: Harmonik ton ekler (amplitude: katman seviyesi)

13. SORUN GİDERME:
    - Clipping/bozulma: MASTER_AMPLITUDE'u azaltın veya weight değerlerini düşürün
    - Çok sessiz çıkış: weight ve amplitude değerlerini artırın
    - İstenmeyen frekanslar: specific_frequencies ile notch filtre ekleyin
    - Çok sentetik ses: naturalness değerlerini artırın (0.7-1.0)
    - Çok karmaşık/gürültülü: Bazı katmanları devre dışı bırakın

NOTLAR:
- Tüm değişiklikler kod içinden yapılır, yeniden çalıştırın
- Değişiklikleri test etmek için kısa DURATION kullanın (5-10s)
- Her parametre aralığı tablolarda belirtilmiştir
- Extreme değerler beklenmeyen sonuçlar üretebilir
""")
    print("=" * 70 + "\n")
    
    return output_signal


# ═══════════════════════════════════════════════════════════════════════════
# PROGRAM BAŞLANGICI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        final_audio = main()
        print("\n✓ Program başarıyla tamamlandı!\n")
    except Exception as e:
        print(f"\n✗ Hata oluştu: {str(e)}\n")
        import traceback
        traceback.print_exc()