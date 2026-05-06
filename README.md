# StockSARIMA × Prophet

Aplikasi forecasting harga saham IDX menggunakan model **SARIMA** dan **Prophet** dengan ensemble prediction.

link apps : https://share.streamlit.io/
---

##  Struktur Proyek

```
stock-forecasting/
├── app.py              # Streamlit dashboard utama
├── data_pipeline.py    # Download & cache data per ticker
├── model_utils.py      # Inference, ensemble, metrics, signals
├── train.py            # Training pipeline SARIMA + Prophet
├── models/             # Model tersimpan per ticker (.pkl)
├── data/               # Data CSV per ticker
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Pakai

### 1. Jalankan aplikasi
```bash
streamlit run app.py
```

### 2. Training model via CLI
```bash
# Train semua ticker
python train.py

# Train ticker tertentu
python train.py "ANTM - PT Aneka Tambang Tbk"
```

---

## 🔑 Fitur Utama

- **Multi-ticker support** — ANTM, BBCA, BBRI, BMRI, TLKM, dll.
- **SARIMA** dilatih pada `log1p(Close)` → tidak ada overflow
- **Prophet** menangani seasonality mingguan & tahunan
- **Ensemble** weighted average (configurable)
- **Sinyal trading** BUY / SELL / HOLD per model
- **Metrics** MAE, RMSE, MAPE yang akurat

---

## ⚠️ Disclaimer

Aplikasi ini hanya untuk tujuan edukasi. Jangan gunakan prediksi ini sebagai satu-satunya dasar keputusan investasi.
