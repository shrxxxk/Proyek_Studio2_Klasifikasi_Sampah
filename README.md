# Studio 2 — Klasifikasi Sampah · Kelompok 18/noobmaster69

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Nelwan Fakhri | 122140173 | Arsitektur model ConvNeXt-Tiny, transfer learning & perancangan custom classifier head, evaluasi metrik Macro-F1. |
| 2  | Ali Akbar | 123140162 | Pipeline dataset & DataLoader, stratified sampling validasi lokal (`my_val/`), penanganan class imbalance (class weights). |
| 3  | Raditya Erza Farandi | 122140209 | Strategi optimasi & scheduler (AdamW + CosineAnnealing), regularisasi (label smoothing & augmentasi citra), verifikasi kontrak submission. |

## Ringkasan Pendekatan

Proyek ini bertujuan mengklasifikasikan citra sampah ke dalam 5 kelas (*botol_plastik*, *gelas_plastik*, *kertas_kardus*, *saset_kemasan*, *styrofoam*) dengan memaksimalkan metrik **Macro-F1** pada validation set tersembunyi.

- **Arsitektur / backbone**: 
  - **ConvNeXt-Tiny** (*pretrained* ImageNet-1K V1: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`) dari `torchvision.models`.
  - Arsitektur modern berbasis konvolusi *depthwise* 7x7, *inverted bottleneck* (rasio 4x), dan *LayerNorm* yang menggabungkan kesederhanaan CNN dengan kapasitas representasi Vision Transformer.
  - **Custom Classifier Head**: Mengganti head bawaan dengan struktur bertingkat yang dilengkapi normalisasi dan regulasi:
    ```python
    nn.Sequential(
        model.classifier[0],  # LayerNorm2d bawaan ConvNeXt (768 channel)
        nn.Flatten(1),
        nn.Dropout(p=0.3),
        nn.Linear(768, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 5),   # 5 kelas klasifikasi
    )
    ```
  - Total parameter: **28.614.757** parameter (~109,16 MB FP32).

- **Strategi pelatihan**: 
  - **Differential Learning Rate**: Backbone dilatih dengan learning rate konservatif (1e-4) untuk mempertahankan bobot fitur umum ImageNet, sedangkan classifier head baru dilatih lebih agresif (5e-4).
  - **Optimizer & Scheduler**: Optimizer **AdamW** (`weight_decay = 1e-4`) dikombinasikan dengan scheduler **CosineAnnealingLR** (`T_max = 25`, `eta_min = 1e-6`).
  - **Loss Function**: `CrossEntropyLoss` yang dimodifikasi dengan:
    1. **Class Weights**: Pembobotan berbasis akar invers frekuensi tiap kelas (ternormalisasi mean 1: `botol_plastik: 1.008`, `gelas_plastik: 1.009`, `kertas_kardus: 1.206`, `saset_kemasan: 0.888`, `styrofoam: 0.889`) untuk mengatasi ketimpangan jumlah sampel kelas minoritas (*kertas_kardus*).
    2. **Label Smoothing**: Nilai smoothing 0.05 untuk memitigasi efek *noisy labels* dari dataset web.
  - **Validasi & Early Stopping**: Validasi dilakukan pada setiap epoch dengan metrik utama **Macro-F1** (`patience = 5`, `min_delta = 0.001`). Model terbaik disimpan pada **Epoch 9** (Macro-F1: 0.9758, val_loss: 0.3122) dan pelatihan berhenti otomatis di Epoch 14.
  - Batch size: 32, Seed: 42 untuk reproduksibilitas.

- **Data tambahan**: 
  - **Tidak menggunakan data tambahan dari luar**. Pelatihan murni memanfaatkan train set resmi (~2.756 gambar) yang dipisahkan secara *stratified* menjadi 80% train (2.205 gambar) dan 20% validasi lokal `my_val/` (551 gambar).

- **Ukuran input & preprocessing**:
  - Resolusi citra: **300 x 300 piksel** (dinaikkan dari resolusi standar 224 untuk memperjelas detail spasial krusial seperti leher/ulir botol vs bibir gelas plastik).
  - Preprocessing Validasi / Evaluasi (`get_transform()`):
    - `Resize((300, 300))`
    - `ToTensor()`
    - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
    - Input dibaca dalam format PIL Image (RGB), identik dengan pipeline resmi `evaluate.py`.
  - Augmentasi Citra Pelatihan:
    - `RandomResizedCrop(300, scale=(0.6, 1.0))`
    - `RandomHorizontalFlip()`
    - `RandomRotation(15)`
    - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
    - `ToTensor()` + `Normalize(mean, std)`

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---|---|
| Preview 1 (14 Okt) | Baseline fine-tuning ResNet-18 | 0.9214 | 0.9180 |
| Preview 2 (21 Okt) | Eksplorasi ConvNeXt-Tiny (224x224, AdamW) | 0.9511 | 0.9485 |
| Final (25 Okt)     | ConvNeXt-Tiny (300x300, Diff LR, Class Weight, Label Smoothing) | **0.9758** (Akurasi: 97.64%) | *Menunggu evaluasi* |

---

### Rincian Evaluasi pada Validasi Lokal (`my_val/` — 551 Sampel)

- **Akurasi Keseluruhan**: **97.64%** (538 dari 551 sampel terprediksi tepat).
- **Macro-F1**: **0.9758** (Jauh melampaui baseline ResNet-18 ~0.92).
- **Loss Validasi**: 0.3122.

#### Classification Report (Per Kelas):

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **botol_plastik** | 0.9813 | 1.0000 | **0.9906** | 105 |
| **gelas_plastik** | 0.9623 | 0.9808 | **0.9714** | 104 |
| **kertas_kardus** | 0.9722 | 0.9589 | **0.9655** | 73 |
| **saset_kemasan** | 0.9706 | 0.9778 | **0.9742** | 135 |
| **styrofoam** | 0.9923 | 0.9627 | **0.9773** | 134 |
| **Macro Average** | **0.9757** | **0.9760** | **0.9758** | 551 |
| **Weighted Average** | **0.9766** | **0.9764** | **0.9764** | 551 |

#### Matriks Konfusi:
- `botol_plastik`: 105 / 105 benar (**100% recall**, 0 kesalahan).
- `gelas_plastik`: 102 / 104 benar (1 tertukar ke *kertas_kardus*, 1 ke *saset_kemasan*).
- `kertas_kardus`: 70 / 73 benar (1 tertukar ke *botol_plastik*, 1 ke *saset_kemasan*, 1 ke *styrofoam*).
- `saset_kemasan`: 132 / 135 benar (1 tertukar ke *botol_plastik*, 2 ke *gelas_plastik*).
- `styrofoam`: 129 / 134 benar (1 tertukar ke *gelas_plastik*, 2 ke *kertas_kardus*, 2 ke *saset_kemasan*).

---

### Catatan Eksperimen Singkat

- **Apa yang dicoba**:
  1. *Baseline Transfer Learning*: Mencoba ResNet-18 pretrained yang menghasilkan Macro-F1 ~0.92.
  2. *Evolusi Arsitektur*: Beralih ke **ConvNeXt-Tiny** karena arsitektur depthwise 7x7 dan inverted bottleneck modern lebih fleksibel menangkap bentuk sampah yang tidak beraturan.
  3. *Resolusi Input*: Membandingkan resolusi 224 x 224 vs 300 x 300.
  4. *Regularisasi & Robustness*: Menambahkan dropout (0.3 dan 0.2), BatchNorm1d pada classifier head, serta *Label Smoothing* (0.05) untuk menahan noise label.
  5. *Class Weighting*: Menambahkan bobot kelas akar invers frekuensi untuk menyeimbangkan kelas *kertas_kardus*.
- **Apa yang berhasil**:
  - Peningkatan resolusi ke 300 x 300 piksel terbukti krusial untuk membedakan fitur visual kecil seperti leher botol PET vs bibir gelas cup plastik, mendongkrak Macro-F1 dari 0.9511 ke **0.9758**.
  - *Differential learning rate* dan scheduler Cosine Annealing berhasil menstabilkan proses fine-tuning tanpa merusak fitur umum ImageNet.
  - Simulasi `evaluate.py` menunjukkan model menyelesaikan evaluasi 551 gambar hanya dalam **13,1 detik** (sangat aman di bawah batas maksimal 15 menit).
- **Apa yang gagal / dihindari**:
  - Classifier head tanpa dropout mengalami *overfitting* cepat (akurasi train mencapai 100% sementara akurasi validasi tertahan di ~94%).
  - Resolusi 224x224 sering membuat model bingung antara sampah kemasan saset yang kusut dengan gelas plastik remuk.

## Cara Reproduksi

### 1. Persiapan Lingkungan

Pastikan Python 3.10+ telah terpasang. Buat dan aktifkan virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Eksperimen

Buka dan jalankan seluruh sel pada notebook:

```bash
jupyter notebook Studi_Proyek_2_CIFAR10_CNN.ipynb
```

*Catatan: Riwayat pelatihan tiap model otomatis dicatat ke dalam `results_cifar10.json`. Jika sesi terputus, eksekusi berikutnya akan otomatis melanjutkan model yang belum selesai dilatih.*

### 3. Memeriksa Hasil Evaluasi

Untuk menampilkan perbandingan metrik akhir secara langsung via terminal:

```bash
python -c "
import json
results = json.load(open('results_cifar10.json'))
for m, d in results.items():
    print(f'{m:15s} | Test Acc: {d[\"history\"][\"test_acc\"][-1]:.2f}% | Params: {d[\"params\"]:,} | FLOPs: {d[\"flops\"]:,}')
"
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```

## Link model.pth

Model terbaik (**ResNet-18**, Macro-F1: 83.49%) dan checkpoint hasil eksperimen dapat diakses melalui:

```
checkpoints/
# Atau link rilis GitHub kelompok:
https://github.com/shrxxxk/Proyek_Studio2_Klasifikasi_Sampah/releases/tag/v1.10.0
```

