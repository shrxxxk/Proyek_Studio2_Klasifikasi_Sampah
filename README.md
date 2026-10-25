# Studio 2 — Eksplorasi dan Perbandingan Varian Arsitektur CNN pada CIFAR-10

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Nelwan Fakhri | 122140173 | Eksplorasi arsitektur modern (ConvNeXt-mini, MobileNetV2), analisis efisiensi komputasi (FLOPs & Params), serta penyusunan pipeline eksperimen. |
| 2  | Raditya Erza Farandi | 122140209 | Implementasi arsitektur CNN klasik & deep (LeNet-5, AlexNet), evaluasi per-kelas (confusion matrix), dan tuning adaptasi resolusi CIFAR-10. |
| 3  | Ali Akbar | 123140162 | Eksplorasi arsitektur residual & homogen (ResNet-18, VGG-16), visualisasi kurva pelatihan, dan analisis trade-off Pareto. |

## Ringkasan Pendekatan

Eksperimen ini membandingkan **6 varian arsitektur Convolutional Neural Network (CNN)** lintas generasi pada dataset **CIFAR-10** untuk menganalisis trade-off antara **efektivitas** (akurasi & Macro-F1) dan **efisiensi** (jumlah parameter, FLOPs/GMACs, dan waktu inferensi/pelatihan).

- **Arsitektur / backbone**:
  1. **LeNet-5**: Baseline klasik (2 layer konvolusi 5×5 + Tanh + AvgPool, 3 layer Linear).
  2. **AlexNet (CIFAR-10 variant)**: 5 layer konvolusi 3×3 dengan MaxPool, penambahan BatchNorm untuk stabilitas optimasi Adam, serta classifier Dense (4096-4096-10).
  3. **VGG-16 (BN)**: Arsitektur homogen 13 layer konvolusi 3×3 dengan BatchNorm, max pooling 2×2, dan classifier 512→512→10 (tanpa dropout).
  4. **ResNet-18**: Arsitektur residual dengan *skip connection*. Stem disesuaikan khusus resolusi 32×32 (konvolusi 3×3 stride 1, padding 1, tanpa max-pooling) agar tidak kehilangan resolusi spasial di tahap awal.
  5. **MobileNetV2**: Arsitektur *lightweight* berbasis *inverted residual* dan *depthwise separable convolution*. Stride konvolusi pertama diatur ke 1×1 dan dropout diatur ke 0.
  6. **ConvNeXt-mini**: Arsitektur CNN modern bergaya Vision Transformer (*stem patchify* 2×2, *depthwise convolution* 7×7, *inverted bottleneck* rasio 4×, LayerNorm, dan aktivasi GELU).

- **Strategi pelatihan**:
  - Pelatihan *from scratch* (tanpa pre-trained weights) selama **10 epoch** penuh tanpa early stopping.
  - Optimizer: **Adam** dengan learning rate konstan **0,001** (tanpa scheduler).
  - Batch size: **64**, fungsi kerugian: **Cross-Entropy Loss**.
  - **Prinsip *fair comparison***: Seluruh model dilatih tanpa *dropout*, tanpa *scheduler*, dan di-*seed* ulang menggunakan **Seed 42** sebelum inisialisasi dan pelatihan tiap arsitektur. Seluruh metrik akhir dievaluasi pada epoch ke-10 (bukan epoch terbaik) untuk menghindari bias pemilihan model pada data uji.

- **Data tambahan**: 
  - Tidak menggunakan data eksternal / tambahan.
  - Menggunakan dataset resmi **CIFAR-10** via `torchvision.datasets.CIFAR10` dengan pembagian default bawaan: **50.000 citra latih** (`train=True`) dan **10.000 citra uji** (`train=False`).

- **Ukuran input & preprocessing**:
  - Citra berwarna RGB berukuran **32×32** piksel (10 kelas).
  - Pipeline transformasi identik: `transforms.ToTensor()` dan normalisasi channel standar CIFAR-10 (`mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`).
  - Tanpa augmentasi data (tanpa *random crop* maupun *horizontal flip*) agar hasil murni merefleksikan kapasitas dan *inductive bias* dari tiap arsitektur.

## Hasil

### Ringkasan Putaran Proyek

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---|---|
| Preview 1 (14 Okt) | Baseline Exploration (LeNet-5 & AlexNet) | 56.92% (LeNet-5) / 82.57% (AlexNet) | - |
| Preview 2 (21 Okt) | Deep CNN & Efficient Conv (VGG-16 & MobileNetV2) | 82.63% (VGG-16) / 77.61% (MobileNetV2) | - |
| Final (25 Okt)     | Best Model: ResNet-18 (Residual Learning) | **83.49%** (Akurasi: 83.50%) | *Menunggu evaluasi* |

---

### Perbandingan Efektivitas dan Efisiensi Seluruh Model

Evaluasi pada test set CIFAR-10 (10.000 citra) pada akhir epoch ke-10:

| Peringkat | Model | Test Acc (%) | Macro-F1 (%) | Train Acc (%) | Gap (%) | Test Loss | Params | Params (M) | Ukuran FP32 (MB) | FLOPs (M) | GMACs | Waktu/Epoch (s) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | **ResNet-18** | **83.50%** | **83.49%** | 97.11% | 13.61% | 0.6916 | 11,173,962 | 11.174 | 42.63 | 1,110.8 | 0.5554 | 88.3 |
| 2 | **VGG-16 (BN)** | **82.81%** | **82.63%** | 91.92% | 9.11% | 0.5623 | 14,986,698 | 14.987 | 57.17 | 626.9 | 0.3135 | 60.5 |
| 3 | **AlexNet** | **82.62%** | **82.57%** | 92.93% | 10.31% | 0.6174 | 35,857,482 | 35.857 | 136.79 | 401.0 | 0.2005 | 63.1 |
| 4 | **MobileNetV2** | **77.60%** | **77.61%** | 86.10% | 8.50% | 0.6886 | 2,236,682 | 2.237 | 8.53 | 48.9 | 0.0245 | 53.1 |
| 5 | **ConvNeXt-mini** | **69.53%** | **69.13%** | 96.18% | 26.65% | 1.4713 | 908,218 | 0.908 | 3.46 | 65.9 | 0.0329 | 37.4 |
| 6 | **LeNet-5** | **57.51%** | **56.92%** | 65.57% | 8.06% | 1.2218 | 62,006 | 0.062 | 0.24 | 1.3 | 0.0007 | 15.1 |

*Catatan:*
- FLOPs dihitung via `torch.utils.flop_counter.FlopCounterMode` pada input single image $1 \times 3 \times 32 \times 32$ (1 multiply-add = 2 FLOPs).
- GMACs (*giga multiply-accumulate*) = FLOPs / 2 / $10^9$.

---

### Akurasi per Kelas pada Data Uji (%)

| Model | Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LeNet-5** | 56.1 | 72.6 | 47.2 | 29.8 | 45.4 | 55.9 | 75.5 | 56.4 | 75.3 | 60.9 |
| **AlexNet** | 91.7 | 93.6 | 80.7 | 64.6 | 76.2 | 73.3 | 88.3 | 81.5 | 88.8 | 87.5 |
| **VGG-16** | 89.5 | 96.3 | 82.9 | 55.7 | 79.7 | 79.6 | 91.5 | 85.7 | 78.4 | 88.8 |
| **ResNet-18** | 86.5 | 93.2 | 79.2 | 63.3 | 87.3 | 83.9 | 84.4 | 86.9 | 86.6 | 83.7 |
| **MobileNetV2** | 83.2 | 91.3 | 65.6 | 53.5 | 80.6 | 80.6 | 85.6 | 68.7 | 87.1 | 79.8 |
| **ConvNeXt-mini** | 82.0 | 82.5 | 50.1 | 54.6 | 59.4 | 49.4 | 79.0 | 79.5 | 78.4 | 80.4 |

---

### Catatan Eksperimen & Analisis Temuan

1. **Efektivitas & Arsitektur Terbaik**:
   - **ResNet-18** membukukan performa tertinggi dengan akurasi uji **83.50%** dan Macro-F1 **83.49%**. Mekanisme *identity shortcut* (residual learning) terbukti paling efektif mengatasi degradasi gradien pada jaringan dalam.
   - Modifikasi stem (konvolusi 3×3 stride 1 tanpa max-pooling) sangat krusial untuk mempertahankan dimensi spasial awal pada citra beresolusi rendah (32×32).

2. **Analisis Efisiensi & Trade-off Pareto**:
   - **Pareto-optimal (Akurasi vs FLOPs)**: `ResNet-18`, `VGG-16`, `AlexNet`, `MobileNetV2`, `LeNet-5`.
   - **Pareto-optimal (Akurasi vs Params)**: `ResNet-18`, `MobileNetV2`, `ConvNeXt-mini`, `LeNet-5`.
   - **MobileNetV2** menjadi model paling efisien secara komputasi: hanya memerlukan **2.237 M parameter** dan **48.9 MFLOPs** (hanya 4.4% FLOPs dari ResNet-18), namun berhasil meraih akurasi **77.60%** (Macro-F1 77.61%).
   - Jumlah parameter **tidak selalu sejalan dengan FLOPs**: AlexNet memiliki parameter terbanyak (35.86 M) akibat ukuran dense classifier ($4096 \times 4096$), namun FLOPs-nya (401.0 M) jauh lebih rendah daripada ResNet-18 (1,110.8 M) yang didominasi operasi konvolusi spasial beresolusi tinggi.

3. **Generalisasi vs Overfitting**:
   - **ConvNeXt-mini** mengalami overfitting paling kentara dengan gap train-test mencapai **26.65%** (train acc 96.18% vs test acc 69.53%, test loss naik ke 1.4713). Arsitektur modern berkapasitas representasi tinggi membutuhkan regularisasi kuat (data augmentasi, dropout, atau stochastic depth) saat dilatih pada 50.000 sampel.
   - **LeNet-5** mengalami *underfitting* (train acc 65.57%, test acc 57.51%) karena kapasitas 2 lapis konvolusi 5×5 terlalu kecil untuk menangkap variasi citra natural 10 kelas.

4. **Karakteristik Klasifikasi per Kelas**:
   - Kelas dengan akurasi tertinggi di seluruh model adalah **automobile** (rata-rata 88.2%) dan **frog** (rata-rata 84.1%).
   - Kelas paling menantang adalah **cat** (rata-rata 53.6%) dan **bird** (rata-rata 67.6%), di mana citra kucing sering tertukar dengan anjing (*dog*) pada matriks konfusi karena kemiripan fitur tekstur bulu dan siluet kepala.

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
https://github.com/nelwanfakhri/Studio2_CIFAR10_CNN/releases
```

