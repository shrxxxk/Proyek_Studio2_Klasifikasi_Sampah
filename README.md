# Studio 2 — Klasifikasi Sampah · Kelompok 18/noobmaster69

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |              |     |                  |
| 2  |              |     |                  |
| 3  |              |     |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: 
- **Strategi pelatihan**: 
- **Data tambahan**: 
- **Ukuran input & preprocessing**:

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
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
```
https://github.com/shrxxxk/Proyek_Studio2_Klasifikasi_Sampah/releases/tag/v1.7.0
```
