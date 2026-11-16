# Forklift Vision AI (prototipe)

Aplikasi Python untuk deteksi person dengan YOLO26, pelacakan BoT-SORT, dan estimasi kedalaman kamera menggunakan Depth Anything V2 **Metric**. Semua hasil jarak, motion, warning, dan TTC bersifat eksperimental. Depth bukan pengukuran fisik tervalidasi atau jarak terhadap bodi forklift. Jangan gunakan sebagai sistem keselamatan, pengereman, atau interlock.

## Struktur dan model

```text
forklift_ai.py
config.json
forklift_vision/
tests/test_logic.py
models/                         # file bobot lokal, tidak disertakan
third_party/Depth-Anything-V2/  # clone resmi, tidak diduplikasi
outputs/
```

Ambil source resmi dan checkpoint metric yang cocok dari [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) dan [repo metric_depth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth). Clone repo resmi ke `third_party/Depth-Anything-V2`; kode menambahkan `third_party/Depth-Anything-V2/metric_depth` ke `sys.path` dan mengimpor `depth_anything_v2.dpt`. Unduh checkpoint **Hypersim Small vits** ke `models/depth_anything_v2_metric_hypersim_vits.pth`. Untuk outdoor, pilih checkpoint **VKITTI Small vits** dan `--depth-model outdoor`; jangan mencampur checkpoint dengan config dataset lain. Nilai max depth config outdoor diimplementasikan sebagai 80 m, indoor 20 m; batas ini bukan klaim akurasi.

Unduh bobot detector `yolo26n.pt` dari mekanisme resmi Ultralytics (file diletakkan di `models/`). Tidak ada bobot atau source pihak ketiga besar yang disertakan. Nama file lokal dapat diberikan dengan `--yolo-model`.

## Instalasi laptop Windows

PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instal PyTorch sesuai OS, versi Python, dan CUDA driver dari [halaman resmi PyTorch](https://pytorch.org/get-started/locally/). Untuk CPU, pilih build CPU di selector resmi; untuk GPU NVIDIA, pilih kombinasi CUDA yang didukung driver dan PyTorch. Lalu:

```powershell
python -m pip install -r requirements.txt
```

Pasang source/checkpoint model seperti bagian di atas, lalu jalankan:

```powershell
python forklift_ai.py --source 0 --device auto
python forklift_ai.py --source videos/test.mp4 --headless
```

## Instalasi laptop Linux

Gunakan Python virtual environment, instal PyTorch build CPU atau CUDA yang sesuai dari [petunjuk resmi](https://pytorch.org/get-started/locally/), lalu `python -m pip install -r requirements.txt`. Pasang OpenCV camera backend dan izin perangkat sesuai distro.

```bash
python forklift_ai.py --source /dev/video0
python forklift_ai.py --source 'rtsp://192.168.1.100:554/stream1' --headless
python forklift_ai.py --source videos/test.mp4 --show-depth
```

Jika kamera gagal dibuka, cek `ls -l /dev/video*`, izin grup `video`, aplikasi lain yang sedang memakai kamera, dukungan backend OpenCV, dan akses jaringan/URL RTSP. RTSP melakukan reconnect terbatas dengan exponential backoff. Opsi `--gstreamer` meminta backend GStreamer OpenCV; pipeline GStreamer kamera vendor dapat memerlukan konfigurasi backend tersendiri.

## NVIDIA Jetson

File `config.jetson.json` berisi konfigurasi runtime Jetson dengan preset ringan (YOLO 480, depth 392 setiap 5 frame, device auto, ambang bahaya `<3 m` dan waspada `3–4 m`). Saat program dijalankan tanpa `--config` di Linux Jetson yang memiliki `/etc/nv_tegra_release`, file ini dipilih otomatis. Di Windows/Linux non-Jetson, program tetap memakai `config.json`.

Sebelum menjalankan, salin model dan source ke lokasi yang ditentukan di konfigurasi. Ubah `source` sekali di `config.jetson.json` ke indeks kamera Jetson (default `0`) atau URL RTSP kamera. Letakkan `floor_calibration.json` di root proyek jika overlay zona lantai sudah dikalibrasi untuk kamera tersebut. Setelah setup, jalankan GUI dengan:

```bash
python3 forklift_ai.py
```

Untuk instalasi tanpa monitor, set `headless` menjadi `true` dalam `config.jetson.json`.

Periksa perangkat dan software sebelum memilih wheel:

```bash
cat /etc/nv_tegra_release
uname -m
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Gunakan panduan resmi [PyTorch NVIDIA Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/) dan matriks JetPack untuk **model Jetson/JetPack yang benar-benar terpasang**. Wheel PyTorch Jetson harus cocok dengan JetPack, Python, dan arsitektur ARM64. Jangan menimpa PyTorch bawaan/kompatibel Jetson dengan wheel x86 atau perintah pip generik. Dependency Ultralytics/OpenCV dan modelnya juga perlu diverifikasi pada ARM64. Karena versi JetPack belum ditentukan, README ini tidak mengklaim satu perintah instalasi cocok untuk semua Jetson.

Setelah memastikan `torch.cuda.is_available()` sesuai kondisi perangkat:

```bash
python3 forklift_ai.py --source 0 --preset jetson
```

Preset hanya mengubah ukuran input dan frekuensi depth; tidak mengubah algoritma. Gunakan CPU secara eksplisit dengan `--device cpu` jika CUDA tidak tersedia. Backend TensorRT bukan syarat.

## Penggunaan CLI

Parameter utama: `--source`, `--device auto|cpu|cuda`, `--yolo-model`, `--depth-model indoor|outdoor`, `--depth-checkpoint`, `--depth-size`, `--yolo-size`, `--conf`, `--depth-every`, `--show-depth`, `--headless`, `--save-video [PATH]`, `--log-file`, `--config`, `--calibrate-floor`, dan `--preset laptop-cpu|laptop-gpu|jetson`.

## Web UI lokal

Pasang Flask melalui requirements proyek (PyTorch tetap dipasang terpisah sesuai platform):

```powershell
python -m pip install -r requirements.txt
python web_forklift.py --host 127.0.0.1 --port 8080
```

Buka `http://127.0.0.1:8080`. Halaman Monitoring menampilkan frame terkini dan status pipeline. Selector **Pilih view** menyediakan kamera asli, deteksi AI, dan peta kedalaman AI. Pada peta depth, warna hanya visualisasi rentang depth; tanda silang menunjukkan titik kaki yang dipakai. Jarak person di panel dihitung dari median depth pada patch kecil di tengah-bawah bounding box, dengan titik kaki referensi di tengah sisi bawah box. Nilainya tetap estimasi camera-axis, bukan jarak terverifikasi ke forklift. Sidebar deteksi dan JPEG memakai `frame_id` yang sama; server menahan sejumlah frame terbaru dalam buffer terbatas. Jika frame yang diminta sudah keluar dari buffer, UI menolak menampilkan gambar yang berbeda dan meminta sinkronisasi ulang.

Di halaman **Settings**, bagian resolusi inferensi AI menyediakan preset berdasarkan sisi panjang kamera dan opsi resolusi asli. Ukuran lebar/tinggi dihitung otomatis dengan mempertahankan rasio kamera. Pilihan diterapkan ke ukuran input YOLO dan Depth Anything, sedangkan frame tampilan dan koordinat zona tetap pada resolusi asli. Pilihan disimpan lokal di `calibration/web_settings.json`.

Di bagian Kalibrasi Zona, tekan **Ambil frame untuk kalibrasi**; frame, waktu, resolusi, serta hasil deteksi dari frame yang sama akan dibekukan selama pengeditan. Bounding box dapat ditampilkan/disembunyikan dan penyertaan box di gambar referensi mengikuti pilihan overlay saat menyimpan. Titik disimpan dalam koordinat piksel frame asli walaupun ukuran gambar di browser berubah. **Batalkan snapshot** atau ambil frame baru untuk mengganti snapshot; penyimpanan yang berhasil juga menutup sesi snapshot tersebut.

Server memulai satu pipeline tanpa debug/reloader dan menjalankan Flask sampai dihentikan; saat keluar pipeline ditutup di blok `finally`. Snapshot kalibrasi dibatasi default 8 sesi dan TTL absolut 900 detik sejak dibuat. Membaca snapshot tidak memperpanjang TTL; snapshot expired dihapus dan tidak dapat dimuat atau disimpan. Jika semua slot masih digunakan, permintaan snapshot baru ditolak dengan pesan agar sesi disimpan atau dibatalkan. Nilai dapat diubah melalui key `calibration_snapshot_max_count` dan `calibration_snapshot_ttl_s` di file konfigurasi.

Editor menyediakan tombol jenis zona, selesai poligon, undo titik, batal poligon, hapus zona, reset, muat, dan simpan. Status membedakan kalibrasi belum tersedia, JSON invalid, resolusi tidak cocok, dan kalibrasi yang berhasil dimuat; status juga mencantumkan jenis zona yang memang ada. Zona yang belum dibuat harus digambar operator. Format tetap `forklift-vision-pixel-zones` versi 1, sama dengan `calibrate_zones.py`; file default `calibration/zone_calibration.json`, gambar referensi di sampingnya. Gambar referensi dibuat dari piksel kamera dengan overlay zona dan opsional bounding box, tanpa elemen browser.

JSON dan JPG referensi divalidasi di file sementara sebelum pemasangan. Pergantian setiap file memakai rename atomik pada filesystem yang sama dan kegagalan pergantian normal menjalankan rollback ke file sebelumnya. Dua file tidak bisa dijamin sebagai satu transaksi atomik oleh filesystem umum: mati listrik/proses pada sela dua rename masih dapat meninggalkan pasangan yang berbeda. Kamera ditutup saat shutdown dan worker ditunggu beberapa detik; bila pembacaan kamera atau inferensi tetap tertahan, worker tidak dimatikan paksa dan status/log menyatakan bahwa ia belum berhenti. Web UI ini tidak membuat `VideoWriter`.

Resolusi harus sama persis; zona nonaktif jika resolusi berubah. Titik kaki tiap person dipetakan ke poligon untuk menampilkan `BAHAYA`, `WASPADA`, `AMAN`, atau `DI LUAR ZONA`. Poligon hanya mengklasifikasikan lokasi pada bidang gambar, bukan jarak fisik. Jarak camera-depth tetap berasal dari Depth Anything pada ROI kaki; zona `AMAN` tidak menghasilkan nilai meter. Jika ROI depth invalid/stale atau model depth tidak tersedia, panel menampilkan alasan dan jarak tetap `UNKNOWN`; status zona tidak mengubahnya menjadi jarak atau jaminan aman.

Kalibrasi jarak berada di halaman **Kalibrasi zona**. Tekan **Ambil frame depth**, klik titik pada orang/kaki atau target yang diketahui jaraknya, masukkan jarak aktual camera-axis dalam meter, lalu tekan **Tambah titik ukur**. Satu titik menghasilkan faktor skala; beberapa titik memakai median rasio aktual terhadap depth mentah. Tekan **Simpan koreksi jarak** untuk menyimpan `calibration/distance_calibration.json` dan menerapkan faktor ke estimasi person berikutnya. Koreksi hanya berlaku untuk kombinasi model/checkpoint dan resolusi yang sama. Ambil sampel pada beberapa rentang jarak dan gunakan pengukuran independen untuk menilai hasil; MAE sampel kalibrasi bukan validasi akurasi.

Untuk dibuka dari LAN, cari alamat IPv4 server lalu bind ke semua interface:

```bash
python3 web_forklift.py --host 0.0.0.0 --port 8080
```

Perangkat lain membuka `http://<IP-komputer-server>:8080`. Mode LAN ini tidak memakai login: setiap perangkat yang dapat menjangkau port bisa melihat video dan mengubah kalibrasi; peringatan ini tampil di halaman. Aplikasi tidak membuka port router atau mengirim data ke layanan luar secara otomatis. Gunakan `--source` untuk override kamera/video, `--model` untuk menunjuk bobot detector lokal, dan `--calibration` untuk memilih file zona.

```bash
python forklift_ai.py --source 0
python forklift_ai.py --source videos/test.mp4 --device cpu --headless
python forklift_ai.py --source 'rtsp://192.168.1.100:554/stream1' --depth-every 5
python forklift_ai.py --source 0 --calibrate-floor
```

`config.json` menyediakan default; argumen CLI menimpa nilai JSON. Key keyboard GUI: `Q` keluar, `D` toggle depth map, `B` toggle box, `S` screenshot, `R` reset history. Screenshot dan video berada di `outputs/recordings/`, log JSONL di `outputs/logs/events.jsonl`. Frame live diproses langsung tanpa antrean aplikasi tak terbatas. File video menggunakan timestamp media atau frame/FPS valid; live menggunakan monotonic timestamp.

## Cara kerja dan interpretasi

YOLO membatasi kelas ke person dan `track(..., tracker="botsort.yaml", persist=True)` mempertahankan tracker antar-frame. ID yang tidak tersedia ditampilkan unknown. Metric model memproses frame sesuai `--depth-every`; ROI torso dipangkas dari tepi, hanya depth finite dalam batas konfigurasi dipakai, lalu median/MAD dan filter temporal menolak variasi tinggi. `raw_depth_m` dan `estimated_camera_depth_m` dicatat. Nilai kamera-axis bukan jarak Euclidean, bukan jarak forklift, dan tidak menjadi pengukuran baru saat depth cache stale.

Closing speed dihitung robust dari beberapa observasi ber-ID dan timestamp valid. `CLOSING` hanya bermakna jarak kamera-orang relatif berkurang; forklift bergerak juga dapat menyebabkannya. `experimental_ttc_s` mengasumsikan kecepatan relatif konstan dan tidak memodelkan lintasan lateral; bukan prediksi tabrakan.

Kalibrasi lantai meminta 4 klik berurutan mengelilingi area lantai yang terlihat lalu koordinat (X,Y) meter per titik. Jadikan titik acuan forklift sebagai `(0, 0)`, sumbu X melintang forklift, dan sumbu Y ke arah depan. Setelah kalibrasi, aplikasi menampilkan zona merah `< critical_distance_m`, kuning dari batas kritis sampai `warning_distance_m` (inklusif), dan hijau di luarnya; label posisi kaki juga menunjukkan zona serta jarak bidang lantai dari titik acuan. Default proyek: merah `<3 m`, kuning `3–4 m`, hijau `>4 m`. Tanpa kalibrasi, overlay menampilkan `FLOOR ZONES UNCALIBRATED`.

Homography memetakan titik kaki ke bidang kalibrasi saja, tidak mendeteksi lantai secara otomatis, tidak mengukur objek di atas lantai, dan tidak menciptakan peta dunia jika kamera bergerak. Overlay zona adalah zona statis di sekitar titik acuan yang diberikan saat kalibrasi. Jika forklift bergerak di dalam pandangan kamera CCTV, zona ini tidak mengikuti forklift; deteksi dan tracking forklift belum diimplementasikan. Ulangi kalibrasi setelah kamera berubah. Zona ini prototipe visual, bukan zona keselamatan tervalidasi.

## Validasi

Jalankan tes sintetis yang tidak memerlukan model, kamera, GPU, atau GUI:

```bash
python -m unittest discover -s tests -v
python -m py_compile forklift_ai.py forklift_vision/*.py tests/test_logic.py
python forklift_ai.py --help
```

Checklist manual: (1) manusia diam + forklift diam: tracking/range relatif stabil; (2) manusia mendekat: range relatif turun; (3) manusia menjauh: range naik; (4) forklift mendekati orang diam juga menurunkan range relatif; (5) orang menyilang tidak otomatis berarti tabrakan; (6) dua orang memperoleh ID jika tracker berhasil; (7) occlusion ditangani tanpa jarak palsu; (8) kamera putus menghasilkan degraded/unknown; (9) depth invalid/stale tidak ditampilkan sebagai valid; (10) CPU tetap berjalan walau lambat.

Validasi metrik wajib dilakukan sebelum penggunaan operasional: ambil sampel pada jarak terukur 1, 2, 3, 5, dan 8 m sesuai target domain/kamera; catat referensi, estimasi, valid/invalid, kondisi cahaya dan sampel. Laporkan tingkat validitas, MAE `mean(abs(pred-reference))`, dan relative error `mean(abs(pred-reference)/reference)`. Jangan mengklaim akurasi atau FPS sebelum mengukur pada perangkat dan data sebenarnya.

## Keterbatasan dan langkah berikutnya

Belum diverifikasi inferensi model, kualitas tracking, MAE, FPS, CUDA atau Jetson pada mesin target tanpa bobot/perangkat yang sesuai. Depth satu kamera sensitif pada domain, refleksi, occlusion dan kondisi gudang; tracker tidak menggantikan estimasi gerak world-coordinate. Berikutnya: validasi versi dependensi/checkpoint, uji rekaman representatif, ukur depth ground truth, uji disconnect/reconnect, tentukan protokol kalibrasi, dan tinjau aturan warning dengan ahli keselamatan. Sistem ini bukan perangkat safety-certified dan tidak mengendalikan forklift.
