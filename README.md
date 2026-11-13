# Chatbot Informasi Jadwal Akademik

Chatbot berbasis Python untuk menjawab pertanyaan mengenai jadwal perkuliahan dan lokasi mengajar dosen melalui terminal. Aplikasi menggunakan Gemini 2.5 Flash melalui Google Gen AI SDK dengan konteks data jadwal berformat CSV.

## Fitur

- Pencarian jadwal berdasarkan nama dosen.
- Pencocokan nama dosen menggunakan pencocokan teks dan fuzzy matching.
- Perhitungan jumlah sesi berdasarkan dosen, hari, ruang, mata kuliah, dan kelas.
- Penyusunan prompt berisi peran, konteks data, aturan, dan pertanyaan.
- Pencatatan riwayat percakapan dan waktu respons.

## Teknologi

- Python
- pandas
- Gemini API melalui Google Gen AI SDK
- python-dotenv
- CSV

## Dataset

Repository ini menyertakan 10 baris contoh dari jadwal akademik yang sudah tidak berlaku. Nama dosen telah diganti dengan nama fiktif beserta gelar untuk keperluan demonstrasi. Kolom lainnya mengikuti data sumber.

Jawaban dan perhitungan chatbot hanya mencakup data contoh yang disertakan.

## Cara Menjalankan

1. Unduh repository dan ekstrak ke komputer.
2. Buka terminal pada folder proyek.
3. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Buka file `.env`, kemudian ganti nilai contoh berikut dengan API key Gemini Anda:

   ```dotenv
   GEMINI_API_KEY=isi_api_key_anda
   ```

   Nilai tersebut hanya placeholder, bukan API key aktif. Jika file `.env` tidak tersedia setelah mengunduh repository, buat file tersebut pada folder yang sama dengan `chatbot.py`.

5. Pastikan dataset bernama `dataset_chatbot_final.csv`.
6. Jalankan chatbot:

   ```bash
   python chatbot.py
   ```

7. Ketik `exit` atau `keluar` untuk mengakhiri program.

Aplikasi memerlukan koneksi internet dan akses API ke model yang dikonfigurasi pada `MODEL_NAME` di `chatbot.py`. Ketersediaan model dan kuota mengikuti akun API yang digunakan.

## Contoh Pertanyaan

- Siapa saja dosen yang mengajar pada hari Senin?
- Dosen siapa yang memiliki sesi mengajar paling banyak?
- Ruangan apa saja yang digunakan pada hari Selasa?

## Riwayat Percakapan

Riwayat pertanyaan, jawaban, dan waktu respons disimpan secara lokal ke `history.csv`.

## API Key

File `.env` dalam repository hanya berisi placeholder. Setelah mengisinya dengan API key asli di komputer, jangan unggah atau commit perubahan file tersebut ke GitHub.

Jika `.env` sudah tercatat dalam repository, menambahkannya ke `.gitignore` tidak otomatis menghentikan pelacakan file tersebut.
