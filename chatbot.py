"""Chatbot informasi jadwal dan lokasi mengajar dosen."""

import time
import csv
import os
import re
import difflib
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Konfigurasi API dan model

load_dotenv()

# API key disimpan di file .env
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY tidak ditemukan. "
        "Pastikan file .env tersedia dan berisi GEMINI_API_KEY=API_KEY_ANDA."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# Konfigurasi keluaran model
GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0,
    max_output_tokens=2000,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    ),
)


# Nama file yang digunakan

DATASET_FILE = "dataset_chatbot_final.csv"
HISTORY_FILE = "history.csv"


# Pengolahan dataset

def load_dataset(file_path: str) -> pd.DataFrame:
    """Membaca dataset jadwal dari file CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File dataset '{file_path}' tidak ditemukan. "
            f"Pastikan file berada satu folder dengan file Python."
        )

    df = pd.read_csv(file_path)
    return df


def dataframe_to_context(df: pd.DataFrame) -> str:
    """Mengubah seluruh DataFrame menjadi context berformat CSV."""
    return df.to_csv(index=False)


def build_aggregation_summary(df: pd.DataFrame) -> str:
    """Membuat ringkasan jumlah sesi dari beberapa kolom dataset."""
    lines = [
        "RINGKASAN HASIL PERHITUNGAN (sudah dihitung otomatis dan akurat, "
        "GUNAKAN INI sebagai acuan angka, jangan menghitung ulang secara manual):"
    ]

    if "DOSEN" in df.columns:
        lines.append("\nJumlah sesi kelas per dosen:")
        for dosen, jumlah in df["DOSEN"].value_counts().items():
            lines.append(f"- {dosen}: {jumlah} sesi")

    if "HARI" in df.columns:
        lines.append("\nJumlah sesi kelas per hari:")
        for hari, jumlah in df["HARI"].value_counts().items():
            lines.append(f"- {hari}: {jumlah} sesi")

    if "RUANG" in df.columns:
        lines.append("\nJumlah sesi kelas per ruang:")
        for ruang, jumlah in df["RUANG"].value_counts().items():
            lines.append(f"- {ruang}: {jumlah} sesi")

    if "MATA KULIAH" in df.columns:
        lines.append("\nJumlah sesi per mata kuliah:")
        for mk, jumlah in df["MATA KULIAH"].value_counts().items():
            lines.append(f"- {mk}: {jumlah} sesi")

    if "KELAS" in df.columns:
        lines.append("\nJumlah sesi per kelas:")
        for kelas, jumlah in df["KELAS"].value_counts().items():
            lines.append(f"- {kelas}: {jumlah} sesi")

    return "\n".join(lines)


def find_matching_dosen(df: pd.DataFrame, user_query: str) -> list:
    """Mencari nama dosen yang sesuai dengan pertanyaan pengguna."""
    if "DOSEN" not in df.columns:
        return []

    dosen_list = df["DOSEN"].dropna().unique().tolist()
    query_lower = user_query.lower()
    matches = set()

    # Cocokkan potongan nama yang muncul pada pertanyaan
    for dosen in dosen_list:
        name_parts = re.sub(r"[.,]", " ", dosen).split()
        for part in name_parts:
            # Gelar akademik tidak digunakan untuk pencocokan
            if part.lower() in {
                "s.kom", "skom", "m.kom", "mkom", "s.t", "st", "m.t", "mt",
                "ph.d", "phd", "s.si", "ssi", "m.si", "msi", "ir", "prof",
                "assoc", "s.pd", "spd", "s.hum", "shum", "m.hum", "mhum",
            }:
                continue
            if len(part) >= 4 and part.lower() in query_lower:
                matches.add(dosen)
                break

    # Gunakan fuzzy matching jika nama belum ditemukan
    if not matches:
        close = difflib.get_close_matches(
            query_lower, [d.lower() for d in dosen_list], n=3, cutoff=0.5
        )
        for c in close:
            for dosen in dosen_list:
                if dosen.lower() == c:
                    matches.add(dosen)

    return list(matches)


def build_dosen_detail_section(df: pd.DataFrame, matched_dosen: list) -> str:
    """Mengambil seluruh jadwal dosen yang ditemukan pada dataset."""
    if not matched_dosen:
        return ""

    lines = [
        "DAFTAR LENGKAP & PASTI SESI MENGAJAR UNTUK DOSEN YANG DIMAKSUD "
        "(hasil query langsung dari data, WAJIB gunakan SELURUH baris ini "
        "apa adanya, JANGAN menambah, mengurangi, atau menduplikasi baris "
        "manapun saat menjawab):"
    ]

    for dosen in matched_dosen:
        subset = df[df["DOSEN"] == dosen]
        lines.append(f"\nDosen: {dosen} (total {len(subset)} sesi)")
        for i, row_dict in enumerate(subset.to_dict("records"), start=1):
            lines.append(
                f"{i}. Mata Kuliah: {row_dict.get('MATA KULIAH', '-')} | "
                f"Kelas: {row_dict.get('Kelas', '-')} | "
                f"Hari: {row_dict.get('HARI', '-')} | "
                f"Waktu: {row_dict.get('WAKTU', '-')} | "
                f"Ruang: {row_dict.get('RUANG', '-')}"
            )

    return "\n".join(lines)


def strip_stray_markdown_list(answer: str) -> str:
    """Mengubah jawaban berbentuk daftar menjadi satu paragraf."""
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^[\*\-\u2022]\s*", "", ln)
        cleaned.append(ln)
    return " ".join(cleaned)


# Penyusunan prompt

def build_prompt(context_data: str, user_query: str) -> str:
    """Menyusun role, context, rules, dan pertanyaan pengguna."""

    role = (
        "Anda adalah asisten virtual kampus yang bertugas memberikan informasi "
        "seputar jadwal dan lokasi mengajar dosen kepada mahasiswa maupun staf "
        "akademik berdasarkan data yang tersedia."
    )

    context = (
        "Berikut adalah data jadwal mengajar dosen dalam format CSV. "
        "Setiap baris merepresentasikan satu sesi kelas, dengan struktur kolom "
        "sebagai berikut:\n"
        "- Semester           : keterangan semester dan kurikulum yang berlaku\n"
        "- Kelas              : nama kelas mahasiswa (contoh: IF-1/SI/II)\n"
        "- KDMK               : kode mata kuliah\n"
        "- MATA KULIAH        : nama mata kuliah yang diajarkan\n"
        "- SKS                : jumlah satuan kredit semester mata kuliah tersebut\n"
        "- KELAS MATAKULIAH   : kode kelas untuk mata kuliah tersebut (contoh: PBO-1)\n"
        "- HARI               : hari perkuliahan berlangsung (Senin s.d. Sabtu)\n"
        "- WAKTU              : jam mulai dan jam selesai perkuliahan\n"
        "- RUANG              : ruangan atau lokasi tempat perkuliahan berlangsung\n"
        "- DOSEN              : nama dosen pengampu mata kuliah tersebut (termasuk "
        "gelar akademik)\n\n"
        f"Data:\n{context_data}"
    )

    rules = (
        "Aturan yang harus Anda patuhi saat menjawab:\n"
        "1. Jawablah HANYA berdasarkan data pada dataset di atas.\n"
        "2. Jangan mengarang atau menambahkan informasi yang tidak ada di dataset.\n"
        "3. Jika informasi yang ditanyakan tidak ditemukan pada dataset, jawab "
        "dengan sopan bahwa data tersebut tidak tersedia pada dataset.\n"
        "4. Gunakan Bahasa Indonesia yang sopan, ramah, dan mudah dipahami.\n"
        "5. Jawaban harus berupa kalimat naratif (free text) dalam beberapa "
        "kalimat yang mengalir, BUKAN dalam bentuk JSON, tabel, ataupun daftar "
        "bernomor/simbol. JANGAN PERNAH memulai baris dengan karakter '*', '-', "
        "atau '•', bahkan jika jumlah datanya banyak. Gunakan kata sambung "
        "seperti 'selain itu', 'kemudian', 'lalu', dan 'terakhir' untuk "
        "merangkai beberapa informasi dalam satu paragraf. Contoh jawaban yang "
        "BENAR: 'Bapak X mengajar tiga mata kuliah. Pertama, beliau mengajar "
        "Basis Data untuk kelas IF-1 pada hari Senin di ruang R101. Selain itu, "
        "beliau juga mengajar Algoritma untuk kelas IF-2 pada hari Selasa di "
        "ruang R102. Terakhir, beliau mengajar Statistika untuk kelas IF-3 pada "
        "hari Rabu di ruang R103.' Contoh yang SALAH (dilarang): jawaban yang "
        "berisi baris-baris berawalan tanda bintang atau strip.\n"
        "6. Jangan menampilkan proses berpikir, analisis internal, atau reasoning "
        "Anda. Tampilkan HANYA jawaban akhirnya saja.\n"
        "7. Untuk pertanyaan yang bersifat pencarian, agregasi, perbandingan, "
        "maupun superlatif (misalnya: dosen yang mengajar paling banyak, hari "
        "dengan jadwal terbanyak, ruang yang paling sering dipakai, jumlah "
        "dosen/mata kuliah/kelas, dan sejenisnya), GUNAKAN ringkasan hasil "
        "perhitungan yang sudah disediakan pada bagian 'RINGKASAN HASIL "
        "PERHITUNGAN' di atas sebagai acuan angka. JANGAN menghitung ulang "
        "secara manual dari data mentah karena rawan keliru.\n"
        "8. Nama dosen pada dataset ditulis lengkap beserta gelar akademik. "
        "Jika pengguna menyebut nama dosen secara singkat, tidak lengkap, atau "
        "dengan sapaan (misalnya 'Pak Hanhan' atau 'Bu Edna'), cocokkan dengan "
        "nama dosen pada dataset yang paling sesuai berdasarkan kemiripan nama.\n"
        "9. Jika pertanyaan pengguna berada di luar topik jadwal dan lokasi "
        "mengajar dosen (misalnya membahas topik umum di luar dataset), jawab "
        "dengan sopan bahwa chatbot ini hanya melayani informasi berdasarkan "
        "dataset jadwal dosen yang tersedia.\n"
        "10. Jika pada CONTEXT terdapat bagian 'DAFTAR LENGKAP & PASTI SESI "
        "MENGAJAR UNTUK DOSEN YANG DIMAKSUD', maka WAJIB gunakan HANYA "
        "baris-baris pada daftar tersebut sebagai acuan saat menjawab "
        "pertanyaan tentang jadwal dosen itu. JANGAN mencari atau "
        "membaca ulang dari data CSV mentah untuk kasus ini, karena daftar "
        "tersebut sudah pasti benar dan lengkap. JANGAN menambah baris yang "
        "tidak ada di daftar, dan JANGAN melewatkan satu pun baris yang ada "
        "di daftar tersebut."
    )

    prompt = (
        f"ROLE:\n{role}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"RULES:\n{rules}\n\n"
        f"USER QUERY:\n{user_query}"
    )

    return prompt


# Permintaan ke Gemini

def ask_gemini(prompt: str) -> str:
    """Mengirim prompt ke Gemini dan mengembalikan jawabannya."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=GENERATION_CONFIG,
        )

        answer = response.text.strip() if response.text else (
            "Maaf, terjadi kesalahan saat memproses jawaban dari model."
        )
        return answer

    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi Gemini API: {e}"


# Riwayat percakapan

def save_history(
    file_path: str,
    timestamp: str,
    question: str,
    answer: str,
    response_time: float
) -> None:
    """Menambahkan hasil percakapan ke history.csv."""
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                ["Timestamp", "Pertanyaan", "Jawaban", "Response Time (detik)"]
            )

        writer.writerow(
            [timestamp, question, answer, f"{response_time:.3f}"]
        )


def print_result(question: str, answer: str, response_time: float) -> None:
    """Menampilkan hasil chatbot pada terminal."""
    print("=" * 58)
    print("Pertanyaan  :")
    print(question)
    print("\nJawaban     :")
    print(answer)
    print(f"\nResponse Time :\n{response_time:.3f} detik")
    print("=" * 58)


def main():
    print("=" * 58)
    print("CHATBOT INFORMASI JADWAL AKADEMIK")
    print("Ditenagai oleh Google Gemini 2.5 Flash")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri program.")
    print("=" * 58)

    # Dataset dibaca sekali saat program dimulai
    try:
        df_jadwal = load_dataset(DATASET_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Seluruh dataset dan hasil agregasi digunakan sebagai context
    context_data = (
        dataframe_to_context(df_jadwal)
        + "\n\n"
        + build_aggregation_summary(df_jadwal)
    )

    while True:
        user_query = input("\nMasukkan pertanyaan Anda: ").strip()

        if user_query.lower() in ["exit", "keluar"]:
            print("\nTerima kasih telah menggunakan chatbot ini. Sampai jumpa!")
            break

        if not user_query:
            print("Pertanyaan tidak boleh kosong. Silakan coba lagi.")
            continue

        # Ambil jadwal khusus jika pertanyaan menyebut nama dosen
        matched_dosen = find_matching_dosen(df_jadwal, user_query)
        dosen_detail = build_dosen_detail_section(df_jadwal, matched_dosen)

        query_context = context_data
        if dosen_detail:
            query_context = context_data + "\n\n" + dosen_detail

        prompt = build_prompt(query_context, user_query)

        start_time = time.perf_counter()

        answer = ask_gemini(prompt)

        # Rapikan keluaran jika model menggunakan bullet
        if re.search(r"^\s*[\*\-\u2022]\s+", answer, flags=re.MULTILINE):
            answer = strip_stray_markdown_list(answer)

        end_time = time.perf_counter()
        response_time = end_time - start_time

        print_result(user_query, answer, response_time)

        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_history(
            HISTORY_FILE,
            timestamp_now,
            user_query,
            answer,
            response_time
        )


if __name__ == "__main__":
    main()
