# import library
import pandas as pd
import numpy as np
import joblib
import logging
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path
from typing import Dict, Tuple, Optional
from rapidfuzz import process, fuzz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

#tahap g
# -------------------- KONFIGURASI --------------------
REFERENCE_DIR = Path(__file__).parent / "reference"  # sesuaikan
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------- FUNGSI PEMBERSIH TEKS --------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# -------------------- LOAD DATA REFERENSI --------------------
def load_ptn_reference() -> pd.DataFrame:
  
    ptn_path = REFERENCE_DIR / "ptn.csv"
    if not ptn_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {ptn_path}")

    df = pd.read_csv(ptn_path)
    if df.empty:
        raise ValueError("reference/ptn.csv kosong.")

    required = {"id_ptn", "nama_ptn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib PTN hilang: {missing}")

    logger.info(f"Memuat PTN: {len(df)} baris.")
    return df

def load_prodi_reference() -> pd.DataFrame:
    prodi_path = REFERENCE_DIR / "prodi.csv"
    if not prodi_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {prodi_path}")

    df = pd.read_csv(prodi_path)
    if df.empty:
        raise ValueError("reference/prodi.csv kosong.")

    required = {"id_ptn", "id_prodi", "nama_prodi", "jenjang", "daya_tampung_snbp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib Prodi hilang: {missing}")

    df_s1 = df[df["jenjang"] == "S1"].copy()
    logger.info(f"Memuat Prodi S1: {len(df_s1)} baris (dari {len(df)} total prodi).")
    return df_s1

# -------------------- KONFIGURASI SINGKATAN PTN & PRODI --------------------
PTN_SINGKATAN = {
    "uin jkt": "uin jakarta",
    "uin jakarta": "uin jakarta",
    "uinj": "uin jakarta",
    "unj": "universitas negeri jakarta",
    "ui": "universitas indonesia",
    "ugm": "universitas gadjah mada",
    "itb": "institut teknologi bandung",
    "unbraw": "universitas brawijaya",
    "ub": "universitas brawijaya",
    "undip": "universitas diponegoro",
    "unpad": "universitas padjadjaran",
    "uns": "universitas sebelas maret",
    "unsoed": "universitas jenderal soedirman",
    "its": "institut teknologi sepuluh nopember",
    "usu": "universitas sumatera utara",
    "ipb": "institut pertanian bogor",
    "uny": "universitas negeri yogyakarta",
    "unej": "universitas jember",
    "unair": "universitas airlangga",
    "unnes": "universitas negeri semarang",
    "unsil": "universitas siliwangi",
    "untan": "universitas tanjungpura",
    "uin jogja": "uin sunan kalijaga",
    "upn jkt": "upn veteran jakarta",
    "upnvj": "upn veteran jakarta",
    "upn": "upn veteran jakarta",
    "upnvjt": "upn veteran jakarta",
    "uppnvj": "upn veteran jakarta",
    "upnvh": "upn veteran jakarta",
    "upnyk": "upn veteran yogyakarta",
    "pnj": "politeknik negeri jakarta",
    "polimedia kreatif": "politeknik negeri media kreatif",
    "upi": "Universitas Pendidikan Indonesia"
}

PRODI_SINGKATAN = {
    # Bahasa & Sastra
    "bhs & sastra inggris": "bahasa dan sastra inggris",
    "pend bhs arab": "pendidikan bahasa arab",
    "pend bhs inggris": "pendidikan bahasa inggris",
    "pend bhs jepang": "pendidikan bahasa jepang",
    # Hubungan Internasional
    "ilmu hi": "ilmu hubungan internasional",
    "hi": "ilmu hubungan internasional",
    # Pendidikan Guru / Keguruan
    "pgsd": "pendidikan guru sekolah dasar",
    "pgsd kampus purwakarta": "pendidikan guru sekolah dasar",
    # Pendidikan Agama
    "pai": "pendidikan agama islam",
    # Kedokteran & Kesehatan
    "pend dokter": "pendidikan dokter",
    "kesmas": "kesehatan masyarakat",
    "ilkom": "ilmu komputer",                
    # Teknik
    "fttm-c": "teknik pertambangan",         
    # Ekonomi & Manajemen
    "mpp": "manajemen bisnis pariwisata",   
    "pwk": "perencanaan wilayah dan kota",
    "skpm": "statistika",                    
    # Sains & Pendidikan
    "ipa": "pendidikan ipa",
    "pend kimia": "pendidikan kimia",
    # Sosial & Bimbingan
    "bk": "bimbingan dan konseling",
    "bimbingan & konseling": "bimbingan dan konseling",
    # Program D3 / D4 (vokasi) yang sering disingkat
    "d3 budi daya ikan": "budidaya perairan",
    "d4 teknologi rekayasa multimedia": "teknologi rekayasa multimedia",
    "d4 pengelolaan perhotelan": "pengelolaan perhotelan",
    "d4 film dan televisi": "produksi film dan televisi",
    "d4 hubungan masyarakat dan komunikasi digital": "hubungan masyarakat dan komunikasi digital",
    "d4 manajemen keuangan": "manajemen keuangan",
    "d4 manajemen rekod dan arsip": "manajemen rekod dan arsip",
    "d4 terapi okupasi": "terapi okupasi",
    "d4 desain grafis": "desain grafis",
    "d4 seni kuliner dan pengolahan jasa makanan": "seni kuliner dan pengolahan jasa makanan",
    "d4 administrasi perkantoran digital": "administrasi perkantoran digital",
    "d4 pemasaran digital": "pemasaran digital",
    "d4 teknologi dan manajemen ternak": "teknologi dan manajemen ternak",
    "d3 keperawatan": "keperawatan",
    "d4 desain mode": "desain mode",
    # Lainnya
    "sastra daerah untuk sastra jawa": "sastra daerah untuk sastra jawa",
    "teknologi hasil perairan": "teknologi hasil perairan",
    "konservasi sumberdaya hutan dan ekowisata": "konservasi sumberdaya hutan dan ekowisata",
    "d4 bahasa inggris untuk komunikasi bisnis dan profesional": "bahasa inggris untuk komunikasi bisnis dan profesional",
    "ilmu ekonomi dan studi pembangunan": "ilmu ekonomi dan studi pembangunan",
    "peternakan psdku pangandaran": "peternakan",
    "kedokteran gigi": "pendidikan dokter gigi",
    "sekolah farmasi": "farmasi",
}
# -------------------- FUNGSI FUZZY MATCHING --------------------
def normalize_univ(
    univ_name: Optional[str],
    ptn_map: Dict[str, int],
    threshold: int = 70
) -> int:
    if pd.isna(univ_name) or not str(univ_name).strip():
        return -1

    raw = str(univ_name).strip().lower()

    # Ekspansi singkatan jika ada
    if raw in PTN_SINGKATAN:
        raw = PTN_SINGKATAN[raw]

    cleaned = clean_text(raw)

    if cleaned in ptn_map:
        return ptn_map[cleaned]

    result = process.extractOne(cleaned, ptn_map.keys(), scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return ptn_map[result[0]]
    return -1

def normalize_prodi(
    prodi_name: Optional[str],
    id_ptn: int,
    prodi_map_by_ptn: Dict[int, Dict[str, int]],  # key clean
    threshold: int = 70
) -> int:
    if pd.isna(prodi_name) or not str(prodi_name).strip() or id_ptn == -1:
        return -1
    raw = str(prodi_name).strip().lower()
    if raw in PRODI_SINGKATAN:
        raw = PRODI_SINGKATAN[raw]

    cleaned = clean_text(raw)
    prodi_dict = prodi_map_by_ptn.get(id_ptn, {})
    if not prodi_dict:
        return -1
    if cleaned in prodi_dict:
        return prodi_dict[cleaned]
    result = process.extractOne(cleaned, prodi_dict.keys(), scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return prodi_dict[result[0]]
    return -1

# -------------------- FUNGSI INTEGRASI UTAMA --------------------
def integrate_student_data(
    student_df: pd.DataFrame,
    ptn_df: pd.DataFrame,
    prodi_df: pd.DataFrame,
    threshold: int = 70   
) -> pd.DataFrame:
    """
    Menambahkan fitur referensi SNPMB ke dalam dataset siswa.
    """
    required_cols = {"univ_p1", "univ_p2", "jurusan_p1", "jurusan_p2"}
    missing_cols = required_cols - set(student_df.columns)
    if missing_cols:
        raise ValueError(f"Dataset siswa tidak memiliki kolom wajib: {missing_cols}")

    df = student_df.copy()
    logger.info(f"Memulai integrasi untuk {len(df)} siswa.")

    # Siapkan mapping (key sudah clean)
    ptn_map = {clean_text(row["nama_ptn"]): row["id_ptn"] for _, row in ptn_df.iterrows()}

    prodi_map_by_ptn: Dict[int, Dict[str, int]] = {}
    for _, row in prodi_df.iterrows():
        id_ptn = row["id_ptn"]
        nama_clean = clean_text(row["nama_prodi"])
        id_prodi = row["id_prodi"]
        prodi_map_by_ptn.setdefault(id_ptn, {})[nama_clean] = id_prodi

    # Dict daya tampung
    daya_map = {
        (row["id_ptn"], row["id_prodi"]): row["daya_tampung_snbp"]
        for _, row in prodi_df.iterrows()
    }

    # Normalisasi pilihan 1
    logger.info("Memproses pilihan 1 (univ_p1, jurusan_p1)...")
    df["id_ptn_p1"] = df["univ_p1"].apply(lambda x: normalize_univ(x, ptn_map, threshold))
    df["id_prodi_p1"] = df.apply(
        lambda row: normalize_prodi(row["jurusan_p1"], row["id_ptn_p1"], prodi_map_by_ptn, threshold),
        axis=1
    )
    df["daya_tampung_p1"] = df.apply(
        lambda row: daya_map.get((row["id_ptn_p1"], row["id_prodi_p1"]), np.nan),
        axis=1
    )

    # Normalisasi pilihan 2
    logger.info("Memproses pilihan 2 (univ_p2, jurusan_p2)...")
    df["id_ptn_p2"] = df["univ_p2"].apply(lambda x: normalize_univ(x, ptn_map, threshold))
    df["id_prodi_p2"] = df.apply(
        lambda row: normalize_prodi(row["jurusan_p2"], row["id_ptn_p2"], prodi_map_by_ptn, threshold),
        axis=1
    )
    df["daya_tampung_p2"] = df.apply(
        lambda row: daya_map.get((row["id_ptn_p2"], row["id_prodi_p2"]), np.nan),
        axis=1
    )

    # Statistik keberhasilan
    total = len(df)
    matched_ptn_p1 = (df["id_ptn_p1"] != -1).sum()
    matched_ptn_p2 = (df["id_ptn_p2"] != -1).sum()
    matched_prodi_p1 = (df["id_prodi_p1"] != -1).sum()
    matched_prodi_p2 = (df["id_prodi_p2"] != -1).sum()

    logger.info(f"Pilihan 1 - PTN cocok: {matched_ptn_p1}/{total} ({matched_ptn_p1/total*100:.1f}%)")
    logger.info(f"Pilihan 1 - Prodi cocok: {matched_prodi_p1}/{total} ({matched_prodi_p1/total*100:.1f}%)")
    logger.info(f"Pilihan 2 - PTN cocok: {matched_ptn_p2}/{total} ({matched_ptn_p2/total*100:.1f}%)")
    logger.info(f"Pilihan 2 - Prodi cocok: {matched_prodi_p2}/{total} ({matched_prodi_p2/total*100:.1f}%)")

    return df   
# -------------------- FUNGSI ENTRY POINT --------------------
def run_stage_g(student_df: pd.DataFrame, threshold: int = 70) -> pd.DataFrame:
    """
    Entry point utama Tahap G.
    """
    logger.info("=" * 50)
    logger.info("MEMULAI TAHAP G - INTEGRASI DATA REFERENSI")
    logger.info("=" * 50)

    ptn_df = load_ptn_reference()
    prodi_df = load_prodi_reference()

    integrated_df = integrate_student_data(student_df, ptn_df, prodi_df, threshold)

    logger.info("=" * 50)
    logger.info("TAHAP G SELESAI")
    logger.info("=" * 50)

    return integrated_df

def normalize_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menormalkan missing value pada fitur referensi SNPMB.
    NaN pada daya tampung diubah menjadi sentinel -1.
    """
    reference_columns = [
        "daya_tampung_p1",
        "daya_tampung_p2",
    ]

    for col in reference_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(-1)
                .astype("int32")
            )

    return df

def aggregate_subject_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengagregasi nilai setiap mata pelajaran dari semester 1-5
    menjadi satu fitur rata-rata per mata pelajaran, kemudian
    menghitung rata-rata seluruh mata pelajaran.
    """

    subjects = [
        "matematika",
        "bahasa indonesia",
        "bahasa inggris",
        "al-qur`an hadits",
        "akidah akhlak",
        "fikih",
        "fisika",
        "kimia",
        "biologi",
        "matematika ipa",
        "ekonomi",
        "sosiologi",
        "prakarya dan kewirausahaan",
        "pend. jasmani,olahraga dan kesehatan",
        "bahasa arab",
        "geografi",
        "sejarah peminatan",
        "teknologi dan ilmu komunikasi",
        "seni budaya",
        "sejarah indonesia",
        "sejarah kebudayaan islam",
        "pend. pancasila dan kewarganegaraan"
    ]
    aggregated_cols = []

    for subject in subjects:

        semester_cols = [
            f"{subject}_sem{i}"
            for i in range(1, 6)
            if f"{subject}_sem{i}" in df.columns
        ]

        if semester_cols:
            df[subject] = df[semester_cols].mean(axis=1)
            aggregated_cols.append(subject)

    # Rata-rata seluruh mata pelajaran
    if aggregated_cols:
        df["rata_rata"] = df[aggregated_cols].mean(axis=1)

    return df

structural_rules = [
    {
        'cols': ['matematika ipa', 'fisika', 'kimia', 'biologi'],
        'condition_mask': lambda df: df['sma'] == 'ips',
    },
    {
        'cols': ['geografi', 'sosiologi', 'ekonomi'],
        'condition_mask': lambda df: df['sma'] == 'ipa',
    },
    {
        'cols': ['teknologi dan ilmu komunikasi'],
        'condition_mask': lambda df: df['angkatan'] < 2024,
    },
    {
        'cols': ['sejarah peminatan'],
        'condition_mask': lambda df: ~((df['sma'] == 'ips') & (df['angkatan'] == 2023)),
    },
    {
        'cols': ['sosiologi'],
        'condition_mask': lambda df: (df['sma'] == 'ipa') | (df['angkatan'] == 2025),
    },
]

def apply_structural_rule(df, cols, condition_mask):
    if callable(condition_mask):
        condition_mask = condition_mask(df)

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Kolom tidak ditemukan: {missing_cols}")

    df.loc[condition_mask, cols] = np.nan
    return df

def handle_structural_missing(df):
    for rule in structural_rules:
        df = apply_structural_rule(df, **rule)
    return df

def fill_sentinel_for_model(df, sentinel=-1):
    subject_cols = sorted({c for rule in structural_rules for c in rule['cols']})
    existing_cols = [c for c in subject_cols if c in df.columns]
    df[existing_cols] = df[existing_cols].fillna(sentinel)
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Seluruh kolom semester
    semester_cols = [
        col for col in df.columns
        if "_sem" in col
    ]
    # Kolom lain yang sudah tidak diperlukan
    reference_cols = [
        "nama siswa",
        "univ_p1",
        "univ_p2",
        "jurusan_p1",
        "jurusan_p2",
    ]

    cols_to_drop = [
        col
        for col in semester_cols + reference_cols
        if col in df.columns
    ]

    return df.drop(columns=cols_to_drop)

def preprocess(df):
    df = run_stage_g(df, threshold=60)
    df = (
        df
        .pipe(normalize_reference_features)
        .pipe(aggregate_subject_scores)
        .pipe(handle_structural_missing)
        .pipe(fill_sentinel_for_model)
        .pipe(drop_unused_columns)
    )
    return df