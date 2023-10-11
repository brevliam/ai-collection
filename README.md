# AI Collection Tools

## Project Overview
Welcome to AI Collection, a groundbreaking project meticulously developed by the AI Programmer & Data Science Division of M-Knows Consulting. This initiative offers a comprehensive suite of cutting-edge AI tools tailored to enhance the efficiency and effectiveness of Collector Departments and Agencies.

Our mission is to revolutionize the way Collector Departments operate by providing a powerful arsenal of AI-driven solutions. These tools have been meticulously designed to optimize processes, streamline workflows, and empower collectors with the intelligence and automation necessary to excel in their roles.

Explore AI Collection and discover how it can transform your department's operations, enabling you to achieve unparalleled levels of productivity and success.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Endpoints](#endpoints)
- [Contributors](#contributors)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yourproject.git

# Navigate to the repo directory
cd ai-collection

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install the project dependencies
pip install -r requirements.txt

# Navigate to the project directory
cd ai_collection

# Start the development server
python manage.py runserver
```

## Project Structure
- `ai_collection/` : Django project directory
- `fitur-[x]/` : Dedicated Django app directories for each feature or tool (x represents the feature's serial number).
- `fitur_[x]/dataset/` : A repository for storing datasets used in the development of the respective feature.
- `fitur_[x]/model/` : A directory dedicated to housing the machine learning models utilized within the feature.
- `fitur_[x]/function/` : A directory designated for the feature.py module, containing methods for inference, data transformation, and preprocessing specific to the feature.
- `fitur_[x]/libraries/` : A directory designed for the utils.py module, encompassing methods for appending new data to the dataset.

## Features
### Fitur 1: [AI Collection Difficulty Scoring](#11-predict-collection-difficulty-score)
> **PIC**: Roni Merdiansah  
**Definition:**  
Pembuatan Model Memprediksi Tingkat Kesulitan Penagihan untuk setiap profil customer, serta merekomendasikan cara penagihan yang efektif  (Clustering, Classification) untuk setiap Profil Pelanggan sehingga mengoptimalkan kinerja  penagihan.

### Fitur 2: [AI Collector Profiling](#21-predict-collector-profile)
> **PIC**: Roviani Amelia  
**Definition:**  
Pembuatan Model Identifikasi dari setiap kolektor, Pola  Penagihan yang efektif vs kurang efektif dari yang bersangkutan dibandingkan dengan profil pelanggan yang ada.

### Fitur 3: [AI Collection Strategy per Account](#31-predict-collection-strategy)
> **PIC**: Muhammad Reyhan Fitriyan  
**Definition:**  
Pembuatan Model Mengefisienkan Strategi  Penagihan, Melalui rekomendasi cara menagih untuk setiap akun pelanggan dan profil cluster  pelanggan, dengan menggunakan data perilaku, Transaksi, nilai scoring dan data profiling yang  ada.

### Fitur 4: [AI Inventory Assignment Optimization and Campaign Management](#41-predict-assignment)
> **PIC**: Muhammad Othman Lutfi  
**Definition:**  
Pembuatan Model  Mengefisienkan penagihan melalui pengoptimalan penugasan akun ke kolektor tertentu, dan  mengikutsertakannya ke program penagihan khusus yang paling cocok (collection campaign).

### Fitur 5: [AI Activity, Location, Distance and Time Optimization](#51-predict-best-time-to-bill)
> **PIC**: Abdul Roid  
**Definition:**  
Pengoptimalan Penggunaan  Waktu dan aktivitas Sumber Daya Dalam Operasi Penagihan, seperti best time to do what,  tingkat efektifitas untuk campaign seperti apa, siapa lebih baik di assign untuk bekerja dimana,  melakukan apa, jam berapa, kapan dari segi hari dan minggu.

### Fitur 7: [AI Workload Analysis, Prediction, and Optimization](#71-predict-workload-score)
> **PIC**: Brev William Fiden Saragih  
**Definition:**  
Pembuatan Model  Memprediksi dan mengoptimalisasi Beban Kerja di Departemen Penagihan sesuai  lokasi, cabang, dan aging, dengan memprediksi intensitas, frekuensi, campaign,  serta trend dari kinerja team, serta rekomendasi optimalisasinya.

### Fitur 8: [AI Collection Cost Effectiveness](#81-predict-cost-effectiveness)
> **PIC**: Anugrah Aidin Yotolembah  
**Definition:**  
Pembuatan model mengefisienkan dan  efektivitas biaya, dengan membandingkan hasil Analisa workload dengan target  kinerja dan loss forecast.

### Fitur 9: [AI Omni ChannelBot](#91-predict-recommended-solution)
> **PIC**: Ahmad Alfaruq  
**Definition:**  
Pembuatan model mengotomatisasi chatbot,  interaksi reminder penagihan dan menjawab pertanyaan pelanggan secara  cepat dan akurat di channel digital, melakukan follow up sesuai janji waktu  pembayaran (NLP), memberikan dukungan teknis, memberikan rekomendasi  produk dan layanan, melakukan eskalasi dan forward ke PIC selanjutnya.

### Fitur 11: [AI First Payment Default Prevention](#111-predict-default-solution-kredit-pinjaman)
> **PIC**: Yoga Fatwanto  
**Definition:**  
Mencegah first payment default atau  debitur realisasi baru menunggak, dengan diagnosis cepat atas data perilaku,  identifikasi profil, pergerakan trend peningkatan portofolio tertentu dalam  portofolio collection, vintage results, rekomendasi penanganan, siapa, dimana, dan  solusi non regular (Diskon, bagi tanggung jawab dengan sales, early write off).

### Fitur 12: [AI Restructure Analysis Automation](#121-predict-recommendation-tenor)
> **PIC**: Putra Al Farizi  
**Definition:**  
Pembuatan model  mengotomatisasi dan Merekemondasi proses persetujuan restrukturisasi dalam  penagihan, seperti layaknya pemberian kredit baru (Analisa kredit ala AI)

### Fitur 13: AI Reschedule Automation
> **PIC**: Nur Azkia Rahmah  
**Definition:**  
Pembuatan Model mengotomatisasi dan  mengoptimalkan proses reschedule dalam penagihan, meliputi mekanisme  reschedule, mekanisme follow up, auto reminder, modifikasi tenor dan waktu  pembayaran, dengan Kecerdasan Artifisial (AI)

### Fitur 16: AI Warehouse Inventory Management, Storing, Selling, Cost  Projection, and Fraud Detection
> **PIC**: Sintya Tri Wahyu Adityawati  
**Definition:**  
Pembuatan model memprediksi dan  mengoptimalkan pengelolaan persediaan dan gudang, serta mengurangi  potensi fraud dalam proses penyimpanan unit sitaan/jaminan.

### Fitur 17: [AI Collection Fraud Prevention](#171-predict-fraud)
> **PIC**: Dwi Duta Mahardewantoro  
**Definition:**  
Menggunakan Kecerdasan Artifisial  (AI) untuk Minimalisasi Fraud Biaya Operasional, Fraud biaya hukum dan biaya  penyitaan di unit collection dan remedial, meliputi pola umum, profil pelaku,  profil modus, pola melibatkan pihak eksternal dan transportasi luar kota

### Fitur 18: [AI Collection and Loss Reserve Forecast](#181-predict-loss-reverse)
> **PIC**: Dhoni Hanif Supriyadi  
**Definition:**  
Pembuatan Model  Memprediksi tagihan yang tidak tertagih dan kerugian yang terjadi dalam  portofolio tagihan perusahaan, termasuk kemungkinan biaya yang akan terjadi  dan kinerja tim penagihan, serta perkiraannya untuk 3-12 bulan kedepan.

### Fitur 19: AI Collection Rehabilitation
> **PIC**: Lusi Yustika Rachman  
**Definition:**  
Pembuatan model memprediksi  proses penanganan pelanggan yang pernah gagal membayar, dan  rekomendasi pengawasannya (siapa, bagaimana, frekuensi, intensitas,  informasi apa yang harus dipantau dan dikumpulkan) agar tidak kembali  macet.

## Endpoints
### 1.1. Predict Collection Difficulty Score
- **URL** : `/fitur-1/predict-difficulty-score/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": "3214104104104101",
  "debtor_name": "Merdi",
  "debtor_gender": "laki-laki",
  "debtor_birth_place": "Bekasi, 10 September 2001",
  "debtor_age": 22,
  "debtor_zip": "17520",
  "debtor_rt": "01",
  "debtor_rw": "01",
  "debtor_address": "Jl. Jend. Cappt No. 14\nBekasi, Jawa Barat 17520",
  "debtor_number": "+62 (0857) 141 4141",
  "bulan_1": 0,
  "bulan_2": 0,
  "bulan_3": 0,
  "bulan_4": 0,
  "bulan_5": 0,
  "bulan_6": 0,
  "bulan_7": 0,
  "bulan_8": 0,
  "bulan_9": 0,
  "bulan_10": 0,
  "bulan_11": 0,
  "bulan_12": 0,
  "loan_amount": 44745252,
  "tenor": 36,
  "billing_frequency": 27,
  "transaction_frequency": 6,
  "interest_rate": 0.19,
  "other_credit": 0,
  "monthly_income": 113185750.01,
  "home_ownership": true,
  "employment_status": "Employed",
  "employment_status_duration": 193,
  "public_records": 0,
  "contact_frequency": 18,
  "dti": 0.36,
  "revolving_credit_balance": 358526673,
  "occupation": "Chef",
  "financial_situation": "E",
  "delay_history": 0,
  "amount_of_late_days": 1583,
  "number_of_dependents": 5,
  "number_of_vehicles": 4,
  "number_of_complaints": 0,
  "external_factors": 0,
  "communication_channel": "Telepon",
  "working_time": "Pagi-Sore",
  "payment_patterns": "Awal Bulan",
  "payment_method": "ATM",
  "aging": 0
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": {
    "collection_difficulty_score": 311.00460515797107,
    "collection_difficulty_category": "sedang"
  }
}
```

### 2.1. Predict Collector Profile
- **URL** : `/fitur-2/collectorlabel/`
- **Method** : POST
- **Request Body** :
```json
{
  "collector_ID": 3276024508020006,
  "collector_gender": "Perempuan",
  "collector_age": 30,
  "collector_location": "Jalan Wortel RT 02/ RW 05, Pekanbaru, Riau",
  "collector_monthly_income": 5500000,
  "collector_occupation": "Pegawai Negeri",
  "collector_marital_status": "Belum Menikah",
  "collector_education_level": "S1",
  "collector_vehicle": "Motor",
  "negotiation_skills": "Baik",
  "proficient_technology_use": "Cukup Baik",
  "flexible_work": "Kurang Fleksibel",
  "communication_channel": "SMS",
  "responsive_rate": 70,
  "collector_personality": "Tegas",
  "working_time": "Pagi-Sore",
  "debtor_feedback": "Cukup",
  "year_experience": 5,
  "experience_level": "Cukup Berpengalaman",
  "success_rate": 60,
  "handled_case_category": "Kredit Kendaraan",
  "handled_case_count": 10,
  "negotiation_result": "Rencana pembayaran alternatif"
}
```
- **Example Response** :
```json
{
"status": 200,
"message": "success",
"result": {
    "age_label": "Muda",
    "location_label": "Muda",
    "behavior_label": "Disiplin",
    "character_label": "Tegas",
    "collector_field_label": "Meja",
    "ses_label": "D",
    "demography_label": "Pemukiman Padat Penduduk"
  }
}
```

### 2.2. Predict Debtor Profile
- **URL** : `/fitur-2/debtorlabel/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_age":"25",
  "debtor_gender":"Perempuan",
  "debtor_nik":"3276024508020006",
  "debtor_address": "Jalan Kapitan Raya RT 08/ RW 01, Sukatani, Tapos, Depok, Jawa Barat",
  "debtor_marital_status": "Belum Menikah",
  "debtor_number_of_dependents": 1,
  "debtor_education_level": "SMA",
  "debtor_occupation": "Pegawai Negeri",
  "debtor_monthly_income": 5000000,
  "debtor_monthly_expense": 3000000, 
  "debtor_asset_ownership": "Tidak",
  "debtor_communication_channel": "Telepon",
  "risk_credit": "Baik",
  "payment_pattern": "Pembayaran Tepat Waktu",
  "payment_method": "ATM",
  "tenor": 12,
  "interest_rate": 0.1,
  "monthly_payment": 7400000,
  "loan_amount": 20000000,
  "loan_type": "Pinjaman Pribadi",
  "working_time": "Pagi-Sore",
  "debtor_personality": "Ramah",
  "history_jan": 0,
  "history_feb": 1,
  "history_mar": 0,
  "history_apr": 0,
  "history_may": 1,
  "history_jun": 1,
  "history_jul": 0,
  "history_aug": 2,
  "history_sep": 1,
  "history_oct": 3,
  "history_nov": 1,
  "history_dec": 0,
  "debtor_aging": "DPK",
  "transaction": 8,
  "remaining_loan":7400000,
  "billing_type": "Telepon"
}
```
- **Example Response** :
```json
{
"status": 200,
"message": "success",
"result": {
    "age_label": "Muda",
    "location_label": "Jawa",
    "behavior_label": "Disiplin",
    "character_label": "Bijaksana",
    "collector_field_label": "Melalui Meja",
    "ses_label": "C",
    "demography_label": "Rusun"
  }
}
```

### 3.1. Predict Collection Strategy
- **URL** : `/fitur-3/predict-collection-strategy/`
- **Method** : POST
- **Request Body** :
```json
{
    "debtor_nik": 5837029610457921,
    "debtor_name": "Ani Putri",
    "debtor_gender": "perempuan",
    "debtor_birth_place": "Jawa Timur, 23-02-1996",
    "debtor_age": 27,
    "debtor_address": "Jl. Sudirman No. 5, Surabaya, Jawa Timur 60111",
    "debtor_zip": 60111,
    "debtor_rt": 3,
    "debtor_rw": 6,
    "debtor_marital_status": "Menikah",
    "debtor_occupation": "Pegawai Swasta",
    "debtor_company": "PT Indofood",
    "debtor_number": "+62 814-6887-1718",
    "collection_day_type": "Hari kerja",
    "loan_amount": 35000000,
    "debtor_education_level": "S-1",
    "credit_score": 650,
    "aging": "Dalam Perhatian Khusus",
    "previous_collection_status": "Berhasil",
    "previous_payment_status": "Terlambat",
    "amount_of_late_days": 7,
    "tenure": 24,
    "debtor_latitude": -6.947324,
    "debtor_longitude": 110.688436
}
```
- **Example Response** :
```json
{
    "status": 200,
    "message": "success",
    "result": {
      "best_collection_time": "pagi, malam",
      "best_collection_method": "Telepon",
      "best_collector_id": 523,
      "best_collector_name": "Donna Hoover",
      "best_collector_distance_to_debtor_in_km": 9.597850484422
    }
}
```

### 4.1. Predict Campaign
- **URL** : `/fitur-4/predict-assignment/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": 4498921163765183,
  "debtor_name": "Melinda Saefullah",
  "debtor_gender": "laki-laki",
  "debtor_birth_place": "Sumatera Barat, 14-07-1997",
  "debtor_age": 26.0,
  "debtor_address": "Jalan Medokan Ayu No. 8\nBitung, GO 56210",
  "debtor_zip": 56210,
  "debtor_rt": 15.0,
  "debtor_rw": 17.0,
  "debtor_marital_status": "Cerai mati",
  "debtor_occupation": "Pegawai Swasta",
  "debtor_company": "CV Aryani Tbk",
  "debtor_number": "+62-194-201-6618",
  "total_text": 15.0,
  "total_calls": 8.0,
  "total_visit": 1.0,
  "foreclosure": 0.0,
  "write_off": 0.0,
  "debtor_location": "Denpasar",
  "debtor_number_of_dependents": 4.0,
  "debtor_cars_count": 3.0,
  "debtor_motorcycle_count": 2.0,
  "debtor_house_count": 1.0,
  "debtor_education_level": "S1",
  "monthly_income": 582758265.0,
  "debtor_personality": "kalem",
  "loan_amount": 4406109356.0,
  "tenor": 12.0,
  "interest_rate": 9.0,
  "delay_history": 10.0,
  "credit_score": 648.0,
  "action_code": "WRPH",
  "category_campaign": "Call"
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": "Lakukan Telfon"
}
```

### 4.2. Predict Assignment
- **URL** : `/fitur-4/predict-campaign/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": 4498921163765183,
  "debtor_name": "Melinda Saefullah",
  "debtor_gender": "laki-laki",
  "debtor_birth_place": "Sumatera Barat, 14-07-1997",
  "debtor_age": 26.0,
  "debtor_address": "Jalan Medokan Ayu No. 8\nBitung, GO 56210",
  "debtor_zip": 56210,
  "debtor_rt": 15.0,
  "debtor_rw": 17.0,
  "debtor_marital_status": "Cerai mati",
  "debtor_occupation": "Pegawai Swasta",
  "debtor_company": "CV Aryani Tbk",
  "debtor_number": "+62-194-201-6618",
  "total_text": 15.0,
  "total_calls": 8.0,
  "total_visit": 1.0,
  "foreclosure": 0.0,
  "write_off": 0.0,
  "debtor_location": "Denpasar",
  "debtor_number_of_dependents": 4.0,
  "debtor_cars_count": 3.0,
  "debtor_motorcycle_count": 2.0,
  "debtor_house_count": 1.0,
  "debtor_education_level": "S1",
  "monthly_income": 582758265.0,
  "debtor_personality": "kalem",
  "loan_amount": 4406109356.0,
  "tenor": 12.0,
  "interest_rate": 9.0,
  "delay_history": 10.0,
  "credit_score": 648.0,
  "action_code": "WRPH",
  "category_campaign": "Call"
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": "Danielle Hamilton"
}
```

### 5.1. Predict Best Time to Bill
- **URL** : `/fitur-5/predict-best-time-to-bill/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik":1234567890987654,
  "debtor_name":"Ibun Bunaiya",  
  "debtor_gender": "Laki-Laki",
  "debtor_birth_place": "Semarang, 15-05-1968",
  "debtor_age": 55,
  "debtor_education_level": "S2",
  "debtor_address": "123 Main Street",
  "debtor_latitude": 40.7128,
  "debtor_longitude": -74.0060,
  "debtor_zip": "12345",
  "debtor_rt": "3",
  "debtor_rw": "4",
  "employment_status": "Full-Time",
  "debtor_working_day": "Senin-Jumat",
  "debtor_working_time": "Pagi-Siang",
  "tenor": 53
}
```
- **Example Response** :
```json
{
    "status": 200,
    "message": "success",
    "result": {
      "best_time_to_bill": "Sore",
      "best_time_to_bill_probability": 
      [
        "Sore : 60.19%",
        "Malam : 39.81%",
        "Pagi : 0.00%",
        "Siang : 0.00%"
      ]
    }
}
```

### 5.2. Predict Recommendation Collectors for Debtor
- **URL** : `/fitur-5/predict-recommended-collectors-assignments/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik":1234567890987654,
  "debtor_name":"Ibun Bunaiya",  
  "debtor_gender": "Laki-Laki",
  "debtor_birth_place": "Semarang, 15-05-1968",
  "debtor_age": 55,
  "debtor_education_level": "S2",
  "debtor_address": "123 Main Street",
  "debtor_latitude": 40.7128,
  "debtor_longitude": -74.0060,
  "debtor_zip": "12345",
  "debtor_rt": "3",
  "debtor_rw": "4",
  "employment_status": "Full-Time",
  "debtor_working_day": "Senin-Jumat",
  "debtor_working_time": "Pagi-Siang",
  "tenor": 53
}
```
- **Example Response** :
```json
{
    "status": 200,
    "message": "success",
    "result": {
      "recommended_collectors_to_assign": 
      [
        "Aaron Smith",
        "Richard Wade",
        "Anna Lin",
        "Isaac Dunn",
        "Andrew Murray II",
        "Glenn Little",
        "Kathy Riggs",
        "John Lopez",
        "Andrew Tyler",
        "Kelly Young"
      ]
    }
}
```

### 5.3. Predict Interaction Efficiency Cluster
- **URL** : `/fitur-5/predict-interaction-efficiency-cluster/`
- **Method** : POST
- **Request Body** :
```json
{
  "collector_name": "Josko Gvardiol",
  "debtor_name": "Jokowi Dodo",
  "collector_latitude": -6.2570919,
  "collector_longitude": 106.572494,
  "debtor_latitude": -6.249463349,
  "debtor_longitude": 107.17905682,
  "departure_time": "08:05",
  "arrival_time": "12:34",
  "transportation_type": "Mobil",
  "call_pickup_duration": 18,
  "door_opening_duration": 32,
  "connection_time": 35,
  "waiting_response_duration": 7405,
  "idle_duration": 1141,
  "nonproductive_duration": 4850
}
```
- **Example Response** :
```json
{
    "status": 200,
    "message": "success",
    "result": {
      "category_cluster": "Efisien dalam Respon dan Interaksi"
    }
}
```

### 7.1. Predict Workload Score
- **URL** : `/fitur-7/predict-workload/`
- **Method** : POST
- **Request Body** :
```json
{
  "collector_name":"Mykhailo Mudryk",
  "collector_address":"Jalan Mawar No. 40\nMetro, Jawa Timur 53794",
  "collector_number":"+62 (097) 530-5219",
  "collector_nik":6952660116035317,
  "collector_zip":53794,
  "collector_rt":1,
  "collector_rw":9,
  "collector_birth_place":"DI Yogyakarta, 20-08-1998",
  "collector_marital_status":"Cerai hidup",
  "collector_gender":"laki-laki",
  "collector_id":"C1001",
  "team_id":"T0201",
  "success_rate":0.3370757033,
  "campaigns_count":110,
  "most_campaign_count":28,
  "collection_difficulty_score_avg":608.3623727332,
  "collector_age":32,
  "weekly_collector_working_hours":50.096235135,
  "weekly_debtor_volume_handled":13,
  "weekly_call_contacts_hours":12.3248313882,
  "weekly_digital_messages_hours":19.5491412667,
  "weekly_field_action_hours":17.6727666326,
  "digital_campaigns_count":35,
  "call_campaigns_count":13,
  "field_campaigns_count":5,
  "team_resolution_time_avg":5.7241665016,
  "team_first_time_resolution_rate":0.3092629799,
  "team_escalation_rate":0.2751522376,
  "team_member_cases_total":64,
  "team_response_time_avg":9.4055413923
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": {
    "workload_score": 524.8029049137016,
    "workload_level": "normal"
  }
}
```

### 7.2. Recommend Campaign
- **URL** : `/fitur-7/recommend-campaign`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik":5233673178135086,
  "debtor_name":"R.A. Yessi Hidayat",
  "debtor_gender":"perempuan",
  "debtor_birth_place":"Sulawesi Selatan, 15-01-1987",
  "debtor_age":36,
  "debtor_address":"Jalan Dipatiukur No. 553\nPalangkaraya, Nusa Tenggara Timur 47573",
  "debtor_zip":47573,
  "debtor_rt":11,
  "debtor_rw":10,
  "debtor_marital_status":"Belum menikah",
  "debtor_occupation":"Profesional",
  "debtor_company":"CV Kurniawan Wasita",
  "debtor_number":"+62 (059) 742 6115",
  "debtor_id":"D0316",
  "action_code":"INCO",
  "debtor_location":"Pekanbaru",
  "debtor_npl_history":111212222212,
  "aging":"DPK",
  "collector_name":"Stephen Lee",
  "collector_address":"Gg. Dr. Djunjunan No. 4\nTarakan, JK 18302",
  "collector_number":"+62-090-511-3257",
  "collector_nik":6266983748483711,
  "collector_zip":18302,
  "collector_rt":9,
  "collector_rw":17,
  "collector_birth_place":"Kalimantan Barat, 30-03-1968",
  "collector_age":55,
  "collector_marital_status":"Belum menikah",
  "collector_gender":"perempuan",
  "workload_score":525.944086336,
  "workload_level":"normal",
  "collector_location":"Makassar",
  "collector_vehicle":"Punya",
  "collector_id":"C0160",
  "debtor_location_coord":"0.5262455:101.4515727",
  "collector_location_coord":"-5.1342962:119.4124282",
  "debtor_collector_distance":2082.8220212786
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": {
    "campaign_recommendation": "Digital",
    "aging": "DPK"
  }
}
```

### 7.3. Recommend Field Collector
- **URL** : `/fitur-7/recommend-field-collector/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik":6835138002000107,
  "debtor_name":"Darmana Wastuti, S.I.Kom",
  "debtor_gender":"perempuan",
  "debtor_birth_place":"Jawa Tengah, 12-02-1998",
  "debtor_age":25,
  "debtor_address":"Jalan Gegerkalong Hilir No. 37\nPalangkaraya, Sulawesi Barat 35394",
  "debtor_zip":35394,
  "debtor_rt":6,
  "debtor_rw":15,
  "debtor_marital_status":"Cerai mati",
  "debtor_occupation":"Buruh",
  "debtor_company":"PD Winarno Anggriawan Tbk",
  "debtor_number":"+62 (0289) 208-9883",
  "debtor_id":"D1000",
  "action_code":"HUPD",
  "debtor_location":"Bandung",
  "debtor_npl_history":122333343232,
  "aging":"Kurang Lancar",
  "debtor_location_coord":"-6.9215529:107.6110212"
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": {
    "collector_name": "Jill Mcguire",
    "collector_address": "Gg. Pasir Koja No. 012\nSabang, Kalimantan Barat 02809",
    "collector_number": "+62-0218-050-2423",
    "collector_nik": 5742248919574299,
    "collector_zip": 2809,
    "collector_rt": 15,
    "collector_rw": 6,
    "collector_birth_place": "Kalimantan Selatan, 03-04-1995",
    "collector_age": 28,
    "collector_marital_status": "Menikah",
    "collector_gender": "laki-laki",
    "collector_location": "Bandung",
    "collector_vehicle": "Punya",
    "collector_id": "C0570",
    "collector_location_coord": "-6.9215529:107.6110212",
    "workload_score": 91.34761187600088
  }
}
```

### 8.1. Predict Cost Effectiveness
- **URL** : `/fitur-8/predict-CostEffectivenessPrediction/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_name": 7922460216135313,
  "CreditScore": 650,
  "Gender": 0,
  "debtor_age": 42,
  "Tenure": 10,
  "arrear_amount": 5,
  "loan_amount": 199992480,
  "debtor_occupation": 4
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": {
    "Effecientcosteffectiveness_score": 1,
    "Effecientcosteffectiveness_label": "NO Efficiency & Effectiveness"
  }
}
```

### 9.1. Predict Recommended Solution
- **URL** : `/fitur-9/predict-recomended-solution/kredit-pinjaman/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_name": "Maya Aulia",
  "debtor_gender": "perempuan",
  "debtor_birth_place": "Jawa Tengah, 13-04-1964",
  "debtor_age": 24,
  "debtor_marital_status": "Menikah",
  "debtor_number_of_dependents": 0,
  "debtor_education_level": "S2",
  "employment_year": 5,
  "monthly_income": 5500000,
  "monthly_expenses": 3400000,
  "asset_value": 270000000,
  "down_payment_percentage": 0.2,
  "paid_down_payment": 6000000,
  "loan_amount": 24000000,
  "tenor": 72,
  "monthly_payment": 457463.13,
  "loan_purpose": "kredit usaha"
}
```
- **Example Response** :
```json

```

### 11.1. Predict Default Solution (Kredit Pinjaman)
- **URL** : `/fitur-11/predict-default-solution/kredit-pinjaman/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": "4133424516345914",
  "debtor_name": "Muh. Dzaky",
  "debtor_gender": "laki-laki",
  "debtor_birth_place": "Bengkulu, 23-09-1997",
  "debtor_age": 40,
  "debtor_address": "Jalan Raya Setiabudhi No. 857\nBanjarbaru, KR ...",
  "debtor_zip": 12188,
  "debtor_rt": 6,
  "debtor_rw": 10,
  "debtor_marital_status": "Menikah",
  "debtor_occupation": "Pengusaha",
  "debtor_company": "CV Nugroho Melani (Persero) Tbk",
  "debtor_number": "+62 (102) 776 3467",
  "debtor_number_of_dependents": 5,
  "debtor_education_level": "SMA",
  "employment_year": 18,
  "residential_zone": "Zona 9: zona daerah industri sub urban",
  "criminality_level": "Tingkat Kriminalitas Tinggi",
  "violence_level": "Tingkat Kekerasan Sedang",
  "gangster_culture": "Sedang",
  "monthly_income": 6000000,
  "monthly_expenses": 2900000,
  "asset_value": 290000000.0,
  "collateral_offered": "Yes",
  "loan_amount": 37100000,
  "interest_rate": 12,
  "tenor": 60,
  "monthly_payment": 825269.01,
  "loan_purpose": "Kebutuhan darurat"
}
```
- **Example Response** :
```json

```

### 11.2. Predict Default Solution (Kredit Benda)
- **URL** : `/fitur-11/predict-default-solution/kredit-benda/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": "7834621118366601",
  "debtor_name": "Maya Aulia ",
  "debtor_gender": "perempuan",
  "debtor_birth_place": "Jawa Tengah, 13-04-1964",
  "debtor_age": 24,
  "debtor_address": "Jalan Cihampelas No. 4\nPekanbaru, Sulawesi Selatan",
  "debtor_zip": "56126",
  "debtor_rt": 20,
  "debtor_rw": 20,
  "debtor_marital_status": "Menikah",
  "debtor_occupation": "Pengusaha",
  "debtor_company": "UD Hidayat",
  "debtor_number": "+62 (808) 299 5117",
  "debtor_number_of_dependents": 0,
  "debtor_education_level": "S2",
  "employment_year": 5,
  "residential_zone": "Zona 4: zona permukiman kelas menengah",
  "criminality_level": "Tingkat Kriminalitas Sedang",
  "violence_level": "Tingkat Kekerasan Tinggi",
  "gangster_culture": "Rendah",
  "monthly_income": 5500000,
  "monthly_expenses": 3400000,
  "asset_value": 270000000.0,
  "down_payment_percentage": 0.2,
  "paid_down_payment": 6000000,
  "loan_amount": 24000000,
  "interest_rate": 6,
  "tenor": 60,
  "monthly_payment": 457463.13,
  "ltv": 75.0,
  "loan_purpose": "kendaraan bermotor"
}
```
- **Example Response** :
```json

```

### 12.1. Predict Recommendation Tenor
- **URL** : `/fitur-12/recommendation-tenor/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": 4133431716345918, 
  "debtor_name": "Cemplunk Zulaika", 
  "debtor_gender": "perempuan", 
  "debtor_birth_place": "Bengkulu, 23-09-1997", 
  "debtor_address": "Jalan Raya Setiabudhi No. 857\nBanjarbaru, KR 12188", 
  "debtor_zip": 12188, 
  "debtor_rt": 6, 
  "debtor_rw": 10, 
  "debtor_marital_status": "Menikah", 
  "debtor_company": "CV Nugroho Melani (Persero) Tbk", 
  "debtor_number": "+62 (102) 776 3467", 
  "debtor_id": 180034427373791, "debtor_age": 32, 
  "debtor_occupation": "Pengusaha", 
  "monthly_income": 50100000, 
  "debt": 7925000.0, 
  "dti": 0.1581836327345309, 
  "monthly_expenses": 3100000, 
  "net_income": 47000000, 
  "asset_value": 89500000, 
  "previous_credit_monthly_income": 26900000, 
  "financial_changes": 23200000, 
  "business_conditions": 35, 
  "credit_score": 414, 
  "amount_of_late_days": 61, 
  "credit_goals": "Investasi", 
  "collateral": 52600000, 
  "interest_rate": 11.101694915254235, 
  "number_of_dependents": 0, 
  "old_tenor": 33, 
  "old_collateral": 32800000, 
  "fee_installments": 100000, 
  "ltv": 0.8927985948477751, 
  "remaining_loan": 39800000, 
  "arrear_amount": 2, 
  "monthly_income_family": 44200000, 
  "monthly_expense_family": 24700000, 
  "family_health": "Tidak sehat", 
  "asset_value_family": 38800000, 
  "old_monthly_payments": 2566954.160246533, 
  "loan_amount": 76245000.0
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": [
      {
          "debtor_name": "Cemplunk Zulaika",
          "debtor_nik": 4133431716345918,
          "debtor_id": 180034427373791,
          "recomendation_tenor": 164.00000000000017,
          "recomendation_monthly_payments": 516521.26395204576
      }
  ]
}
```

### 12.2. Predict Request Loan
- **URL** : `/fitur-12/request-loan/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_nik": 4133431716345918, 
  "debtor_name": "Cemplunk Zulaika", 
  "debtor_gender": "perempuan", 
  "debtor_birth_place": "Bengkulu, 23-09-1997", 
  "debtor_address": "Jalan Raya Setiabudhi No. 857\nBanjarbaru, KR 12188", 
  "debtor_zip": 12188, 
  "debtor_rt": 6, 
  "debtor_rw": 10, 
  "debtor_marital_status": "Menikah", 
  "debtor_company": "CV Nugroho Melani (Persero) Tbk", 
  "debtor_number": "+62 (102) 776 3467", 
  "debtor_id": 180034427373791, 
  "debtor_age": 32, 
  "debtor_occupation": "Pengusaha", 
  "monthly_income": 50100000, 
  "debt": 7925000.0, 
  "dti": 0.1581836327345309, 
  "monthly_expenses": 3100000, 
  "net_income": 47000000, 
  "previous_credit_monthly_income": 26900000, 
  "financial_changes": 23200000, 
  "business_conditions": 35, 
  "asset_value": 89500000, 
  "interest_rate": 11.101694915254235, 
  "collateral": 52600000, 
  "old_collateral": 32800000, 
  "ltv": 0.8927985948477751, 
  "tenor": 164
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": [
    {
      "debtor_name": "Cemplunk Zulaika",
      "debtor_nik": 4133431716345918,
      "debtor_id": 180034427373791,
      "request_loan": 76244999.99999997
    }
  ]
}
```

### 17.1. Predict Fraud
- **URL** : `/fitur-17/predict-fraud/`
- **Method** : POST
- **Request Body** :
```json
{
	"debtor_nik":3285167540917312,
	"debtor_name":"Erik Ten Hag",
	"debtor_gender":"laki-laki",
	"debtor_birth_place":"Haaksbergen, 02-02-1970",
	"debtor_age":53,
	"debtor_address":"Gang Setiabudhi No. 8\nTangerang, NT 42062",
	"debtor_zip":42062,
	"debtor_rt":13,
	"debtor_rw":4,
	"debtor_marital_status":"Menikah",
	"num_of_dependents":2,
	"debtor_education_level":"S2",
	"debtor_occupation":"Professional",
	"debtor_company":"Manchester United F.C.",
	"debtor_number":"+62 (103) 543 9582",
	"monthly_income":9812845,
	"monthly_expense":3785491,
	"loan_type":"Pinjaman Bisnis",
	"debtor_tenor":18,
	"loan_amount":15053912,
	"payment_history_jan":0,
	"payment_history_feb":0,
	"payment_history_mar":1,
	"payment_history_apr":2,
	"payment_history_may":1,
	"payment_history_jun":1,
	"payment_history_jul":0,
	"payment_history_aug":1,
	"payment_history_sep":2,
	"payment_history_oct":3,
	"payment_history_nov":2,
	"payment_history_dec":1,
	"debtor_aging":"DPK",
	"transaction_frequency":11,
	"remaining_loan":1589234,
	"billing_frequency":1,
	"billing_type":"Surat",
	"debtor_response":"Positif",
	"consumerism_level":"Rendah",
	"debtor_mode":"Membayar utang",
	"collector_type":"Komunikasi",
	"collector_reputation":"Baik",
	"collector_fee":5493410,
	"transportation_type":"Mobil",
	"distance":543,
	"travel_frequency":1,
	"transportation_cost":430102
}
```
- **Example Response** :
```json
{
  "status": 200,
  "message": "success",
  "result": {
    "fraud_score": 166.2346405254836,
    "fraud_label": "No Fraud"
  }
}
```

### 18.1. Predict Loss Reverse
- **URL** : `/fitur-18//predict-loss-reverse/`
- **Method** : POST
- **Request Body** :
```json
{
  "debtor_name": "Alika Januar",
  "debtor_nik": 4595204238824022,
  "debtor_address": "Gang Setiabudhi No. 8 Tangerang, NT 42062",
  "debtor_zip": 42062,
  "debtor_rt": 13,
  "debtor_rw": 4,
  "debtor_birth_place": "Sulawesi Tengah, 03-03-1999",
  "debtor_age": 24,
  "debtor_number": "+62 (08) 963 3964",
  "debtor_occupation": "Pegawai Swasta",
  "debtor_marital_status": "Cerai hidup",
  "debtor_company": "CV Oktaviani Wulandari",
  "debtor_gender": "perempuan",
  "debtor_education_level": "D1",
  "employment_type": "Retired",
  "number_of_dependents": 4,
  "net_income": 18500000.0,
  "payment_date": "2023-09-03",
  "loan_amount": 81100000,
  "tenor": 3,
  "amount_of_late": 118,
  "late_payment_amount": 9,
  "credit_score": 235,
  "arrears_amounts": 2,
  "aset": 4000000,
  "bil1_late1": 0,
  "bill_late2": 21,
  "bill_late3": 26,
  "bill_late4": 10,
  "bill_late5": 16,
  "bill_late6": 3,
  "bill_late7": 16,
  "bill_late8": 10,
  "bill_late9": 0,
  "bill_late10": 14,
  "bill_late11": 2,
  "bill_late12": 0,
  "arrears1": 0,
  "arrears2": 1,
  "arrears3": 2,
  "arrears4": 0,
  "arrears5": 1,
  "arrears6": 0,
  "arrears7": 1,
  "arrears8": 0,
  "arrears9": 0,
  "arrears10": 0,
  "arrears11": 0,
  "arrears12": 0,
  "aging": "Kurang lancar"
}
```
- **Example Response** :
```json

```

## Contributors
A big shout-out and thanks to the amazing individuals who have contributed to this project:

- [ABDUL ROID](https://github.com/abroid)
- [AHMAD ALFARUQ](https://github.com/0KNV1)
- [ANUGRAH AIDIN YOTOLEMBAH](https://github.com/AnugrahAidinYotolembah)
- [BREV WILLIAM FIDEN SARAGIH](https://github.com/brevliam)
- [DHONI HANIF SUPRIYADI](https://github.com/dhonihanif)
- [DWI DUTA MAHARDEWANTORO](https://github.com/dutaaamahar)
- [LUSI YUSTIKA RACHMAN](https://github.com/https://github.com/LusiYustikaRachman)
- [MUHAMMAD OTHMAN LUTFI](https://github.com/othmanlutfii)
- [MUHAMMAD REYHAN FITRIYAN](https://github.com/mreyhanf)
- [NUR AZKIA RAHMAH](https://github.com/azkiaarh)
- [PUTRA AL FARIZI](https://github.com/PutraAlFarizi15)
- [RONI MERDIANSAH](https://github.com/Dapperson)
- [ROVIANI AMELIA](https://github.com/rovianiameliaa)
- [SINTYA TRI WAHYU ADITYAWATI](https://github.com/sintya1234)
- [YOGA FATWANTO](https://github.com/yogafatwanto)