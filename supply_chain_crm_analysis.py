import pandas as pd
import numpy as np
import datetime as dt
import psycopg2
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100000)

import os

db_params = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "supply_chain_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),  
    "port": os.getenv("DB_PORT", "5432")
}
conn = psycopg2.connect(**db_params)
conn.set_client_encoding('UTF8')
df = pd.read_sql_query("SELECT * FROM final_master_supply_chain", conn)
conn.close()

# 2. AYKIRI DEĞERLER İÇİN EŞİK BELİRLEME
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Sayısal sütunları düzelt ve baskıla
num_cols = ['sales', 'profit_margin']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    replace_with_thresholds(df, col)

# 3. CRM METRİKLERİNİN (RFM Tahmin Yapısı) HAZIRLANMASI
df['order_date_dateorders'] = pd.to_datetime(df['order_date_dateorders'])
today_date = df['order_date_dateorders'].max() + dt.timedelta(days=2)

cltv_df = df.groupby('customer_id').agg({
    'order_date_dateorders': [
        lambda x: (x.max() - x.min()).days, # Recency (Kendi içinde)
        lambda x: (today_date - x.min()).days # T (Müşteri yaşı)
    ],
    'order_id': lambda x: x.nunique(), # Frequency
    'sales': lambda x: x.sum() # Monetary (Toplam)
})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
# Monetary'yi "işlem başına ortalama" yapalım
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# Modeller için haftalık dönüşüm
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
# Sadece tekrar eden müşterileri alalım (Frequency > 1)
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
print("📊 CRM Tahmin Tablosu Hazır!")
print(cltv_df.head())


# --- 4. BG-NBD MODELİNİN EĞİTİLMESİ ---
print("\n🧪 BG-NBD Modeli eğitiliyor (Gelecek işlem sayısı tahmini)...")
# penalizer_coef: Katsayıların aşırı büyümesini engelleyen ceza terimi.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 3 Ay (12 hafta) içinde beklenen en çok satın alma yapacak 10 müşteri
cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
print("\n📊 3 Ay İçinde En Çok İşlem Yapması Beklenen İlk 5 Müşteri:")
print(cltv_df.sort_values("expected_purc_3_month", ascending=False).head())


# --- 5. GAMMA-GAMMA MODELİNİN EĞİTİLMESİ ---
print("\n🧪 Gamma-Gamma Modeli eğitiliyor (Gelecek kâr tahmini)...")
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
# İşlem başına beklenen ortalama kâr
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
print("\n✅ İşlem başına beklenen kâr tahminleri hazır.")


# --- 6. 6 AYLIK CLTV TAHMİNİ ---
cltv_df["cltv_prediction"] = ggf.customer_lifetime_value(bgf,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T'],
                                                        cltv_df['monetary'],
                                                        time=6,     # 6 Aylık projeksiyon
                                                        freq="W",   # Haftalık bazda
                                                        discount_rate=0.01)

# Segmentlerin oluşturulması (A en değerli, D en az değerli)
cltv_df["segment"] = pd.qcut(cltv_df["cltv_prediction"], 4, labels=["D", "C", "B", "A"])
"""
odun sonunda oluşturduğun o ABCD segmentleri, sadece geçmiş satışlara göre değil, 6 aylık gelecek beklentisine göre yapıldı:

A Segmenti (Premium): Gelecekteki en değerli hazine. Onlara VIP hizmet verilmeli.
B Segmenti: Sadık ama kâr marjı artırılabilir.
C Segmenti: Kararsızlar. Doğru bir kampanya ile B'ye çıkabilirler.
D Segmenti: Maliyeti yüksek, getirisi düşük. Onlar için fazla reklam harcaması yapılmamalı.
Z-Single-Transaction: Sadece bir kez gelip gitmiş olanlar. Onları sisteme dahil etmek için "Hoş geldin" kampanyaları lazım.
"""
"""
SegmentTakma :A- Yıldızlar (Star Class)Gelecek 6 ayda 
en çok kârı bırakacak olan elit kitle.Onları el üstünde tutmak, özel lojistik hat tanımlamak.B Sadıklar 
(Loyals)Düzenli alışveriş yapan, kârlı ama A kadar "zengin" bırakmayanlar.Çapraz satış (Cross-sell)
yaparak onları A'ya taşımak lazım.C Potansiyeller (Potentials)Arada bir gelen, kârlılığı orta şekerli grup."
Daha fazla alırsak kargo bedava" gibi promosyonlarla canlandır.D Riskli / Düşük Değer (Low Value)Hem az harcıyor 
hem de nadir geliyor. Gelecek vaatleri çok düşük.Onlar için çok reklam harcaması yapılmamalı, maliyet korunmali.
ZTek Seferlikler (One-Shotters)Sisteme girmiş, bir kez almış ve bir daha uğramamış olanlar.Onları "Churn" (terk etmiş) kabul etme, 
geri kazanma kampanyası.
"""




print("\n🚀 FINAL CLTV TAHMİNLERİ VE SEGMENTLER (TOP 10)")
print(cltv_df.sort_values("cltv_prediction", ascending=False).head(10))

# Segmentlerin özet istatistikleri
print("\n📊 SEGMENT BAZLI ÖZET:")
print(cltv_df.groupby("segment").agg({"cltv_prediction": ["mean", "count", "sum"]}))


""""Sadece geçmiş satışlara bakmadım. BG-NBD ile müşterilerin satın alma frekansını, Gamma-Gamma ile
 işlem başına beklenen kârı modelledim. 6 aylık bir projeksiyon çıkararak müşterileri 4 segmente ayırdım.
  Örneğin; A segmentindeki müşterilerimizin kişi başı getirisi D segmentinden 4 kat fazla. 
  Bu sayede pazarlama ekibine hangi müşteriye 'ücretsiz kargo' tanımlaması gerektiğini, hangisine 'indirim kuponu' 
  verilmesi gerektiğini bilimsel olarak kanıtladım."""


# --- 7. TÜM SÜTUNLARI KAPSAYAN VE NaN BIRAKMAYAN MASTER BİRLEŞTİRME ---
print("\n🔗 Tahminler ve tüm CRM metrikleri birleştiriliyor, boşluklar mühürleniyor...")
try:
    # 1. cltv_df'deki TÜM sütunları (Recency, T, Frequency, Monetary, Tahminler) alıyoruz
    cltv_results = cltv_df.reset_index()
    # 2. Tip Eşitleme (Merge hatasını engellemek için metin formatına zorluyoruz)
    df['customer_id'] = df['customer_id'].astype(str)
    cltv_results['customer_id'] = cltv_results['customer_id'].astype(str)
    # 3. SOL BİRLEŞTİRME (Left Join): 180.519 satırı koru, yanına tüm analizleri ekle
    df_final_master = df.merge(cltv_results, on='customer_id', how='left')
    # --- 4. KRİTİK DÜZELTME: TÜM YENİ GELEN SÜTUNLARI TARAYALIM ---
    # customer_id dışındaki yeni eklenen tüm sütunları listeleyelim
    added_cols = cltv_results.columns.drop('customer_id').tolist()

    for col in added_cols:
        if col == 'segment':
            # Segmenti boş olanları (tek seferlikler) "Z-Single-Transaction" yap
            df_final_master[col] = df_final_master[col].astype(str).replace(['nan', 'None', ''], 'Z-Single-Transaction')
        else:
            # Sayısal olan her şeyi (T, recency, prediction vb.) 0.0 yapalim
            # Boşlukları (None/NaN) yakalar ve Power BI için sayısal 0'a çevirirelimn
            df_final_master[col] = pd.to_numeric(df_final_master[col], errors='coerce').fillna(0.0)

    print(f"✅ Birleştirme bitti! Sütun Sayısı: {len(df_final_master.columns)} | Satır Sayısı: {len(df_final_master)}")

    # --- 5. FINAL ADIM: VERİTABANI MİMARİSİNİ MÜHÜRLEME ---
    print("\n🏗️ Veritabanı mimarisi son haline getiriliyor...")

    import psycopg2
    from psycopg2.extras import execute_values
    import numpy as np

    conn = psycopg2.connect(**db_params)
    conn.set_client_encoding('UTF8')
    cur = conn.cursor()

    # Eski tabloları temizle
    print("🧹 Eski tablolar temizleniyor...")
    cur.execute("DROP TABLE IF EXISTS final_master_supply_chain CASCADE;")
    cur.execute("DROP TABLE IF EXISTS final_master_crm_prediction CASCADE;")
    cur.execute("DROP TABLE IF EXISTS supply_chain_analytics_master CASCADE;")

    # Yeni master tabloyu kur
    table_name = "supply_chain_analytics_master"
    columns = df_final_master.columns.tolist()

    # Sütun isimlerini çift tırnak içine alarak SQL hatalarını engelliyoruz
    col_definitions = ", ".join([f'"{c}" TEXT' for c in columns])
    create_query = f"CREATE TABLE {table_name} ({col_definitions});"
    cur.execute(create_query)

    print(f"💎 Yeni master tablo oluşturuldu: {table_name}")

    # Veriyi SQL uyumlu (NaN -> None) hale getirip toplu yükle
    df_final_master_fixed = df_final_master.replace({np.nan: None})
    data_values = [tuple(x) for x in df_final_master_fixed.to_numpy()]

    insert_query = f"INSERT INTO {table_name} ({', '.join([f'\"{c}\"' for c in columns])}) VALUES %s"

    print(f"⏳ {len(df_final_master)} satır tam kadro PostgreSQL'e mühürleniyor...")
    execute_values(cur, insert_query, data_values)

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n🚀 İŞLEM %100 BAŞARIYLA TAMAMLANDI!")
    print(f"💎 Veritabanındaki 'Akıllı Master Tablo': {table_name}")

except Exception as e:
    print(f"❌ Kritik Hata: {e}")


df_final_master.head(150)


#NOT:Shemaa  Python ile yepyeni bir sütun (Gecikme Olasılığı vb.) eklediğimizde, sadece View koduna bir satır ekleyecez ve tüm raporların saniyeler içinde güncellenecek.
# SQL tarafında ham tabloya dokunmadan, sanal tablolar üzerinden analiz yapmak
#Vri poestresql sutun tipleridogur olamdan iletildi falakt orda en dogru sekilde  duzldtidli. Sonrajki çalısmalarda temiz her sey dogru şir data ile  calısacagız...

