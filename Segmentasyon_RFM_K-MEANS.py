import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

# --- 1. AYARLAR VE VERİ ÇEKME (ENCODING & CURSOR FIX) ---
import os

# GİZLİLİK PROTOKOLÜ: Şifreler kodun içinde değil, sistem çevre değişkenlerinden güvenle okunur.
db_params = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "supply_chain_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Gs.20021905"),
    "port": "5432"
}
engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

print("🚀 Veriler çekiliyor (Tüm sistemler kontrol ediliyor)...")
try:
    conn = psycopg2.connect(**db_params)
    conn.set_client_encoding('UTF8')
    cur = conn.cursor()

    # Analiz için fact_sales
    cur.execute("SELECT customer_id, order_date, sales, order_id, is_late, profit FROM fact_sales")
    df = pd.DataFrame(cur.fetchall(), columns=['customer_id', 'order_date', 'sales', 'order_id', 'is_late', 'profit'])

    # Tüm sütunlar için cleaned_supply_chain
    cur.execute("SELECT * FROM cleaned_supply_chain")
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df_ham = pd.DataFrame(rows, columns=colnames)

    cur.close()
    conn.close()
    print(f"✅ Veri okuma başarılı! Ham Tablo: {len(df_ham.columns)} sütun.")
except Exception as e:
    print(f"❌ Kritik Veri Çekme Hatası: {e}")

# --- 2. RFM METRİKLERİ VE SKORLAMA ---
df['order_date'] = pd.to_datetime(df['order_date'])
today_date = df['order_date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (today_date - x.max()).days,
    'order_id': lambda x: x.nunique(),
    'sales': lambda x: x.sum(),
    'is_late': 'mean',
    'profit': 'sum'
})
rfm.columns = ['recency', 'frequency', 'monetary', 'late_rate', 'total_profit']

# Skorlama
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RFM_SCORE"] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

# Segment İsimlendirme
seg_map = {
    r'[1-2][1-2]': 'hibernating', r'[1-2][3-4]': 'at_Risk', r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep', r'33': 'need_attention', r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising', r'51': 'new_customers', r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['rfm_segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

print("\n" + "=" * 20 + " DURAK 1: RFM SEGMENTLERİ (İLK 10) " + "=" * 20)
print(rfm[['rfm_segment', 'RFM_SCORE', 'monetary']].head(10))

# --- 3. OPTİMAL K ANALİZİ (ELBOW & SILHOUETTE) ---
print("\n🧪 Kümeleme başarımı analiz ediliyor...")
scaler = StandardScaler()
features = ['recency', 'frequency', 'monetary', 'late_rate', 'total_profit']
X_scaled = scaler.fit_transform(rfm[features])

inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_scaled)
    inertia.append(kmeans_test.inertia_)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k} için Silhouette Skoru: {score:.4f}")

# Elbow Plot Grafiği
plt.switch_backend('Agg')
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o", color="blue", linestyle="--")
plt.xlabel("Küme Sayısı (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Optimal K Seçimi")
plt.savefig('optimal_k_elbow_plot.png')
plt.close()
print("✅ 'optimal_k_elbow_plot.png' kaydedildi.")

# --- 4. K-MEANS KÜMELEME ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# --- 5. İSİMLENDİRME ÖNCESİ İSTATİSTİKSEL RÖNTGEN ---
print("\n" + "=" * 40 + " 🔬 KÜME BAZLI İSTATİSTİKSEL ANALİZ (MEAN, MEDIAN, STD) " + "=" * 40)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Her kümenin detaylı analizi
print(rfm.groupby("kmeans_cluster")[features].agg(["mean", "median", "std", "count"]).round(2))

# --- 6. KÜMELERE İSİM VERME VE ÇAPRAZ TABLO ---
cluster_names = {
    0: 'VIP_Champions', 1: 'New_High_Risk_Late', 2: 'Low_Value_Efficient',
    3: 'Lost_Loyalists', 4: 'High_Volume_Low_Margin'
}
"""
0: VIP_Champions: Hem çok alıyor hem de kârlı.
1: New_High_Risk_Late: Yeni gelmiş ama lojistik olarak sıkıntılı (belki bölgesi zorlu).
2: Low_Value_Efficient: Az alıyor ama operasyonu sorunsuz.
3: Lost_Loyalists: Eskiden sadıktı ama şu an uyuyor.
4: High_Volume_Low_Margin: Çok ürün alıyor ama bize kâr bırakmıyor (belki sürekli indirim kovalıyor).
"""
"""
Hangi müşterinin lojistikten dolayı mağdur olduğunu (is_late),
Hangisinin şirkete kâr yerine zarar yazdığını (profit),
Hangi bölgede hangi kümenin yoğunlaştığını biliyorsun.
"""



rfm['cluster_name'] = rfm['kmeans_cluster'].map(cluster_names)

print("\n" + "=" * 40 + " RFM SEGMENT vs K-MEANS CLUSTER ÇAPRAZ TABLO " + "=" * 40)
print(pd.crosstab(rfm['rfm_segment'], rfm['cluster_name']))

# --- 7. TABLOLARI BİRLEŞTİRME (DATA TYPE FIX) ---
print("\n🔗 Tablolar birleştiriliyor (Master Dataset)...")
df_ham['customer_id'] = pd.to_numeric(df_ham['customer_id'], errors='coerce')
rfm.index = pd.to_numeric(rfm.index, errors='coerce')

df_final = df_ham.merge(rfm[['rfm_segment', 'kmeans_cluster', 'cluster_name']], on='customer_id', how='left')

# --- 8. SQL'E MÜHÜRLEME ---
try:
    conn = psycopg2.connect(**db_params)
    conn.set_client_encoding('UTF8')
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS final_master_supply_chain;")

    cols_query = ", ".join([f'"{c}" TEXT' for c in df_final.columns])
    cur.execute(f"CREATE TABLE final_master_supply_chain ({cols_query});")

    data_list = [tuple(x) for x in df_final.values]
    placeholders = ", ".join(["%s"] * len(df_final.columns))
    cur.executemany(f"INSERT INTO final_master_supply_chain VALUES ({placeholders})", data_list)

    conn.commit()
    print("\n🔥 İŞLEM TAMAM! Dev Master Tablo SQL'e mühürlendi.")
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ SQL Yazma Hatası: {e}")

# --- 9. FİNAL: TÜM DATANIN HEAD() ÇIKTISI ---
print("\n" + "=" * 60)
print(f" 🚀 EFSANE FİNAL TABLO ÖNİZLEME (Toplam Sütun: {len(df_final.columns)}) ")
print("=" * 60)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 15)
print(df_final.head(20))


"""Sadece RFM'e baksaydım, mesela 1. satırdaki müşteriye "Potansiyel Sadık" deyip geçecektik. Ama K-Means sayesinde onun
 aslında "Lojistik Mağduru" olduğunu da gördüm hemen bir özür kuponu gönderelim"""


