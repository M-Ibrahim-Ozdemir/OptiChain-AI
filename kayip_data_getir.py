# --- KAYIP TABLOYU (final_master_supply_chain) GERİ GETİRME OPERASYONU ---
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

print("🔄 Kayıp tablo geri yükleniyor...")

db_params = {
    "host": "localhost",
    "database": "supply_chain_db",
    "user": "postgres",
    "password": "Gs.20021905",
    "port": "5432"
}

try:
    conn = psycopg2.connect(**db_params)
    conn.set_client_encoding('UTF8')
    cur = conn.cursor()

    # 1. ADIM: Elindeki temizlenmiş 'df' objesini (eğer hafızadaysa) veya
    # veritabanındaki ham 'cleaned_supply_chain' tablosunu kullan.
    # Eğer hafızada df yoksa, önce veritabanındaki ana tablodan çekiyoruz:
    df_temp = pd.read_sql_query('SELECT * FROM "cleaned_supply_chain"', conn)

    # 2. ADIM: Sildiğimiz o tabloyu (final_master_supply_chain) tekrar oluştur
    table_name = "final_master_supply_chain"
    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

    columns = df_temp.columns.tolist()
    create_query = f"CREATE TABLE {table_name} ({', '.join([f'\"{c}\" TEXT' for c in columns])});"
    cur.execute(create_query)

    # Veriyi yükle
    data_values = [tuple(x) for x in df_temp.replace({np.nan: None}).to_numpy()]
    insert_query = f"INSERT INTO {table_name} ({', '.join([f'\"{c}\"' for c in columns])}) VALUES %s"

    execute_values(cur, insert_query, data_values)
    conn.commit()

    cur.close()
    conn.close()
    print(f"✅ BAŞARILI: '{table_name}' tablosu PostgreSQL'e geri geldi!")
    print("Şimdi ana kodunu en baştan gönül rahatlığıyla çalıştırabilirsin.")

except Exception as e:
    print(f"❌ Hata: {e}")