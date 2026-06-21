import pandas as pd
import psycopg2
from io import StringIO
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100000)

# 1. Temizlenmiş veriyi oku
print("Temizlenmiş veri okunuyor...")
df = pd.read_csv('DataCo_Cleaned_Final.csv')

# 2. Sütun isimlerini SQL uyumlu 
df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for col in df.columns]

# 3. SQL Bağlantı Bilgileri
import os

db_params = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "supply_chain_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),  
    "port": os.getenv("DB_PORT", "5432")
}

print(f"Veri '{db_params['database']}' veritabanına UTF-8 formatında akıtılıyor...")
start_time = time.time()

try:
    # Bağlantıyı kur
    conn = psycopg2.connect(**db_params)
    # --- KRİTİK NOKTA 1: Bağlantı kodlamasını ayarla ---
    conn.set_client_encoding('UTF8')
    cur = conn.cursor()

    # --- KRİTİK NOKTA 2: Bellek tamponunu UTF-8 olarak oluştur ---
    output = StringIO()
    # index=False, veriyi sekmeyle ayırıyoruz (tab-separated)
    df.to_csv(output, sep='\t', header=False, index=False, encoding='utf-8')
    output.seek(0)

    # Tabloyu sıfırla ve yeniden oluştur
    cur.execute("DROP TABLE IF EXISTS cleaned_supply_chain;")

    # Sütunları oluştur (ID'ler ve Sayılar dahil hepsi başlangıçta TEXT)
    columns = ", ".join([f'"{col}" TEXT' for col in df.columns])
    cur.execute(f"CREATE TABLE cleaned_supply_chain ({columns});")

    # ASIL HIZLI KISIM: copy_from
    cur.copy_from(output, 'cleaned_supply_chain', sep='\t', null="")

    conn.commit()
    cur.close()
    conn.close()

    end_time = time.time()
    print("\n" + "=" * 50)
    print(f"KESİN BAŞARI! Veri UTF-8 olarak SQL'e aktarıldı.")
    print(f"Süre: {round(end_time - start_time, 2)} saniye.")
    print("=" * 50)

except Exception as e:
    print(f"Bir hata oluştu: {e}")




print(df.head(50))

# 3. Sütun İsimleri ve Veri Tipleri (Röntgen)
print("\n--- SÜTUNLAR VE TİPLERİ ---")
print(df.info())
