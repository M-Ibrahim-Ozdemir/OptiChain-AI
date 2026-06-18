import pandas as pd
import psycopg2
from io import StringIO

# 1. Veriyi Oku
print("Veri dosyası okunuyor, lütfen bekleyin...")
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# --- KRİTİK ADIM: Karakter Temizliği ---
print("Özel karakterler temizleniyor (ASCII uyumu için)...")
# Verideki tüm metinleri ASCII formatına zorluyoruz, tanımadıklarını siliyoruz
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')

# 2. SQL Bağlantı Parametreleri
db_params = {
    "host": "localhost",
    "database": "supply_chain_db",
    "user": "postgres",
    "password": "",
    "port": "5432"
}

# 3. Veriyi SQL'e Aktar
print("Lojistik Verisi SQL'e aktarılıyor... (Temiz Mod)")
try:
    conn = psycopg2.connect(**db_params)
    conn.set_client_encoding('SQL_ASCII')
    cur = conn.cursor()

    # Tabloyu oluştur
    cur.execute("DROP TABLE IF EXISTS raw_data;")
    columns = ", ".join([f'"{col}" TEXT' for col in df.columns])
    cur.execute(f"CREATE TABLE raw_data ({columns});")

    # Bellek üzerinden hızlı transfer
    output = StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)

    cur.copy_from(output, 'raw_data', sep='\t', null="")

    conn.commit()
    cur.close()
    conn.close()

    print("\n" + "=" * 50)
    print("Veri 'raw_data' tablosuna tertemiz aktarıldı.")
    print("=" * 50)

except Exception as e:
    print(f"Hala bir direnç var: {e}")