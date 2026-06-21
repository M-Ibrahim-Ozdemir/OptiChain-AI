import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Veriyi Oku
df_ = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
df = df_.copy()

# --- TÜM SÜTUNLARI GÖRME AYARI ---

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100000)

# Şimdi ilk 50 satıra bak
print(df.head(50))


# Sadece sütun isimlerinin listesini istersen:
print("\n--- TÜM SÜTUN İSİMLERİ ---")
print(df.columns.tolist())

# 3. Sütun İsimleri ve Veri Tipleri (Röntgen)
print("\n--- SÜTUNLAR VE TİPLERİ ---")
print(df.info())

# 4. Sayısal Verilerin Özeti (Ortalama satış, min/max fiyat vb.)
print("\n--- İSTATİSTİKSEL ÖZET ---")
print(df.describe([0.05,0.1,0.5,0.75,0.90,0.95,0.99]).T)

# 5. Eksik Veri Var mı? (Temizlik gerek mi?)
print("\n--- EKSİK VERİ SAYISI ---")
print(df.isnull().sum().sort_values(ascending=False).head(10))

missing_values = df.isnull().sum()
total_missing_values = (missing_values).sum()
total_cells = np.prod(df.shape)
percent_missing = (total_missing_values/total_cells) * 100
print("Percent of data that is missing :", percent_missing)
print(missing_values)


duplicated = df.duplicated().sum()
print(duplicated)
