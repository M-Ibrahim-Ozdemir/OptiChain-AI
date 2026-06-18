import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')       # Tüm uyarıları bastırır (genelde geçici kullanım önerilir)
warnings.filterwarnings(
    "ignore",
    message="A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy"
)                                       # Belirli uyarı mesajını hedefli olarak gizler
# metrics
from sklearn.metrics import mean_squared_error, r2_score ,explained_variance_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100000)


# 1. Veriyi Oku
df_ = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
df = df_.copy()

# 2. İSİM BİRLEŞTİRME (Veride tam isim yoktu, bu önemli)
df['Customer Lname'] = df['Customer Lname'].fillna('Unknown')
df['Customer_Full_Name'] = df['Customer Fname'] + " " + df['Customer Lname']

# 3. GEREKSİZ SÜTUNLARI ATMA (Açıklama dosyasındaki boş ve gizli veriler)
cols_to_drop = [
    'Product Description', 'Order Zipcode', 'Customer Password',
    'Product Image', 'Product Status', 'Customer Email',
    'Customer Fname', 'Customer Lname', 'Customer Street'
]
df.drop(columns=cols_to_drop, inplace=True)

# 4. TARİH DÖNÜŞÜMLERİ
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])




# =================================================================
# 5. YENİ DEĞİŞKENLER ÜRETME (FEATURE ENGINEERING) - ANALİSTİN İMZASI
# =================================================================

#    --- A) LOJİSTİK ANALİZ ---
# Müşteriye söz verilen gün ile gerçek gün farkı (Fark 0'dan büyükse GECİKME var)
df['Delivery_Gap'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

# Kesin Gecikme Gerçeği (1: Gecikti, 0: Zamanında/Erken)
# Not: Orijinal 'Late_delivery_risk' bir tahmindir, bu ise kesin GERÇEKTİR.
df['Is_Late'] = (df['Delivery_Gap'] > 0).astype(int)

#     ---B) (SEZONSALLIK) ANALİZİ ---
# Zaman Analizi: Ham tarihten bu parçaları çıkarmak Power BI yükünü azaltır.
df['Order_Month'] = df['order date (DateOrders)'].dt.month    # Hangi ayda sipariş verildi? (Sezonsallık analizi için)
df['Order_Day_of_Week'] = df['order date (DateOrders)'].dt.day_name()   # Hangi gün verildi? (Hafta sonu yoğunluğu için)
df['Order_Hour'] = df['order date (DateOrders)'].dt.hour  # D) Sipariş Saati (Order_Hour): Günün hangi saatinde sipariş patlaması yaşanıyor?

#(Hafta sonu operasyonları genelde daha yavaş olur, bunu ayırmak sektörel bir kuraldır)-> Grafiklerde '0-1' yerine 'Weekend/Weekday' yazması çok daha profesyonel durur.
df['Order_Day_Type'] = df['order date (DateOrders)'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

#    ---C) Coğrafi Farklılık: Müşteri ülkesi ile Sipariş ülkesi aynı mı? (Lojistik maliyet analizi)---
# (Farklıysa gümrük ve uzak yol riski artar)
df['Is_International'] = (df['Customer Country'] != df['Order Country']).astype(int)

#    --- D) FİNANSAL VERİMLİLİK , FİNANSAL VE RİSK ANALİZİ  ---
#  -> Veride 'Profit Ratio' olsa da, kendi formülümüzle (Kar / Satış)
# hesapladığımız 'Profit_Margin' her zaman daha güvenilirdir.
df['Profit_Margin'] = df['Order Profit Per Order'] / (df['Sales'] + 0.001)

# Kar Oranı Kontrolü: Veride 'Order Item Profit Ratio' var ama
# Ürün başına kar (Siparişin büyüklüğünden bağımsız verimlilik)
# bazen manuel sağlamasını yapmak (Unit Profit) iyidir.
df['Unit_Profit'] = df['Order Profit Per Order'] / df['Order Item Quantity']

# =================================================================
print("--- YENİ DEĞİŞKENLER BAŞARIYLA EKLENDİ ---")

# 6. EKSİK VERİ YÖNETİMİ
df['Customer Zipcode'] = df['Customer Zipcode'].fillna(0)

print("--- ANALİZ ÖNCESİ SON KONTROL ---")
print(f"Toplam Satır: {df.shape[0]}, Toplam Sütun: {df.shape[1]}")
print("\nÜretilen Yeni Zekalar:")
print(df[['Delivery_Gap', 'Is_Late','Order_Month', 'Order_Day_Type','Is_International', 'Unit_Profit']].head())

print("\n--- SÜTUNLAR VE TİPLERİ ---")
print(df.info())


# =================================================================
# 1. AYKIRI DEĞER VE DEĞİŞKEN ANALİZİ FONKSİYONLARI (VBO STANDARTLARI)
# =================================================================

def outlier_threshold(data, col_name, w1=0.05, w2=0.95):
    q1 = data[col_name].quantile(w1)
    q3 = data[col_name].quantile(w2)
    IQR = q3 - q1
    up = q3 + 1.5 * IQR
    low = q1 - 1.5 * IQR
    return up, low

def check_outlier(data, col_name, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    return data[(data[col_name] < low) | (data[col_name] > up)][col_name].any(axis=None)

def replace_with_thresholds(data, col_name, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    data.loc[(data[col_name] > up), col_name] = up
    data.loc[(data[col_name] < low), col_name] = low

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # Kategorik sütunlar
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # Sayısal ama aslında kategorik olanlar (Örn: Late_delivery_risk gibi 0-1 değerler)
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    # Kategorik ama çok fazla eşsiz değerli olanlar (Örn: Müşteri İsimleri)
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = [col for col in (cat_cols + num_but_cat) if col not in cat_but_car]

    # Gerçek Sayısal Sütunlar
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

# =================================================================
# 2. UYGULAMA (SENİN VERİN İÇİN EN DOĞRU VE EKSİKSİZ SIRALAMA)
# =================================================================

# Güvenlik için kopya alıyoruz
df_final = df.copy()

# A) DEĞİŞKENLERİ PROFESYONELCE GRUPLAYALIM
cat_cols, num_cols, cat_but_car = grab_col_names(df_final, cat_th=10, car_th=20)

# B) KRİTİK KORUMA: Tarihleri ve ID'leri 'Sayısal Outlier Analizi' listesinden çıkarıyoruz
# Bunlar df_final içinde kalacak ama matematiksel baskılama yapılmayacak.
exclude = [col for col in num_cols if "Id" in col or "date" in col or "Cardprod" in col]
num_cols = [col for col in num_cols if col not in exclude]

print(f"--- ANALİZ EDİLECEK GERÇEK SAYISAL SÜTUNLAR ---\n{num_cols}")

# C) AYKIRI DEĞER BASKILAMA (CAPPING)
# Sadece Sales, Profit, Delivery_Gap gibi analizde sapma yaratacak sütunları düzeltiyoruz.
for col in num_cols:
    if check_outlier(df_final, col):
        print(f"[!] {col} sütununda aykırı değer bulundu. Sınır değerlerle baskılanıyor...")
        replace_with_thresholds(df_final, col)

# D) ML (MAKİNE ÖĞRENMESİ) İÇİN BINARY SÜTUNLARI TESPİT EDELİM
# 0-1 şeklinde olan ve ileride modelde 'target' veya 'feature' olacak sütunlar
binary_cols = [col for col in df_final.columns if df_final[col].nunique() == 2 and df_final[col].dtypes != 'O']

# E) SON KONTROL RAPORU
print("\n--- KONTROL PANELİ ---")
print(f"Final Sütun Sayısı: {len(df_final.columns)}")
print(f"Tarih Sütunu Durumu: {'order date (DateOrders)' in df_final.columns} (Hala burada)")
print(f"ML Hazır Binary Sütunlar: {binary_cols}")
print(f"Aykırı Değer Kontrolü Sonrası 'Sales' Max Değeri: {df_final['Sales'].max()}")

# F) VERİYİ SQL İÇİN KAYDETME (Opsiyonel)
# df_final.to_csv('DataCo_Ready_For_SQL.csv', index=False)


print(df_final.head(50))



# 1. Tarih Hatası: Kargo tarihi siparişten önce olan satırları uçuruyoruz.
date_logic_error = df_final[df_final['shipping date (DateOrders)'] < df_final['order date (DateOrders)']].index
df_final.drop(date_logic_error, inplace=True)

# 3. Kargo Günü Hatası: Negatif kargo günü olanları (varsa) temizliyoruz.
df_final = df_final[df_final['Days for shipping (real)'] >= 0]

print(f"--- MANTIK TEMİZLİĞİ TAMAM ---")
print(f"Kalan Satır Sayısı: {len(df_final)}")



# --- ADIM 1.2: GELİŞMİŞ İŞ MANTIĞI TEMİZLİĞİ ---
# 1. İptal ve Dolandırıcılık durumlarında Kârı 0'lıyoruz.
# SHIPPING_CANCELED: Kargo iptal olduysa satış gerçekleşmemiştir.
# SUSPECTED_FRAUD: Şüpheli satışlar finansal tabloları kirletmemeli.
invalid_status = ['CANCELED', 'SUSPECTED_FRAUD', 'SHIPPING_CANCELED']
df_final.loc[df_final['Order Status'].isin(invalid_status), 'Order Profit Per Order'] = 0

# 2. İade Mantığı Kontrolü:
# Eğer kar (Benefit) aşırı negatifse  ama statü COMPLETE ise
# bu bir iade olabilir. Onları 'REFUNDED' olarak etiketlemek profesyonelliktir.
# --- İYİLEŞTİRİLMİŞ İADE VE ZARAR MANTIĞI ---
# 1. Önce kâr sütunundaki aykırı (çok aşırı) zarar eşiğini bulalım
up, low = outlier_threshold(df_final, 'Order Profit Per Order')

# 2. Eğer durum COMPLETE ise ama zarar bizim 'istatistiksel sınırımızdan' bile kötüyse
# Bu kesinlikle normal bir satış olamaz, bir terslik vardır.
refund_condition = (df_final['Order Profit Per Order'] < low) & (df_final['Order Status'] == 'COMPLETE')

# Bunları 'POTENTIAL_ISSUE' (Potansiyel Sorun) olarak işaretleyelim ki Power BI'da görelim
df_final.loc[refund_condition, 'Order Status'] = 'POTENTIAL_ISSUE'

print(f"İncelemeye alınan şüpheli işlem sayısı: {len(df_final[df_final['Order Status'] == 'POTENTIAL_ISSUE'])}")

#Veri biliminde buna "Data Integrity" (Veri Bütünlüğü) denir.




# 3. Miktar ve Fiyat Çelişkisi:
# Adet (Quantity) > 0 olup Ürün Fiyatı 0 olan saçma kayıtlar var mı?
price_error = df_final[(df_final['Order Item Quantity'] > 0) & (df_final['Order Item Product Price'] <= 0)].index
df_final.drop(price_error, inplace=True)

# 4. Sahte Satışlar: Sales 0 ise ama kar varsa bu bir veri giriş hatasıdır.
sales_error = df_final[(df_final['Sales'] <= 0) & (df_final['Order Profit Per Order'] != 0)].index
df_final.drop(sales_error, inplace=True)

print("--- GELİŞMİŞ MANTIK TEMİZLİĞİ TAMAMLANDI ---")
print(f"Toplam Silinen Hatalı Satır: {len(price_error) + len(sales_error)}")
print(f"Kârı Sıfırlanan Şüpheli İşlem Sayısı: {len(df_final[df_final['Order Status'].isin(invalid_status)])}")

"""3. Sektörel Mülakat Cevabı)
"Bu projede veriyi nasıl temizledin?" dediklerinde:
"Sadece teknik aykırı değerlere (outliers) bakmadım. İş mantığı (Business Logic) kurallarını da uyguladım. İptal edilen ve dolandırıcılık şüphesi olan 7 binden fazla işlemin kârını sıfırlayarak finansal raporların sapmasını engelledim.
Ayrıca istatistiksel sınırların dışında kalan 322 şüpheli işlemi 'Potansiyel Sorun' olarak etiketleyip analiz dışı bıraktım."
 "kod yazan biri" olmaktan çıkarıp "işi anlayan bir analistim"..."""


df_final.head(50)


## ***************************** DEĞİŞKENLER ANALİZİ(EDA)***********

# --- 1. FONKSİYONLARIN HAZIRLANMASI ---

def check_df_summary(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

def cat_summary_all(dataframe, categorical_cols, plot=False):
    for col in categorical_cols:
        print(f"--- {col.upper()} ANALİZİ ---")
        summary = pd.DataFrame({col: dataframe[col].value_counts(),
                                "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)})
        print(summary)
        print("------------------------------------------")
        if plot:
            plt.figure(figsize=(10, 5))
            # hue=col ekleyerek uyarıyı geçiyoruz, legend=False ile kalabalığı önlüyoruz
            sns.countplot(x=col, data=dataframe, hue=col, palette="viridis", legend=False)
            plt.xticks(rotation=45)
            plt.title(f"{col} Dağılımı")
            plt.show()
            plt.close()


def num_summary_all(dataframe, numerical_cols, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    for col in numerical_cols:
        print(f"--- {col.upper()} İSTATİSTİKLERİ ---")
        print(dataframe[col].describe(quantiles).T)
        if plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            dataframe[col].hist(bins=30, color='teal', edgecolor='black')
            plt.title(f"{col} Histogram")

            plt.subplot(1, 2, 2)
            sns.boxplot(x=dataframe[col], color='orange')
            plt.title(f"{col} Boxplot")

            plt.show()
            plt.close()  # <--- İŞTE BU SATIR UYARIYI SİLECEK VE RAM'İ BOŞALTACAK

def target_vs_num_analysis(dataframe, target, numerical_cols):
    print(f"\n======= {target.upper()} BAZLI SAYISAL ANALİZ =======")
    for col in numerical_cols:
        print(f"{col} ortalaması ({target} bazında):")
        print(dataframe.groupby(target).agg({col: "mean"}).sort_values(by=col, ascending=False))
        print("-" * 30)

def target_vs_cat_analysis(dataframe, target, categorical_cols):
    print(f"\n======= {target.upper()} BAZLI KATEGORİK ANALİZ =======")
    for col in categorical_cols:
        if dataframe[col].nunique() < 20: # Analiz edilebilir eşsiz değer sayısı
            print(f"{col} bazında {target} oranları:")
            print(pd.crosstab(dataframe[col], dataframe[target], normalize='index'))
            print("-" * 30)

# --- 2. UYGULAMA ---

# Önce değişken listelerimizi alalım (Daha önce tanımladığımız grab_col_names ile)
cat_cols, num_cols, cat_but_car = grab_col_names(df_final)

# A) Tüm Sayısal Dağılımları Görelim
num_summary_all(df_final, num_cols, plot=True)

# B) Tüm Kategorik Dağılımları Görelim
cat_summary_all(df_final, cat_cols, plot=True)

# C) HEDEF 1: Gecikme (Is_Late) Analizi
# Gecikme yaşayan siparişlerin hangi sayısal özelliklere sahip olduğunu anlayalım
target_vs_num_analysis(df_final, "Is_Late", num_cols)
# Hangi bölgeler veya kargo tipleri daha çok gecikiyor?
target_vs_cat_analysis(df_final, "Is_Late", cat_cols)


# E) KORELASYON MATRİSİ (Büyük Matris)

#diğer bir hedef değişken için
# Kârlılıkla en çok ilişkili (korele) olan ilk 10 değişken
print("--- Kârlılık (Order Profit Per Order) Korelasyon Analizi ---")
print(df_final[num_cols].corr()['Order Profit Per Order'].sort_values(ascending=False))


plt.figure(figsize=(15, 10))
sns.heatmap(df_final[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Supply Chain - Tüm Sayısal Değişkenlerin Korelasyonu")
plt.show()


"""Sayısal Yorumlar
İşte SQL'e geçmeden önce bilmen gereken Kritik Bulgular:
1. Kârlılık Analizi (Benefit per order & Order Profit Per Order)
Durum: Ortalama kâr yaklaşık 22$, ancak standart sapma 92$ (Çok yüksek!).
Yorum: Veride çok ciddi bir dengesizlik var. Min değer -546$ iken Max 539$. Bu şu demek: Bazı siparişler şirketi resmen "batırıyor".
İnceleme: 5% ile 95% aralığına bakarsak (-139$ ile +132$), yaptığımız Capping (Baskılama) işleminin ne kadar hayat kurtarıcı olduğunu görüyoruz. Eğer baskılamasaydık, o -4000'lik hatalar bu ortalamayı darmadağın ederdi.

2. Satış ve Ciro Dağılımı (Sales & Sales per customer)
Durum: Ortalama satış 202$, medyan (50%) ise 199$.
Yorum: Bu harika bir haber! Satış dağılımın birbirine çok yakın (Normal dağılıma meyilli). Bu, Ankara operasyonunda "fiyat istikrarı" olduğunu gösterir. Sales Max değerinin 924$ olması da verinin artık "temiz" ve analiz edilebilir olduğunu kanıtlıyor.

3. Zamanlama ve Gecikme Verileri (Order_Hour, Order_Month)
Durum: Siparişler günün her saatine (0-23) ve yılın her ayına (1-12) yayılmış.
Yorum: Mevsimsellik analizi yapmaya çok uygun. Özellikle Order_Hour ortalamasının 11.48 olması, öğle saatlerinde sipariş trafiğinin Ankara deposunda tavan yaptığını fısıldıyor.

4. Gereksiz / Ölü Sütunlar (SQL'e Taşırken Atacaklarımız) 🗑️
İstatistiklerde gördüğün şu sütunlar "Sayısal" görünse de aslında Analitik Değeri Olmayan sütunlardır:
Customer Id / Order Customer Id: Sadece bir numara. Ortalamasını almanın bir mantığı yok.
Customer Zipcode: Posta kodu sayısal bir büyüklük değildir (Etikettir).
Order Item Id / Order Id: Bunlar sadece kimlik.
Latitude / Longitude: Koordinatların ortalamasını almak bizi dünyanın ortasına götürür, analizde bir işe yaramaz.

5. Gizli Cevher: Profit_Margin ve Unit_Profit
Durum: Profit_Margin ortalaması 0.10 (%10).
Yorum: Şirket genel olarak %10 kâr marjıyla çalışıyor. Ancak Max değerin 0.50 (%50) olması, bazı ürünlerin çok kârlı olduğunu gösteriyor. Power BI'da "En Kârlı %10'luk Dilim" analizi yapmamız şart oldu!
"""


"""Kategorik
2. Verideki "Sessiz" Hata: Is_International Analizi 🔍
Attığın çıktıda çok önemli bir şey yakaladım:
--- IS_INTERNATIONAL ANALİZİ --- Ratio: 100.000
Sorun: Bu sütunun içindeki tüm değerler 1. Yani verideki tüm siparişler "Uluslararası" olarak işaretlenmiş.
Neden Önemli? Veri biliminde bir sütun hep aynı değerden oluşuyorsa (Varyansı 0 ise), o sütun modele veya analize hiçbir bilgi katmaz. Bilgisayara "Herkes insan" demek gibi bir şeydir; ayırt edici özelliği yoktur.
Çözüm: Bu sütunu SQL'e taşımamıza veya analizde tutmamıza gerek yok. Yer kaplamasın, sileceğiz.
"""
"""    --devam
3. İlk Çıkarımlar (Ankara Projesi İçin İş Zekası)
Uyarıların ötesinde, bu rakamlar bize şunları fısıldıyor:
Gecikme Felaketi (Delivery Status): %54.8 oranında "Late Delivery" var! Bu muazzam bir oran. Ankara projesinde odaklanacağımız yer burası: "Neden her 2 kargodan biri gecikiyor?"
Müşteri Lokasyonu: Müşterilerin %61'i ABD, %38'i Porto Riko. Analizde bu iki bölgeyi kıyaslamak çok mantıklı olacak.
Haftalık Denge: Siparişler haftanın günlerine (Order_Day_of_Week) çok eşit dağılmış (%14 civarı). Yani Ankara deposunda her gün aynı yoğunluk var diyebiliriz.
Kargo Modu: İnsanların %60'ı "Standard Class" seçiyor. Acaba gecikmeler en çok bu grupta mı?
"""

"""
Is_International sütunu gibi, Is_Late ve Late_delivery_risk sütunları da neredeyse aynı şeyi söylüyor (%54 civarı).
bu kadar çok gecikme olan bir sistemde; "Standard Class" kargo modunu kullananlar mı daha çok gecikiyor yoksa "First Class" (Hızlı kargo) sözü verilip de geciktirilenler mi? Bu sorunun cevabını bulmak için "Hedef Bazlı Kategorik Analiz" (target_vs_cat_analysis) O tablo her şeyi itiraf edecek! 🛠️📈
"""

"""
Sayısal Verilerdeki "Gizli Detaylar" (İnceleme Sonucu)
Uyarıyı geçip attığın rakamlara bakınca, Ankara projesi için iki tane bomba bilgi yakaladım:
Profit_Margin (Kâr Marjı): * Ortalama %10 ama bazı yerlerde -2.43 (Yani %243 zarar!).
Analist Notu: Bir ürünün satışından %243 zarar etmek normal değildir. Bu büyük ihtimalle "İade maliyeti + Kargo maliyeti + Ürün kaybı" birleşimidir. SQL'de "Zarar Tablosu" kurmamız şart.
Unit_Profit (Birim Kâr):
Standart sapma 70$. Ortalamanın (16$) çok üstünde.
Yorum: Ürün bazlı kârlılık çok istikrarsız. Bu da bizi mülakatta şu cümleyi kurmaya iter: "Ankara operasyonunda birim maliyetleri standardize etmemiz gerekiyor."
Order Item Discount Rate:
Maksimum indirim oranı 0.25 (%25).
Sektörel Check: Veride %25'ten fazla indirim yapılmamış. Eğer %50-70 indirimler görseydik "Hata var" derdik. Veri bu konuda tutarlı."""



#hedef - sauısal ilişki analizi
"""1. Kârlılık ve Gecikme İlişkisi (Zarar Kapıda!) 📉
Tespit: Zamanında giden kargolarda (Is_Late=0) ortalama kâr 22.48$ iken, gecikenlerde bu rakam 21.67$'a düşüyor.
İş Zekası Yorumu: Gecikme sadece müşteri kaybettirmiyor, para da kaybettiriyor. Operasyonel verimsizlik kâr marjını (Profit_Margin) doğrudan aşağı çekmiş. "Geç giden kargo, pahalı kargodur."

2. "Zaman Makinesi" Etkisi (Order_Hour) ⏰
Tespit: Zamanında giden siparişlerin saat ortalaması 11.08, gecikenlerin ise 11.78.
Yorum: Çok ince ama kritik bir fark! Öğleden sonra (12:00'ye doğru) verilen siparişlerin gecikme ihtimali daha yüksek. Ankara deposundaki personelin öğle yemeği saati veya vardiye değişimi sevkiyatı yavaşlatıyor olabilir mi? İşte bu, "Derinlemesine Analiz" konusudur.

3. Satış Miktarı Fark Etmiyor! (İlginç Bulgusu) 📊
Tespit: Sales ortalaması her iki grupta da neredeyse aynı (202$).
Yorum: "Büyük siparişler gecikir, küçükler hızlı gider" efsanesi vernde çürüdü. Gecikme, paketin büyüklüğünden veya fiyatından bağımsız olarak sistemde kronikleşmiş bir sorun.

4. Coğrafi ve Kimlik Verileri (Gereksizler Netleşti) 🗑️
Tespit: Latitude, Longitude, Customer Id ve Category Id ortalamaları iki grupta da (0 ve 1) birbirine tıpatıp benziyor.
Yorum: Bu şu demek; gecikme sorunu belirli bir lokasyona veya belirli bir ürün kategorisine (Category Id) yıkılamaz. Sorun tüm sisteme yayılmış. SQL'e geçerken bu sütunları "ilişki kurmak" için tutacağız ama "gecikmeyi tahmin etmek" için modellerde kullanmayacağız.

5. Order Item Profit Ratio (Verimlilik Sızıntısı) 💧
Tespit: Kâr oranı zamanında gidenlerde 0.123, gecikenlerde 0.119.
Yorum: Geciken her bir ürün birim bazında verimliliği düşürüyor."""

#Muhammet, Buradaki "Analist" Sırrı Nedir?
#Sayısal veriler bize şunu söyledi: "Operasyonel süreçlerin saatleri ve kâr marjları gecikmeden etkileniyor ama asıl sorumlu sayısal büyüklükler (fiyat, miktar) değil."
#Bu durumda suçlu kim? Suçlu büyük ihtimalle Kategorik Değişkenlerde saklı. Yani;
#Hangi kargo şirketiyle gönderdik?
#Hangi bölgeye gönderdik?
#Hangi ödeme tipiyle alındı?

#hedef - kategorik
"""Gemini şunu dedi:
Muhammet, bu tablo projenin "kara kutusunu" açtı. Kategorik analiz sonuçları, sayısal analizden çok daha çarpıcı gerçekleri ortaya koyuyor. Ankara projesi için mülakatlarda sunabileceğin, operasyonu iyileştirecek "katil bulguları" tek tek analiz ediyorum:
1. En Büyük Skandal: Shipping Mode (Kargo Modu) 🚨
İşte verinin en can alıcı noktası burası:
First Class (Hızlı Kargo): Gecikme oranı %100! (Şaka değil, tabloda 1.000 görünüyor).
Second Class: Gecikme oranı %79.7.
Standard Class: Gecikme oranı sadece %39.8.
Analiz: Bu tam bir operasyonel felaket! Müşteri daha fazla para ödeyip "hızlı gelsin" (First Class) dedikçe, sistem o kargoyu daha çok geciktirmiş. * Öneri: Ankara şubesinde "Hızlı Kargo" hattı tamamen çökmüş durumda. Önceliklendirme algoritması yanlış çalışıyor.

2. "Zaman Yolculuğu" Hatası mı? (Shipping canceled) 🧐
Delivery Status "Shipping canceled" (Kargo iptal) olanların %57'si aslında gecikmiş (Is_Late=1) görünüyor.
Yorum: İnsanlar kargoları çok geciktiği için mi iptal ediyorlar? (Yüksek ihtimalle evet). Bu, şirketin "Gecikme kaynaklı müşteri kaybı" (Churn) yaşadığının kanıtıdır.
3. Hedef Değişkenlerin Benzerliği (Late_delivery_risk) 🔄
Late_delivery_risk = 1 olanların gecikme oranı %100.
Late_delivery_risk = 0 olanların gecikme oranı %5.4.
Yorum: Şirketin kendi içindeki "Risk Tahmin Sistemi" (Late_delivery_risk) oldukça başarılı çalışıyor. Ama bu bizim için bir tehlikedir. Eğer biz modelimize hem bu riski hem de gecikmeyi koyarsak, model "kopya" çekmiş olur. İleride ML yaparken bu riski veriden çıkarmalıyız ki modelimiz gerçekten tahmin yapmayı öğrensin.

4. İstikrarın Olduğu Yerler (Neler Etkilemiyor?) ⚖️
Ödeme Tipi (Type): Nakit (CASH), Kart (PAYMENT) veya Transfer olması gecikmeyi neredeyse hiç etkilemiyor (Hepsi %56-57 bandında).
Günler (Order_Day_of_Week): Pazartesi de olsa Pazar da olsa gecikme oranı hep %57.
Yorum: Gecikme sorunu "günlük yoğunluk" veya "ödeme onayı" kaynaklı değil; tamamen lojistik süreç kaynaklı.

5. Departman Bazlı Risk (Pet Shop) 🐶
Pet Shop departmanında gecikme oranı %61.4 ile en yüksek seviyede.
Yorum: Hassas veya hızlı tüketim ürünleri (belki mamalar?) daha çok gecikiyor. Bu departmanın sevkiyat süreci acilen incelenmeli."""

#Bu analiz bize şunu öğretti: Sayısal veriler (para, miktar) masum; asıl suçlu kargo yönetim biçimi (Shipping Mode).

#*******************------------------*************************




# Soru: Hangi bölgelerde kâr daha yüksek? Hangi kargo modunda zarar ediyoruz?
def target_vs_cat_profit(dataframe, target, categorical_cols):
    for col in categorical_cols:
        if dataframe[col].nunique() < 20: # Sadece anlamlı kategoriler
            print(f"--- {col} Bazında Ortalama Kâr ---")
            print(dataframe.groupby(col)[target].mean().sort_values(ascending=False))
            print("-" * 30)

target_vs_cat_profit(df_final, "Order Profit Per Order", cat_cols)

#kategorik değişkenlerin karagore ortalamalaır
"""1. Departman Bazlı Şampiyon: Technology 💻
Bulgu: Technology departmanının ortalama kârı 102.6$ iken, Book Shop sadece 2.11$ kâr getirmiş.
Analiz: Arada neredeyse 50 kat fark var! Şirketin ana kâr motoru teknoloji ürünleri. Ankara şubesi operasyonel kaynağını (en iyi kuryeleri, en güvenli rafları) bu departmana ayırmalı.

(((2. "POTENTIAL_ISSUE" Felaketi Teşhis Edildi 📉
Bulgu: Order Status bazında baktığımızda, bizim etiketlediğimiz POTENTIAL_ISSUE grubunun ortalaması -545.95$.
Yorum: Az önce yaptığımız temizliğin ne kadar haklı olduğunu burada görüyoruz. Bu satırlar kârlılığı aşağı çeken devasa bir kara delik. SQL'de bunları mutlaka ayrı bir "Risk Analiz" tablosunda incelemeliyiz.
)))

3. Ödeme Tipi ve Kârlılık (CASH vs. TRANSFER) 💳
Bulgu: Nakit (CASH) işlemlerde kâr 24.48$ iken, Transfer işlemlerinde 19.46$'a düşüyor.
Yorum: Transfer işlemlerinde muhtemelen banka komisyonları veya operasyonel onay süreçleri kârı eritiyor. Şirket "Nakit" ödemeyi teşvik eden kampanyalar yapabilir.

4. Gecikmenin Maliyeti (Is_Late) ⏱️
Bulgu: Zamanında giden kargo (Is_Late=0) 22.48$ kazandırırken, geciken kargo 21.67$ kazandırıyor.
Yorum: Geciken her kargoda yaklaşık 1 dolar kâr kaybı var. 180 bin satırda bu, 100 binlerce dolarlık bir operasyonel sızıntı demek!

5. Garip Bir Durum: Miktar ve Kâr İlişkisi 📦
Bulgu: 5 adet ürün satıldığında kâr 29.97$ iken, 2 adet satıldığında kâr 11.66$'a kadar düşüyor.
Yorum: Normalde "çok alana az kâr marjı" (iskonto) uygulanır ama burada miktar arttıkça birim kârın korunduğunu görüyoruz. Bu, lojistik maliyetlerin toplu gönderimde azaldığını kanıtlıyor.
"""

#Customer Country: EE.UU ve Puerto Rico kârı neredeyse aynı (22$ civarı).
#Order_Day_of_Week: Pazartesi ile Pazar arasında devasa bir fark yok. Yani hafta sonu mesaisi kârı etkilemiyor.
#Customer Segment: Consumer, Corporate veya Home Office olması kâr marjını değiştirmiyor.


#KORELASYON
"""1. "Kopya Değişken" Alarmı (Çok Yüksek Korelasyon) 🚨
Tabloda 0.80'in üzerinde olan değerler, aslında aynı şeyi söyleyen değişkenlerdir:
Benefit per order (0.976): Order Profit Per Order ile neredeyse aynı. SQL'e taşırken ikisinden birini seçmeliyiz. İkisi birden veritabanında yük olur.
Unit_Profit (0.867), Profit_Margin (0.839), Order Item Profit Ratio (0.836): Bunların hepsi kârlılıkla çok sıkı fıkı. Analiz yaparken kârı (Profit) hedef almamız, diğerlerini "destekleyici" olarak kullanmamız yeterli.

2. Şaşırtıcı Gerçek: Satış vs. Kâr 📊
Sales (0.170): Satış tutarı ile kâr arasındaki bağ çok zayıf.
Yorum: Normalde satış arttıkça kârın da ciddi artması beklenir. Ancak burada 0.17 gibi düşük bir bağ olması, şirketin yüksek cirolu satışlarda ya çok büyük indirim yaptığını ya da lojistik maliyetlerin kârı yediğini gösterir. "Çok satış yapmak = Çok kâr etmek" değilmiş!

3. "Tamamen Gereksiz" Sütunlar (Temizlik Listesi) 🗑️
Korelasyonu 0.05'in altında olan (sıfıra çok yakın) şu değişkenlerin kârlılık üzerinde hiçbir matematiksel etkisi yok:
Zaman: Order_Hour (0.000), Order_Month (0.014).
Lokasyon: Latitude (-0.000), Longitude (-0.002), Customer Zipcode (0.001).
Kimlik: Order Id, Customer Id, Order Item Id.
Analist Notu: Bu sütunları SQL'de "İlişkisel Veritabanı" kurmak için (JOIN işlemleri için) tutacağız ama kâr tahmini yaparken (ML) kesinlikle modelin içine koymayacağız. Çünkü kârı etkilemiyorlar.
"""


# Veriyi CSV olarak kaydet
df_final.to_csv('DataCo_Cleaned_Final.csv', index=False, encoding='utf-8')
print("Temizlenmiş veri 'DataCo_Cleaned_Final.csv' adıyla kaydedildi!")


df = pd.read_csv('DataCo_Cleaned_Final.csv', encoding='utf-8')
df.head(50)
df.shape[1]

