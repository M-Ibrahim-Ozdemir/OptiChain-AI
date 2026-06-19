"""
((4 FARKLI CALISMA ELE ALDIM NİHAİ SONUCU İNCELEYİNİZ...))

1: Problem Tanımı (Analiz Konumuz):
"First Class (Hızlı Kargo) kullananlar ile Standard Class kullananların kâr marjları (profit_margin) veya gecikme oranları
arasında istatistiksel olarak anlamlı bir fark var mıdır?"
Neden bu? Şirket yönetimi "First Class daha pahalı ama gerçekten değiyor mu?"
 iye soruyor. Biz de onlara bilimsel kanıt sunacağız."""


import pandas as pd
import numpy as np
import psycopg2
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# 1. TÜM MASTER DATAYI ÇEKELİM
import os

# GİZLİLİK PROTOKOLÜ: Şifreler kodun içinde değil, sistem çevre değişkenlerinden güvenle okunur.
# 🔒 %100 KORUMALI MUTLAK GİZLİLİK MODU
db_params = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "supply_chain_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),  # 🎯 BAK BURADAKİ ŞİFREYİ TAMAMEN SİLDİK, SADECE DEĞİŞKEN KALDI!
    "port": os.getenv("DB_PORT", "5432")
}

print("Master Data yükleniyor...")
try:
    conn = psycopg2.connect(**db_params)
    conn.set_client_encoding('UTF8')
    cur = conn.cursor()

    cur.execute("SELECT * FROM final_master_supply_chain")
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df_final = pd.DataFrame(rows, columns=colnames)

    cur.close()
    conn.close()
    print(f"✅ Başarılı! {df_final.shape[0]} satır ve {df_final.shape[1]} sütun hazır.")
except Exception as e:
    print(f"❌ Hata: {e}")

# 2. VERİ TİPLERİNİ DÜZELTELİM (Analiz için şart)
# SQL'den gelen sayısal değerleri float yapıyoruz
numeric_cols = ['profit_margin', 'sales', 'unit_profit', 'days_for_shipping_real', 'late_delivery_risk']
for col in numeric_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# 3. İLK RÖNTGEN: KARGO MODLARININ KÂR MARJI ANALİZİ
print("\n" + "=" * 40)
print(" 📊 KARGO MODLARINA GÖRE KÂR MARJI ÖZETİ ")
print("=" * 40)
summary = df_final.groupby("shipping_mode")["profit_margin"].agg(["mean", "median", "std", "count"]).round(4)
print(summary)


"""Sonuçları incelededim: Röntgen Analizi:
Tabloya baktığımızda çok ilginç bir durum var:
First Class: 0.1137 kâr ortalamasıyla en tepede.
Same Day: 0.1061 ile en düşükte kalmış.
Arada yaklaşık %0.7'lik bir fark var. "Aman ne olacak %1 bile değil" denebilir ama 180 bin satırlık bir operasyonda
bu milyonlarca dolar fark demek! "Bu fark tesadüf mü, yoksa First Class gerçekten daha mı kârlı """


# --- ADIM 2: VARSAYIM KONTROLLERİ ---
print("\n" + "="*40)
print(" 🧪 VARSAYIM KONTROLLERİ ")
print("="*40)

# 1. Normallik Varsayımı (Shapiro-Wilk)
# Not: Veri setimiz çok büyük olduğu için Shapiro yerine büyük veride daha stabil olan
# histogram ve merkezi limit teoremi mantığına bakılır ama biz yine de örneklem üzerinden test edelim.
from scipy.stats import shapiro

# Her gruptan rastgele 3000 örnek alarak test edelim (Shapiro 5000 satırdan fazlasını sevmez)
for mode in df_final['shipping_mode'].unique():
    sample = df_final[df_final['shipping_mode'] == mode]['profit_margin'].sample(3000, random_state=42)
    stat, p = shapiro(sample)
    print(f"{mode} için Normallik p-value: {p:.4f}")

# 2. Varyans Homojenliği (Levene Testi)
# H0: Grupların varyansları eşittir.
modes = [df_final[df_final['shipping_mode'] == m]['profit_margin'].dropna() for m in df_final['shipping_mode'].unique()]
stat, p_levene = levene(*modes)
print(f"\nVaryans Homojenliği (Levene) p-value: {p_levene:.4f}")



"""Normallik Testi (Shapiro): Tüm gruplar için p-value 0.0000 çıktı.
Anlamı: P-value < 0.05 olduğu için H0 (Veriler normal dağılır) hipotezini reddediyoruz. Verimiz normal dağılmıyor (ki 180 bin satırlık gerçek hayat verilerinde bu beklediğimiz bir durumdur).
Varyans Homojenliği (Levene): p-value 0.1731 çıktı.
Anlamı: P-value > 0.05 olduğu için H0 (Varyanslar eşittir) hipotezini reddedemiyoruz. Yani kargo modlarındaki kâr dağılımlarının "dalgalanma boyutu" birbirine benzer."""

"""Normallik varsayımı sağlanmadığı için biz artık Parametrik (T-Test/ANOVA) testleri kullanamayız.
Eğer sadece iki grup olsaydı Mann-Whitney U yapardık. Ancak bizim elimizde 4 tane kargo modu var (Standard, First, Second, Same Day). Bu yüzden ANOVA'nın non-parametrik karşılığı olan Kruskal-Wallis testini yapacağız."""


# --- ADIM 3: HİPOTEZ TESTİNİN UYGULANMASI ---

print("\n" + "="*40)
print(" ⚖️ KRUSKAL-WALLIS TESTİ (NON-PARAMETRİK) ")
print("="*40)

from scipy.stats import kruskal

# Grupları hazırlayalım
group_standard = df_final[df_final['shipping_mode'] == 'Standard Class']['profit_margin']
group_first = df_final[df_final['shipping_mode'] == 'First Class']['profit_margin']
group_second = df_final[df_final['shipping_mode'] == 'Second Class']['profit_margin']
group_sameday = df_final[df_final['shipping_mode'] == 'Same Day']['profit_margin']

# Testi uygulayalım
stat, p_kruskal = kruskal(group_standard, group_first, group_second, group_sameday)

print(f"Kruskal-Wallis Test İstatistiği: {stat:.4f}")
print(f"P-Value Değeri: {p_kruskal:.4f}")

if p_kruskal < 0.05:
    print("\n✅ SONUÇ: Gruplar arasında istatistiksel olarak ANLAMLI bir fark vardır.")
    print("Yani kargo modu seçimi kârlılığı gerçekten etkiliyor (Tesadüf değil!).")
else:
    print("\n❌ SONUÇ: Gruplar arasında anlamlı bir fark yoktur.")
    print("Aradaki küçük farklar tamamen şansa bağlı gelişmiştir.")


""""Evet, First Class kargo ortalamada biraz daha kârlı görünüyor olabilir
ama 180 bin satırlık veriyegenel olarak baktığımızda,
bu küçük fark tamamen rastlantısal.
Kargo modunu değiştirmek kâr marjını kökten etkilemiyor."""

""""Kargo modlarının kâr marjı üzerindeki etkisini Kruskal-Wallis testi ile analiz ettim.
İlk bakışta First Class daha avantajlı görünse de p-value 0.46 çıktı.
Yani istatistiksel olarak kargo modları arasında anlamlı bir kârlılık farkı yok.
Bu da şirketin lojistik stratejisini kâr marjı yerine hız veya müşteri memnuniyeti odaklı kurması gerektiğini kanıtladı.
"""




#2.Çalışma:  Kargo modu kârlılığı etkilemiyor olabilir, peki ya gecikmeleri etkiliyor mu?. Müşteri "First Class" için para ödüyor, gerçekten daha az mı gecikiyor?
#"Debit vs Diğerleri" veya "Kargo Modu vs Gecikme" testi yapalım. Ama bu sefer değişkenimiz sürekli bir sayı (kâr marjı) değil, ikili bir durum (0-1: Gecikti mi?).
# "Proportions Z-Test" (Oran Testi).

"""0-1 verilerinde "Normal dağılım" aranmaz, çünkü veri zaten iki uçtadır.
Bu yüzden oran karşılaştırmalarında doğrudan Z-Test (Proportions Z-Test) veya Ki-Kare (Chi-Square) kullanılır.
"""



# --- ADIM 4: ORAN TESTİ (PROPORTIONS Z-TEST) ---
print("\n" + "="*40)
print(" ⚖️ KARGO MODU vs GECİKME ORANI ANALİZİ ")
print("="*40)

from statsmodels.stats.proportion import proportions_ztest

# 1. First Class ve Standard Class gruplarını hazırlayalım
# Gecikenlerin sayısı (late_delivery_risk == 1 olanlar)
fc_late_count = df_final[(df_final['shipping_mode'] == 'First Class') & (df_final['late_delivery_risk'] == 1)].shape[0]
sc_late_count = df_final[(df_final['shipping_mode'] == 'Standard Class') & (df_final['late_delivery_risk'] == 1)].shape[0]

# Toplam sayıları
fc_total = df_final[df_final['shipping_mode'] == 'First Class'].shape[0]
sc_total = df_final[df_final['shipping_mode'] == 'Standard Class'].shape[0]

# 2. Testi Uygulayalım
count = np.array([fc_late_count, sc_late_count])
nobs = np.array([fc_total, sc_total])

stat, p_ztest = proportions_ztest(count, nobs)

print(f"First Class Gecikme Oranı: %{ (fc_late_count/fc_total)*100:.2f}")
print(f"Standard Class Gecikme Oranı: %{ (sc_late_count/sc_total)*100:.2f}")
print(f"\nZ-Test P-Value: {p_ztest:.4f}")

if p_ztest < 0.05:
    print("\n✅ SONUÇ: Gecikme oranları arasında İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VAR!")
else:
    print("\n❌ SONUÇ: Gecikme oranları arasında anlamlı bir fark yok.")


""""Yaptığım Proportions Z-Test sonucunda, First Class kargo modunun gecikme oranının (%95.32),
Standard Class'tan (%38.07) istatistiksel olarak anlamlı derecede yüksek olduğunu kanıtladım
(p < 0.05). Bu durum, şirketin 'Hızlı Teslimat' vaadiyle sunduğu premium hizmetin operasyonel 
bir darboğazda olduğunu ve acilen süreç iyileştirmesi gerektiğini bilimsel olarak ortaya koymaktadır."""



#adem "hız" konusunda bir sorun bulduk,"Süre (Gün)" boyutuna bakalım.
#Bu sefer tekrar başa dönüyoruz, çünkü "Gün" verisi (1 gün, 3 gün, 5 gün) sayısal bir veridir.

#Problem 3: Teslimat Süreleri (Days for shipping real)
#"First Class gerçekten Standard Class'tan daha kısa sürede mi ulaşıyor?"


# --- ADIM 5: TESLİMAT GÜN SAYISI VARSAYIM KONTROLLERİ ---
print("\n" + "="*40)
print(" 🧪 TESLİMAT SÜRESİ (GÜN) VARSAYIMLARI ")
print("="*40)

# Veriyi hazırlayalım (Shipping Real sütunu)
df_final['days_for_shipping_real'] = pd.to_numeric(df_final['days_for_shipping_real'], errors='coerce')

# 1. Normallik (Shapiro)
for mode in ['First Class', 'Standard Class']:
    sample = df_final[df_final['shipping_mode'] == mode]['days_for_shipping_real'].sample(3000, random_state=42)
    stat, p = shapiro(sample)
    print(f"{mode} için Gün Sayısı Normallik p-value: {p:.4f}")

# 2. Varyans Homojenliği (Levene)
stat, p_levene = levene(
    df_final[df_final['shipping_mode'] == 'First Class']['days_for_shipping_real'],
    df_final[df_final['shipping_mode'] == 'Standard Class']['days_for_shipping_real']
)
print(f"Varyans Homojenliği p-value: {p_levene:.4f}")


"""first Class için Normallik p-value: 1.0000 (Range Zero Uyarısı):  "Range zero", First Class verilerinin neredeyse tamamı aynı değerden oluşuyor demek. Eğer herkes 2 günde alıyorsa, varyans sıfırdır ve Shapiro testi 1.0000 döner. Bu aslında doğal bir dağılım değil, "sabitlenmiş" bir veridir.
Standard Class için Normallik p-value: 0.0000: Standard Class normal dağılmıyor. Yani teslimat süreleri çok değişken; kimi 3 günde alıyor, kimi 6 günde.
Varyans Homojenliği p-value: 0.0000: Varyanslar eşit değil! Yani First Class çok stabil (herkes aynı sürede alıyor gibi), Standard Class ise çok dağınık."""


"""Normallik yok, varyans homojenliği yok. Bu durumda yine parametrik testlerden
(T-Test) uygulayamıcaz. Mann-Whitney U testini kullanacağız. Çünkü iki grubu (First vs Standard) karşılaştırıyoruz ve varsayımlarımız patladı"""


# --- ADIM 6: TESLİMAT SÜRESİ KARŞILAŞTIRMA (MANN-WHITNEY U) ---
print("\n" + "="*40)
print(" ⚖️ TESLİMAT HIZI TESTİ (MANN-WHITNEY U) ")
print("="*40)

from scipy.stats import mannwhitneyu

# Grupları çekelim
fc_days = df_final[df_final['shipping_mode'] == 'First Class']['days_for_shipping_real']
sc_days = df_final[df_final['shipping_mode'] == 'Standard Class']['days_for_shipping_real']

# Testi uygulayalım
stat, p_mann = mannwhitneyu(fc_days, sc_days)

print(f"First Class Ortalaması: {fc_days.mean():.2f} gün")
print(f"Standard Class Ortalaması: {sc_days.mean():.2f} gün")
print(f"\nMann-Whitney U P-Value: {p_mann:.4f}")

if p_mann < 0.05:
    print("\n✅ SONUÇ: Teslimat süreleri arasında ANLAMLI bir fark var.")
    if fc_days.mean() < sc_days.mean():
        print("İstatistiksel olarak First Class DAHA HIZLIDIR.")
    else:
        print("İstatistiksel olarak Standard Class DAHA HIZLIDIR veya fark yoktur.")
else:
    print("\n❌ SONUÇ: Teslimat süreleri arasında anlamlı bir fark yok.")


    #Nihai Sonuc
"""Verilerin bize söylediği iki zıt gerçek var:
Hız Testi: First Class (2.00 gün), Standard Class'tan (4.00 gün) istatistiksel olarak daha hızlı. (p < 0.05)
Gecikme Testi: First Class'ın gecikme oranı %95, Standard Class'ın ise %38."""

"""
(((("Sistemi analiz ettiğimde ilginç bir paradoks yakaladım. First Class kargolar ortalamada 2 gün ile "
"Standard Class'tan (4 gün) anlamlı derecede daha hızlı teslim ediliyor. Ancak ironik bir şekilde,"
" First Class siparişlerin %95'i gecikmiş olarak işaretlenmiş. Bu da bize operasyonel bir hız sorunu olmadığını, "
"aksine planlama (scheduling) hatası olduğunu gösteriyor. Şirket, First Class için muhtemelen '1 gün' gibi gerçekçi"
" olmayan bir teslimat süresi vaat ediyor. Fiziksel olarak 2 günde teslim etse bile, sistemde gecikmiş görünüyor. "
"Bu durum müşteri memnuniyetini haksız yere düşürüyor ve acilen 'Söz verilen teslimat süresi' (scheduled days) "
"parametrelerinin revize edilmesi gerekiyor.")))))"""


#4.ANALİZ
# --- ADIM 7: ÖDEME TİPİ vs GECİKME ANALİZİ (Z-TEST) ---
print("\n" + "="*40)
print(" ⚖️ ÖDEME TİPİ (DEBIT) vs GECİKME ORANI ")
print("="*40)

# 1. Grupları hazırlayalım
# Debit ödeyenler
debit_late = df_final[(df_final['type'] == 'DEBIT') & (df_final['late_delivery_risk'] == 1)].shape[0]
debit_total = df_final[df_final['type'] == 'DEBIT'].shape[0]

# Diğerleri (Debit olmayan her şey)
others_late = df_final[(df_final['type'] != 'DEBIT') & (df_final['late_delivery_risk'] == 1)].shape[0]
others_total = df_final[df_final['type'] != 'DEBIT'].shape[0]

# 2. Testi uygulayalım
count = np.array([debit_late, others_late])
nobs = np.array([debit_total, others_total])
stat, p_payment = proportions_ztest(count, nobs)

print(f"Debit Ödeme Gecikme Oranı: %{ (debit_late/debit_total)*100:.2f}")
print(f"Diğer Ödeme Türleri Gecikme Oranı: %{ (others_late/others_total)*100:.2f}")
print(f"\nP-Value Değeri: {p_payment:.4f}")

if p_payment < 0.05:
    print("\n✅ SONUÇ: Ödeme tipi ile gecikme arasında ANLAMLI bir ilişki var!")
else:
    print("\n❌ SONUÇ: Ödeme tipi gecikmeyi etkilemiyor.")



"""Bu tablo bize şunu söylüyor: Debit Kart ile ödeme yapan müşterilerin siparişleri, diğer yöntemlere göre daha fazla gecikiyor.
Finansal Onay Süreçleri: Debit kartlarda provizyon veya banka onay süreçleri, kredi kartı veya nakit ödemeye göre sistemde birkaç saat daha fazla bekliyor olabilir.
Operasyonel Sıralama: Belki de depo yönetim sistemi, "Nakit" veya "Transfer" ödemeleri (paranın direkt geçmesi sebebiyle) önceliklendiriyor, Debit ödemeleri bir sonraki toplama listesine bırakıyor."""

#SUNUM
""" Operasyonel verimliliği ölçmek adına ödeme yöntemlerinin teslimat performansına etkisini inceledim. Proportions Z-Test sonucunda, Debit Kart ödemelerinin gecikme oranının diğer yöntemlerden anlamlı derecede yüksek olduğunu (p < 0.05) tespit ettim.
Bu durum, finansal onay süreçlerinin lojistik akışı yavaşlattığına dair bir sinyal vermektedir. Şirkete, ödeme sistemleri ile depo yönetim sistemi (WMS) arasındaki entegrasyonu incelemeyi ve Debit onay sürelerini optimize etmeyi önerdim.
-4 Büyük İstatistiksel Analizi Tamamladık!
Kargo Modu vs. Kâr: Arada anlamlı bir fark yok (Kâr için kargo modu önemsiz).
Kargo Modu vs. Gecikme Skandalı: First Class %95 gecikiyor (Planlama hatası).
Hız Testi: First Class aslında hızlı ama vaatler yanlış (Süreç hatası).
Ödeme Tipi vs. Gecikme: Debit ödemeler daha riskli (Finans-Lojistik bağı)."""


