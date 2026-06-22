<p align="center">
  <img src="logo.png" alt="OptiChain AI Logo" width="180"/>
</p>

<h1 align="center">🚀 OptiChain AI v2.0</h1>
<h3 align="center">Kurumsal Uçtan Uca Veri Analitiği, Tahmine Dayalı Risk Teşhisi ve Akıllı S&OP Karar Destek Ekosistemi</h3>

<p align="center">
  <a href="https://optichain-ai-dxnp6cdz3hkeicoudpz4cx.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Canlı_Uygulama-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live App"/>
  </a>
  <img src="https://img.shields.io/badge/Veritabanı-PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Backend_Mimari-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/İş_Zekası-Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black" alt="Power BI"/>
</p>

<p align="center">
  <a href="1-Data_Analytics_&_Business_Intelligence/supply_chain_powerBI_compressed (1).pdf" target="_blank">
    <img src="https://img.shields.io/badge/📄_Yönetici_Raporu-Power_BI_PDF_İndir-0078D4?style=for-the-badge" alt="Power BI Executive PDF"/>
  </a>
</p>

---

<h2 align="center">📈 1. ÖZET VE İŞ DEĞERİ ÖNERİSİ</h2>

<p align="center" style="font-size: 16.5px; line-height: 1.7; max-width: 850px; margin: 0 auto; text-align: justify;">
  <b>🌐 Sektörel Problem Tanımı ve Projenin Vizyonu:</b><br>
  Modern tedarik zinciri, CRM ve lojistik ağlarında karşılaşılan en büyük operasyonel darboğaz, <b>tahmin edilemeyen sevkiyat gecikmeleri</b>, <b>verimsiz stok-talep dengesizlikleri</b> ve veritabanı seviyesinde kontrol edilemeyen <b>kirli veri akışlarıdır</b>. Zamanında teslim edilemeyen tek bir sipariş müşteri sadakatini (<b>Churn Risk</b>) doğrudan tetiklerken; ambarlarda biriken kontrolsüz atıl stoklar kurumsal elde tutma maliyetlerini (<b>Holding Cost</b>) maksimize etmekte, veri tabanındaki lojistik ve finansal mantık hataları ise makine öğrenmesi modellerinin yanlış öğrenmesine (<b>Garbage In, Garbage Out</b>) yol açmaktadır.
  <br><br>
  <b>OptiChain AI v2.0;</b> ham tedarik zinciri ve lojistik veri setini <b>uçtan uca</b> işleyerek detaylı analizler, stratejik tespitler ve operasyon yöneticileri ile C-Level karar vericilerin günlük kararlarını optimize eden, yapay zeka tabanlı küresel bir <b>S&OP (Satış ve Operasyon Planlama) Karar Destek Sistemidir</b>. Bu ekosistem; Python ile çok aşamalı veri temizliğinden, ileri düzey makro keşifçi veri analizlerine (EDA), parametrik/non-parametrik <b>A/B Testlerine</b>, müşteri analitiği (<b>RFM, K-Means Kümeleme, CLTV, BG/NBD & Gamma-Gamma</b>) modellerine, <b>100+'den fazla gelişmiş DAX fonksiyonu</b> barındıran kurumsal <b>Power BI</b> finansal raporlama katmanına, PostgreSQL mimarisindeki veri yönetişimi kalkanlarına (<b>Data Quality Validation & Risk Views</b>), <b>CatBoost & SHAP</b> tabanlı açıklanabilir makine öğrenmesi (<b>XAI</b>) çıkarım hatlarına ve derin öğrenme (<b>LSTM</b>) satış simülasyonlarına kadar uzanan devasa bir yelpazeyi tek bir çatıda birleştirmektedir.
</p>

<br>

<h2 align="center">🛠️ 2. STRATEJİK KATMANLAR VE ÇÖZÜM KÜMESİ</h2>

<div style="max-width: 850px; margin: 0 auto; font-size: 15px; line-height: 1.6; text-align: left;">
  <p>🎯 <b>Çok Aşamalı Python Veri Temizliği & Tanıma:</b> Ham veri setleri Python (Pandas/NumPy) mimarisiyle detaylıca gürültüden arındırılmış, veri tipleri optimize edilmiş ve lojistik iş birimlerinin ihtiyaç duyduğu yeni türetilmiş değişkenler ve makro metrikler boru hattına eklenmiştir.</p>
  
  <p>💎 <b>Gelişmiş CRM & Müşteri Analitiği (RFM, K-Means & CLTV):</b> Müşteri tabanı kuramsal varsayımlarla değil; Python ile RFM analitiği, K-Means algoritması ve BG/NBD & Gamma-Gamma modelleriyle ileri düzey CLTV (Müşteri Yaşam Boyu Değeri) tahminleme süreçlerine tabi tutulmuştur. En kârlı VIP müşteri segmentleri matematiksel olarak izole edilmiştir.</p>
  
  <p>📊 <b>İstatistiksel Doğrulama (A/B Testing):</b> Dağıtım kanalları, teslimat rotaları ve segment bazlı performans kırılımları rastgele kabullerle değil, parametrik ve non-parametrik hipotez testleriyle (A/B Testleri) matematiksel olarak doğrulanmıştır.</p>
  
  <p>🛡️ <b>İleri Düzey SQL Veri Kalitesi & Sağlık Kontrolü (Data Governance):</b> PostgreSQL seviyesinde kurulan <code>v_data_quality_alerts</code> görünümüyle 180 bin satırlık kurumsal verinin sağlık kontrolü anlık olarak yapılmaktadır. Finansal, lojistik ve zamanlama mantık hataları sistem tarafından otomatik olarak yakalanıp kategorize edilerek Power BI segment analizlerine hazır hale getirilmiştir.</p>
  
  <p>🚨 <b>Proaktif Risk Önceliklendirme & Operasyonel Görünüm (SQL Views):</b> Yazılan <code>v_risky_orders</code> view katmanı ile şüpheli işlemler (Fraud), aşırı finansal kayıplar ve VIP gecikme krizleri hiyerarşik olarak risk puanlarına (Priority 1-6) ayrılmış, karar vericilere anlık aksiyon alma gücü sunulmuştur.</p>
  
  <p>🧠 <b>Açıklanabilir Makine Öğrenmesi (CatBoost & SHAP):</b> Sevkiyatlar henüz ambar kapısından çıkmadan önce gecikme olasılığını hesaplayan bir sınıflandırma modeli entegre edilmiştir. SHAP (XAI) katmanıyla, her bir risk faktörünün karara etkisi operasyonel bazda tamamen şeffaflaştırılmıştır.</p>
  
  <p>📈 <b>Proaktif Derin Öğrenme Simülasyonları (LSTM):</b> Girilen tarihten 30 gün sonrası için günlük dinamik ciro dalgalanmalarını öngören zaman serisi modeli kurgulanmıştır. Sistem, kampanya şoklarını simüle ederek ambar stoklarının kritik seviyeye düşeceği günü önceden öngörüp proaktif "Tedarik Emirleri" üretmektedir.</p>
  
  <p>💻 <b>Kurumsal Dağıtım & İş Zekası Katmanı (Power BI, FastAPI & Streamlit):</b> Canlı sistem mimarisi FastAPI backend sunucusuyla izole edilmiş, C-Level yöneticilerin senaryo analizi yapabileceği premium bir Streamlit frontend paneli ve kurumsal Power BI raporlama katmanıyla mühürlenmiştir.</p>
</div>

---
