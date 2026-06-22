<p align="center">
  <img src="logo.png" alt="OptiChain AI Logo" width="200"/>
</p>

<h1 align="center">🚀 OptiChain AI v2.0</h1>
<h3 align="center">Kurumsal Uçtan Uca Veri Analitiği - Tahmine Dayalı gecikme riski ve Akıllı Satış ve Operasyon Planlaması Karar Ekosistemi</h3>

<p align="center">
  <a href="https://optichain-ai-dxnp6cdz3hkeicoudpz4cx.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Application-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live App"/>
  </a>
  <img src="https://img.shields.io/badge/Database-PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
</p>

---

<h2 align="center">📈 1. EXECUTIVE SUMMARY & BUSINESS VALUE PROPOSITION</h2>

<p align="center" style="font-size: 18px; line-height: 1.6; max-width: 850px; margin: 0 auto;">
  <b>🌐 Sektörel Problem Tanımı ve Projenin Vizyonu</b>
  <br><br>
  Modern tedarik zinciri, CRM ve lojistik ağlarında karşılaşılan en büyük operasyonel darboğaz, 
  <b>tahmin edilemeyen sevkiyat gecikmeleri</b>, <b>verimsiz stok-talep dengesizlikleri</b> ve 
  veritabanı seviyesinde kontrol edilemeyen <b>kirli veri akışlarıdır</b>. 
  Zamanında teslim edilemeyen tek bir sipariş müşteri sadakatini (<b>Churn Risk</b>) doğrudan tetiklerken; 
  ambarlarda biriken kontrolsüz atıl stoklar kurumsal elde tutma maliyetlerini (<b>Holding Cost</b>) maksimize etmekte, 
  veri tabanındaki lojistik ve finansal mantık hataları ise makine öğrenmesi modellerinin yanlış öğrenmesine 
  (<b>Garbage In, Garbage Out</b>) yol açmaktadır.
  <br><br>
  <b>OptiChain AI v2.0</b>; ham lojistik, finans, CRM ve operasyonel veri setlerini <b>uçtan uca</b> işleyerek 
  operasyon yöneticilerinin ve C-Level karar vericilerin günlük stratejik kararlarını optimize eden, 
  yapay zeka tabanlı küresel bir <b>S&OP (Satış ve Operasyon Planlama) Karar Destek Sistemidir</b>. 
  Bu ekosistem; Python ile çok aşamalı veri temizliğinden ileri düzey müşteri analitiğine (<b>RFM, K-Means, CLTV</b>), 
  PostgreSQL mimarisindeki kurumsal veri kalitesi kalkanlarından (<b>Data Quality Validation Views</b>) 
  açıklanabilir yapay zeka (<b>XAI</b>) ve derin öğrenme simülasyonlarına kadar geniş bir yelpazeyi 
  iş zekası (<b>Power BI</b>) vizyonuyla birleştirmektedir.
</p>

<br>

<h3 align="center">🛠️ Stratejik Mühendislik Katmanları ve Çözüm Kümesi</h3>

<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
  🎯 <b>Çok Aşamalı Python Veri Temizliği & Tanıma:</b> Ham veri setleri Python (Pandas/NumPy) mimarisiyle gürültüden arındırılmış, veri tipleri optimize edilmiş ve lojistik iş birimlerinin ihtiyaç duyduğu makro metrikler türetilmiştir.
  <br><br>
  💎 <b>Gelişmiş CRM & Müşteri Analitiği (RFM, K-Means & CLTV):</b> Müşteri tabanı sadece kuramsal varsayımlarla değil; Python kullanılarak RFM analitiği, K-Means Kümeleme algoritması ve BG/NBD & Gamma-Gamma modelleriyle ileri düzey CLTV (Müşteri Yaşam Boyu Değeri) tahminleme süreçlerine tabi tutulmuştur. Şirketin en kârlı VIP segmentleri matematiksel olarak izole edilmiştir.
  <br><br>
  📊 <b>İstatistiksel Doğrulama (A/B Testing):</b> Dağıtım kanalları, teslimat rotaları ve segment bazlı performans kırılımları parametrik ve non-parametrik hipotez testleriyle (A/B Testleri) matematiksel olarak doğrulanmıştır.
  <br><br>
  🛡️ <b>İleri Düzey SQL Veri Kalitesi & Sağlık Kontrolü (Data Governance):</b> PostgreSQL seviyesinde kurulan <code>v_data_quality_alerts</code> görünümüyle 180 bin satırlık kurumsal verinin sağlık kontrolü anlık olarak yapılmaktadır. Finansal, lojistik ve zamanlama mantık hataları sistem tarafından otomatik olarak yakalanıp kategorize edilerek Power BI raporlama katmanına hazır hale getirilmiştir.
  <br><br>
  🚨 <b>Proaktif Risk Önceliklendirme & Operasyonel Görünüm (SQL Views):</b> Yazılan <code>v_risky_orders</code> view katmanı ile şüpheli işlemler (Fraud), aşırı finansal kayıplar ve VIP gecikme krizleri hiyerarşik olarak risk puanlarına (Priority 1-6) ayrılmış, karar vericilere anlık aksiyon alma gücü sunulmuştur.
  <br><br>
  🧠 <b>Açıklanabilir Makine Öğrenmesi (CatBoost & SHAP):</b> Sevkiyatlar henüz ambar kapısından çıkmadan önce gecikme olasılığını hesaplayan bir sınıflandırma modeli entegre edilmiştir. SHAP (XAI) katmanıyla, her bir risk faktörünün karara etkisi operasyonel bazda şeffaflaştırılmıştır.
  <br><br>
  📈 <b>Proaktif Derin Öğrenme Simülasyonları (LSTM):</b> 30 günlük dinamik ciro dalgalanmalarını öngören zaman serisi modeli kurgulanmıştır. Sistem, kampanya şoklarını simüle ederek ambar stoklarının kritik seviyeye düşeceği günü önceden öngörüp proaktif "Tedarik Emirleri" üretmektedir.
  <br><br>
  💻 <b>Kurumsal Dağıtım & İş Zekası Katmanı (Power BI, FastAPI & Streamlit):</b> Canlı sistem mimarisi FastAPI backend sunucusuyla izole edilmiş, C-Level yöneticilerin senaryo analizi yapabileceği premium bir Streamlit frontend paneli ve kurumsal Power BI raporlama katmanıyla mühürlenmiştir.
</p>

---
