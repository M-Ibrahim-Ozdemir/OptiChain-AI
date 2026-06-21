"""Aksiyon Zorluğu: Sadece 'Normal' demek yerine, hatanın tam olarak nerede olduğunu (Zaman hatası mı? Finansal hata mı?) söylemeli ki Power BI'da bunları kategorize edebilelim.
"""

"""Veri Güvenliği: Validation View (Hata Yakalayıcı)
Veride saçma sapan bir durum var mı? Mesela kâr marjı %100'den büyük mü? Veya satış miktarı 0'dan küçük mü? Bu tarz hataları yakalayan bir Sağlık Kontrolü görünümü kuralım."""

DROP VIEW IF EXISTS v_data_quality_alerts;

CREATE VIEW v_data_quality_alerts AS
SELECT 
    "order_id", 
    "customer_id", 
    "sales", 
    "order_profit_per_order",
    "days_for_shipping_real",
    "days_for_shipment_scheduled",
    CASE 
        -- FINANSAL MANTIK HATALARI
        WHEN "sales" <= 0 THEN 'KRITIK: Sales Zero or Negative'
        WHEN "order_profit_per_order" > "sales" THEN 'MANTIKSIZ: Profit Higher Than Sales'
        WHEN "order_profit_per_order" < -1000 THEN 'UYARI: Extreme Loss (>1000$)'
        
        -- LOJISTIK MANTIK HATALARI
        WHEN "days_for_shipping_real" < 0 THEN 'LOJISTIK: Negative Delivery Days'
        WHEN "days_for_shipping_real" > 30 THEN 'LOJISTIK: Impossible Delivery Days (>30)'
        
        -- VERI TAMLIGI
        WHEN "customer_id" IS NULL OR "order_id" IS NULL THEN 'SISTEM: Missing ID Record'
        
        -- ZAMAN MANTIK HATALARI
        WHEN "days_for_shipping_real" > ("days_for_shipment_scheduled" + 10) THEN 'ZAMAN: Severe Delay Gap'
        
        ELSE 'Healthy Data'
    END as health_status,
    
    -- Hata Kategorisi (Power BI Gorsellestirmesi Icin)
    CASE 
        WHEN "sales" <= 0 OR "order_profit_per_order" > "sales" THEN 'Financial Error'
        WHEN "days_for_shipping_real" < 0 OR "days_for_shipping_real" > 30 THEN 'Logistics Error'
        WHEN "customer_id" IS NULL THEN 'System Error'
        WHEN "days_for_shipping_real" > ("days_for_shipment_scheduled" + 10) THEN 'Timing Error'
        ELSE 'None'
    END as error_category
FROM supply_chain_analytics_master;



"""Aksiyon Odaklı: Power BI'da bir grafik yapıp Hangi kategoride daha çok hata var? diyebiliriz. Mesela Zamanlama Hatası çoksa, Python'daki temizleme koduna geri dönüp shipping_date sütununa bakman gerektiğini anlarsın.
Kâr/Satış Dengesi: Bir ürünün kârı, satış fiyatından büyük olamaz. Eğer böyle bir kayıt varsa ya veri girişi hatalıdır ya da kur farkı hesaplaması patlamıştır. Bunu yakalamak çok kritiktir.
Lojistik Gerçekçilik: Gerçek dünyada bir ürün -5 günde teslim edilemez veya bir smart watch'un kargosu 60 gün sürmez (genelde). Bu tarz uç değerleri (outliers) yakalayarak ML modelinin (XGBoost) bu saçma verilerden öğrenmesini engelleriz.
"""

"""
Power BI'da Veri Doğrulama Masası adlı bir sayfa ayırıp buradaki error_category sütununu kullanarak sadece hatası olan satırları bir tabloda listeleyip. Raporu kullanan yöneticiye:
Sistemdeki 180 bin verinin sağlık kontrolünü anlık yapıyorum. Şu an lojistik verilerinde ornek: %0.5 oranında mantık hatası var, bunları analize dahil etmiyorum."
"""