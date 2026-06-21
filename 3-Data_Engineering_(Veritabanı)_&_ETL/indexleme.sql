-- 1. TEMEL BAĞLANTI İNDEKSLERİ
CREATE INDEX IF NOT EXISTS idx_master_order_id ON supply_chain_analytics_master(order_id);
CREATE INDEX IF NOT EXISTS idx_master_customer_id ON supply_chain_analytics_master(customer_id);
CREATE INDEX IF NOT EXISTS idx_master_product_id ON supply_chain_analytics_master(product_card_id);

-- 2. LOJİSTİK PERFORMANS İNDEKSLERİ
CREATE INDEX IF NOT EXISTS idx_master_is_late ON supply_chain_analytics_master(is_late);
CREATE INDEX IF NOT EXISTS idx_master_delivery_status ON supply_chain_analytics_master(delivery_status);
CREATE INDEX IF NOT EXISTS idx_master_shipping_mode ON supply_chain_analytics_master(shipping_mode);

-- 3. ZAMAN VE TAKVİM İNDEKSLERİ
CREATE INDEX IF NOT EXISTS idx_master_order_date ON supply_chain_analytics_master(order_date_dateorders);
CREATE INDEX IF NOT EXISTS idx_master_order_month ON supply_chain_analytics_master(order_month);
CREATE INDEX IF NOT EXISTS idx_master_order_hour ON supply_chain_analytics_master(order_hour);

-- 4. CRM VE STRATEJİK İNDEKSLER
CREATE INDEX IF NOT EXISTS idx_master_loyalty_segment ON supply_chain_analytics_master(segment);
CREATE INDEX IF NOT EXISTS idx_master_cltv_prediction ON supply_chain_analytics_master(cltv_prediction);

-- 5. COĞRAFİ VE FİNANSAL İNDEKSLER
CREATE INDEX IF NOT EXISTS idx_master_order_region ON supply_chain_analytics_master(order_region);
CREATE INDEX IF NOT EXISTS idx_master_is_international ON supply_chain_analytics_master(is_international);
CREATE INDEX IF NOT EXISTS idx_master_sales ON supply_chain_analytics_master(sales);

---------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_m_order_id ON supply_chain_analytics_master(order_id);
CREATE INDEX IF NOT EXISTS idx_m_cust_id ON supply_chain_analytics_master(customer_id);
CREATE INDEX IF NOT EXISTS idx_m_late_flag ON supply_chain_analytics_master(is_late);
CREATE INDEX IF NOT EXISTS idx_m_order_date ON supply_chain_analytics_master(order_date_dateorders);
CREATE INDEX IF NOT EXISTS idx_m_order_final ON supply_chain_analytics_master(order_id);



-- 6. VERİTABANI MOTORUNU GÜNCELLE
VACUUM ANALYZE supply_chain_analytics_master;