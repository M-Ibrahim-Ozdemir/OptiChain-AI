-- Eski görünümü silelim veya üzerine yazalım
CREATE OR REPLACE VIEW v_customer_performance_tracker AS
SELECT 
    "customer_id",
    "customer_full_name",
    "segment" as python_segment, -- Python'daki BG-NBD segmentin burada guvende kalir
    "sales",
    "order_profit_per_order",
    CASE 
        WHEN "sales" > 500 AND "order_profit_per_order" > 150 THEN 'Star Performer'
        WHEN "order_profit_per_order" < 0 THEN 'Urgent Intervention (Loss)'
        ELSE 'Normal Performance'
    END as instant_performance_status -- Turkce karakterler temizlendi
FROM supply_chain_analytics_master;