-- Önce eski hatalı view'ı silelim
DROP VIEW IF EXISTS v_risky_orders CASCADE;

-- Şimdi senin orijinal mantığınla, sütun isimlerini mühürleyerek yeniden kuralım
CREATE OR REPLACE VIEW v_risky_orders AS
SELECT 
    "order_id"::INTEGER, 
    "customer_full_name", 
    "segment", 
    "order_profit_per_order" as profit,
    "delivery_gap"::INTEGER, 
    "order_status",
    CASE 
        WHEN "order_status" = 'SUSPECTED_FRAUD' THEN 'Priority 1: Fraud'
        WHEN "order_profit_per_order" < -200 THEN 'Priority 2: Heavy Loss'
        WHEN "segment" = 'A' AND "is_late" = 1 THEN 'Priority 3: VIP Delay'
        WHEN "is_late" = 1 THEN 'Priority 4: General Delay'
        WHEN "delivery_gap" > 3 THEN 'Priority 5: High Gap'
        ELSE 'Priority 6: Other Risks'
    END as risk_priority
FROM supply_chain_analytics_master
WHERE 
    "order_profit_per_order" < -50 
    OR "is_late" = 1 
    OR "order_status" = 'SUSPECTED_FRAUD' 
    OR "delivery_gap" > 2;