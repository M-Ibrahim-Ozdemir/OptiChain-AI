CREATE OR REPLACE VIEW v_customer_360_analysis AS
WITH CustomerSummary AS (
    SELECT -- İlk CTE: Müşteri bazlı temel toplamlar
        customer_id,
        customer_full_name,
        COUNT(order_id) as total_orders,
        SUM(sales) as lifetime_sales,
        AVG(order_profit_per_order) as avg_profit_per_order
    FROM supply_chain_analytics_master
    GROUP BY customer_id, customer_full_name
),
RankedProducts AS (
    SELECT   -- İkinci CTE: Her müşterinin en çok para harcadığı kategoriyi bulalım (Window Function ile)
        customer_id, 
        category_name,
        ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY SUM(sales) DESC) as rank
    FROM supply_chain_analytics_master
    GROUP BY customer_id, category_name
)
SELECT   -- Final Select: CTE'leri birleştirip anlamlı bir analitik tablo çıkarıyoruz
    cs.*, 
    rp.category_name as top_category
FROM CustomerSummary cs
JOIN RankedProducts rp ON cs.customer_id = rp.customer_id
WHERE rp.rank = 1;

-- Test etmek için:
SELECT * FROM v_customer_360_analysis LIMIT 50;


--tek seferde müşterinin genel harcamasını ve favori kategorisini aynı satırda görürüz.


