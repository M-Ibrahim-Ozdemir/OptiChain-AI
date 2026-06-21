DROP VIEW IF EXISTS 

    v_customer_360_analysis, v_customer_performance_tracker, v_data_quality_alerts,

    v_dim_customers, v_dim_location, v_dim_products, v_dim_time, 

    v_fact_sales, v_risky_orders CASCADE;



-- ==========================================================================

-- 2. ADIM: TÜM SAYISAL SÜTUNLARI MÜHÜRLE (62 SÜTUNLUK YAPIYA UYGUN)

-- ==========================================================================

ALTER TABLE supply_chain_analytics_master 

    -- Kimlik ve Kategorik Tam Sayılar

    ALTER COLUMN "order_id" TYPE INTEGER USING "order_id"::NUMERIC::INTEGER,

    ALTER COLUMN "customer_id" TYPE INTEGER USING "customer_id"::NUMERIC::INTEGER,

    ALTER COLUMN "product_card_id" TYPE INTEGER USING "product_card_id"::NUMERIC::INTEGER,

    ALTER COLUMN "order_item_id" TYPE INTEGER USING "order_item_id"::NUMERIC::INTEGER,

    ALTER COLUMN "category_id" TYPE INTEGER USING "category_id"::NUMERIC::INTEGER,

    ALTER COLUMN "customer_zipcode" TYPE INTEGER USING "customer_zipcode"::NUMERIC::INTEGER,

    ALTER COLUMN "department_id" TYPE INTEGER USING "department_id"::NUMERIC::INTEGER,

    ALTER COLUMN "order_customer_id" TYPE INTEGER USING "order_customer_id"::NUMERIC::INTEGER,

    ALTER COLUMN "order_item_cardprod_id" TYPE INTEGER USING "order_item_cardprod_id"::NUMERIC::INTEGER,

    ALTER COLUMN "order_item_quantity" TYPE INTEGER USING "order_item_quantity"::NUMERIC::INTEGER,

    ALTER COLUMN "product_category_id" TYPE INTEGER USING "product_category_id"::NUMERIC::INTEGER,

    

    -- Operasyonel Sayılar

    ALTER COLUMN "is_late" TYPE INTEGER USING "is_late"::NUMERIC::INTEGER,

    ALTER COLUMN "late_delivery_risk" TYPE INTEGER USING "late_delivery_risk"::NUMERIC::INTEGER,

    ALTER COLUMN "is_international" TYPE INTEGER USING "is_international"::NUMERIC::INTEGER,

    ALTER COLUMN "delivery_gap" TYPE INTEGER USING "delivery_gap"::NUMERIC::INTEGER,

    ALTER COLUMN "order_month" TYPE INTEGER USING "order_month"::NUMERIC::INTEGER,

    ALTER COLUMN "order_hour" TYPE INTEGER USING "order_hour"::NUMERIC::INTEGER,

    ALTER COLUMN "days_for_shipping_real" TYPE INTEGER USING "days_for_shipping_real"::NUMERIC::INTEGER,

    ALTER COLUMN "days_for_shipment_scheduled" TYPE INTEGER USING "days_for_shipment_scheduled"::NUMERIC::INTEGER,

    

    -- Finansal Metrikler (Eksiksiz)

    ALTER COLUMN "sales" TYPE DOUBLE PRECISION USING "sales"::NUMERIC,

    ALTER COLUMN "order_item_total" TYPE DOUBLE PRECISION USING "order_item_total"::NUMERIC,

    ALTER COLUMN "order_profit_per_order" TYPE DOUBLE PRECISION USING "order_profit_per_order"::NUMERIC,

    ALTER COLUMN "product_price" TYPE DOUBLE PRECISION USING "product_price"::NUMERIC,

    ALTER COLUMN "profit_margin" TYPE DOUBLE PRECISION USING "profit_margin"::NUMERIC,

    ALTER COLUMN "unit_profit" TYPE DOUBLE PRECISION USING "unit_profit"::NUMERIC,

    ALTER COLUMN "benefit_per_order" TYPE DOUBLE PRECISION USING "benefit_per_order"::NUMERIC,

    ALTER COLUMN "order_item_discount" TYPE DOUBLE PRECISION USING "order_item_discount"::NUMERIC,

    ALTER COLUMN "order_item_discount_rate" TYPE DOUBLE PRECISION USING "order_item_discount_rate"::NUMERIC,

    ALTER COLUMN "order_item_product_price" TYPE DOUBLE PRECISION USING "order_item_product_price"::NUMERIC,

    ALTER COLUMN "order_item_profit_ratio" TYPE DOUBLE PRECISION USING "order_item_profit_ratio"::NUMERIC,

    ALTER COLUMN "sales_per_customer" TYPE DOUBLE PRECISION USING "sales_per_customer"::NUMERIC,

    ALTER COLUMN "latitude" TYPE DOUBLE PRECISION USING "latitude"::NUMERIC,

    ALTER COLUMN "longitude" TYPE DOUBLE PRECISION USING "longitude"::NUMERIC,

    

    -- CRM, RFM ve Tahminleme Metrikleri

    ALTER COLUMN "recency" TYPE DOUBLE PRECISION USING "recency"::NUMERIC,

    ALTER COLUMN "frequency" TYPE DOUBLE PRECISION USING "frequency"::NUMERIC,

    ALTER COLUMN "monetary" TYPE DOUBLE PRECISION USING "monetary"::NUMERIC,

    ALTER COLUMN "T" TYPE DOUBLE PRECISION USING "T"::NUMERIC,

    ALTER COLUMN "expected_purc_3_month" TYPE DOUBLE PRECISION USING "expected_purc_3_month"::NUMERIC,

    ALTER COLUMN "expected_average_profit" TYPE DOUBLE PRECISION USING "expected_average_profit"::NUMERIC,

    ALTER COLUMN "cltv_prediction" TYPE DOUBLE PRECISION USING "cltv_prediction"::NUMERIC;



-- ==========================================================================

-- 3. ADIM: EKSİKSİZ STAR SCHEMA VIEW YAPISI

-- ==========================================================================



-- E. Satış & Finans Fact (Tüm sayısal veriler burada mühürlendi)

CREATE OR REPLACE VIEW v_fact_sales AS

SELECT

    "order_id", "customer_id", "product_card_id", "order_item_id",

    "order_date_dateorders" as sale_date, 

    "type" as payment_type,

    "sales", 

    "order_item_total", 

    "order_profit_per_order" as profit, 

    "benefit_per_order",

    "profit_margin", 

    "unit_profit",

    "order_item_quantity", 

    "order_item_product_price", 

    "order_item_discount", 

    "order_item_discount_rate",

    "days_for_shipping_real", 

    "days_for_shipment_scheduled", 

    "delivery_gap",

    "is_late", 

    "late_delivery_risk", 

    "order_status", 

    "shipping_mode", 

    "delivery_status",
	
	"sales_per_customer"

FROM supply_chain_analytics_master;



-- A. Müşteri Zekası

CREATE OR REPLACE VIEW v_dim_customers AS

SELECT DISTINCT ON ("customer_id")

    "customer_id", "customer_full_name", "customer_segment" as original_segment,

    "segment" as calculated_loyalty_segment, "customer_city", "customer_country",

    "recency", "frequency", "monetary", "T", "cltv_prediction",

    "expected_purc_3_month", "expected_average_profit"

FROM supply_chain_analytics_master
ORDER BY "customer_id", "order_date_dateorders" DESC;



-- B. Ürün Boyutu

CREATE OR REPLACE VIEW v_dim_products AS

SELECT DISTINCT

    "product_card_id", "product_name", "category_name", "department_name",

    "product_price", "category_id", "product_category_id"

FROM supply_chain_analytics_master;



-- C. Zaman Zekası

CREATE OR REPLACE VIEW v_dim_time AS

SELECT DISTINCT

    "order_date_dateorders" as full_timestamp, "order_month",

    "order_day_of_week", "order_day_type", "order_hour"

FROM supply_chain_analytics_master;



-- D. Lokasyon Boyutu

CREATE OR REPLACE VIEW v_dim_location AS
SELECT DISTINCT ON ("customer_id")
    "customer_id", "customer_city", "customer_country", "customer_state",
    "order_city", "order_country", "order_region", "market",
    "latitude", "longitude", "is_international"
FROM supply_chain_analytics_master
ORDER BY "customer_id", "order_date_dateorders" DESC;