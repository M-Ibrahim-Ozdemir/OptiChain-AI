import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# ==========================================================================
# 💎 1. ADIM: SAYFA GLOBAL AYARLARI & OPTICHAIN AI MARKALAMA
# ==========================================================================
st.set_page_config(
    page_title="OptiChain AI - Risk & S&OP Yönetim Paneli",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 UX/UI PREMIUM RECOV: Geliştirilmiş Kurumsal CSS ve Hizalama Katmanı
st.markdown("""
    <style>
    .stApp {
        background-color: #121927;
        color: #FFFFFF;
    }
    div[data-testid="stSidebar"] div blockquote {
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stSidebarUserContent"] img {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        border-radius: 12px !important;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4) !important;
    }
    .premium-header-card {
        background: linear-gradient(135deg, #1a2333 0%, #151d2a 100%);
        border: 1px solid #2d3d5a;
        border-radius: 14px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.3);
    }
    .premium-title {
        background: linear-gradient(90deg, #f0a500 0%, #ffc837 50%, #f0a500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 38px !important;
        font-weight: 800 !important;
        margin-bottom: 12px !important;
        letter-spacing: 1px;
    }
    .premium-subtitle {
        font-size: 18px !important;
        color: #cbd5e0 !important;
        font-style: italic !important;
        font-weight: 400 !important;
        max-width: 800px;
        margin: 0 auto !important;
        line-height: 1.6 !important;
    }
    p, label, span, .stWidgetLabel p {
        font-size: 19px !important;
        font-weight: 500 !important;
        color: #FFFFFF !important;
    }
    h2, h3 {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #FFFFFF !important;
        border-bottom: 2px solid #2d3d5a;
        padding-bottom: 8px;
    }
    .stSelectbox div[data-baseweb="select"], .stNumberInput div input {
        font-size: 18px !important;
        font-weight: bold !important;
        height: 48px !important;
    }
    .stButton>button {
        background-color: #f0a500 !important;
        color: #121927 !important;
        font-size: 22px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        height: 55px !important;
        box-shadow: 0px 4px 15px rgba(240, 165, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        background-color: #ffb733 !important;
    }
    div[data-testid="stSidebarUserContent"] p, div[data-testid="stSidebarUserContent"] label {
        font-size: 17px !important;
    }
    .stAlert div div {
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] {
        background-color: #1a2333 !important;
        border: 1px solid #2d3d5a !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sol Üst Kurumsal Logo ve Başlık Yapısı
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=110)
st.sidebar.title("🤖 OptiChain AI v2.0")
st.sidebar.caption("Global Tahmine Dayalı Tedarik Zinciri Analitiği")
st.sidebar.markdown("---")

# Sayfa Seçim Menüsü
page = st.sidebar.radio("📋 Operasyonel Radarlar",
                        ["🚨 Sipariş Gecikme Riski & SHAP Analizi", "📈 LSTM Global Talep & Stok Tahmini"])

# FastAPI Sunucu Adresleri
BASE_API_URL = "https://1effb1f6a28f7be4-193-255-91-141.serveousercontent.com"
PREDICT_API_URL = f"{BASE_API_URL}/predict-risk"
LSTM_API_URL = f"{BASE_API_URL}/predict-lstm-forecast"


# ==========================================================================
# 🛰️ ÖNBELLEKLİ VERİ ENTEGRASYON KATMANI (POSTGRESQL BAĞLANTILARI)
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_geography_hierarchy_from_api():
    """FastAPI üzerinden veritabanındaki Market->Region->City bağını çeker."""
    try:
        response = requests.get(f"{BASE_API_URL}/geography-hierarchy")
        if response.status_code == 200:
            return response.json().get("hierarchy", {})
        st.error(f"Backend Hata Kodu Döndü: {response.status_code} | Detay: {response.text}")
        return {}
    except Exception as e:
        st.error(f"Bağlantı esnasında ağ/sorgu hatası oluştu: {str(e)}")
        return {}


# Coğrafi Hiyerarşiyi Veritabanından Canlı Yüklüyoruz
geo_hierarchy = load_geography_hierarchy_from_api()

if not geo_hierarchy:
    st.error(
        "❌ Backend (main.py) sunucusuna ulaşılamadı veya PostgreSQL bağlantısı kapalı! Lütfen backend'i kontrol edin.")
    st.stop()

# ==========================================================================
# 🚨 1. SAYFA: SİPARİŞ GECİKME TAHMİNİ & ENTEGRE SHAP ANALİZİ
# ==========================================================================
if page == "🚨 Sipariş Gecikme Riski & SHAP Analizi":

    st.markdown("""
        <div class="premium-header-card">
            <div class="premium-title"> 🛡 Sipariş Risk Analizi ve Canlı SHAP Açıklama Katmanı</div>
            <div class="premium-subtitle">
                Sipariş henüz yola çıkmadan önce sızıntısız operasyonel verileri girerek gecikme olasılığını ve kök sebeplerini analiz edin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("📋 Lojistik & Operasyonel Girdiler")

        market_list = list(geo_hierarchy.keys())
        market = st.selectbox("🌍 Pazar Yeri (Market)", market_list)

        region_list = list(geo_hierarchy[market].keys()) if market in geo_hierarchy else []
        order_region = st.selectbox("📍 Teslimat Bölgesi (Region)", region_list)

        city_list = geo_hierarchy[market][order_region] if order_region in geo_hierarchy[market] else []
        order_city = st.selectbox("🏙️ Hedef Şehir (City)", city_list)

        try:
            coord_res = requests.get(f"{BASE_API_URL}/city-coordinates?city={order_city}").json()
            latitude = coord_res.get("latitude", 0.0)
            longitude = coord_res.get("longitude", 0.0)
            st.sidebar.success(f"📍 {order_city} Aktif: Enlem {latitude} | Boylam {longitude}")
        except:
            latitude, longitude = 0.0, 0.0

        department_name = st.selectbox("🏬 Ürün Departmanı",
                                       ["Men's Footwear", "Cleats", "Apparel", "Technology", "Pet Shop"])
        customer_segment = st.selectbox("👥 Müşteri Segmenti", ["Home Office", "Corporate", "Consumer"])

        order_day_of_week = st.selectbox("📅 Sipariş Günü",
                                         ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        order_hour = st.slider("🕒 Sipariş Saati (0-23)", 0, 23, 14)
        order_month = st.slider("📆 Sipariş Ayı (1-12)", 1, 12, 7)

        order_item_quantity = st.number_input("📦 Sipariş Ürün Adeti", min_value=1, max_value=100, value=2)
        sales = st.number_input("💰 Toplam Satış Cirosu ($)", min_value=1.0, value=124.79)
        profit_margin = st.number_input("📈 Kar Marjı Oranı (Örn: 0.05)", min_value=-2.0, max_value=1.0, value=0.07)
        unit_profit = st.number_input("💵 Birim Başına Kar ($)", value=9.36)

        st.markdown("### 💎 CRM & Müşteri Sadakat Geçmişi")
        recency = st.number_input("⏱️ Recency (Son Siparişten Beri Geçen Gün)", value=4.0)
        frequency = st.number_input("🔄 Frequency (Toplam Sipariş Sayısı)", value=18.0)
        monetary = st.number_input("💳 Monetary (Şirkete Bıraktığı Toplam Ciro $)", value=479.07)
        cltv_prediction = st.number_input("🏆 Hesaplanan CLTV Öngörüsü", value=493.77)
        expected_purc_3_month = st.number_input("🔮 Önümüzdeki 3 Ay Beklenen Sipariş Hacmi", value=5.20)
        expected_average_profit = st.number_input("📊 Öngörülen Ortalama Kar ($)", value=9.36)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("🚀 RİSK ANALİZİNİ BAŞLAT")

    with col_result:
        st.subheader("🎯 Yapay Zeka Karar & Teşhis Paneli")

        if submit_btn:
            payload = {
                "market": market, "order_region": order_region, "order_city": order_city,
                "department_name": department_name, "customer_segment": customer_segment,
                "order_day_of_week": order_day_of_week, "order_hour": order_hour, "order_month": order_month,
                "order_item_quantity": order_item_quantity, "sales": sales, "profit_margin": profit_margin,
                "unit_profit": unit_profit, "recency": recency, "frequency": frequency,
                "monetary": monetary, "cltv_prediction": cltv_prediction, "latitude": latitude,
                "longitude": longitude, "expected_purc_3_month": expected_purc_3_month,
                "expected_average_profit": expected_average_profit
            }

            with st.spinner("⏳ FastAPI Canlı Risk Pipeline'ı koşturuluyor..."):
                try:
                    response = requests.post(PREDICT_API_URL, json=payload)
                    res_data = response.json()

                    if response.status_code == 200:
                        risk_prob = res_data["gecikme_olasiligi"] * 100
                        model_decision = res_data["model_karari"]

                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=risk_prob, domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Sevkiyat Gecikme Riski (%)",
                                   'font': {'size': 22, 'color': '#FFFFFF', 'family': 'Arial'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#FFFFFF",
                                         'tickfont': {'size': 16}},
                                'bar': {'color': "#f0a500" if model_decision == "GECİKME RİSKİ" else "#2ed573"},
                                'bgcolor': "#1a2333", 'borderwidth': 2, 'bordercolor': "#2d3d5a",
                                'steps': [
                                    {'range': [0, 40], 'color': 'rgba(46, 213, 115, 0.1)'},
                                    {'range': [40, 54.06], 'color': 'rgba(255, 165, 0, 0.1)'},
                                    {'range': [54.06, 100], 'color': 'rgba(255, 71, 87, 0.1)'}
                                ],
                                'threshold': {'line': {'color': "red", 'width': 5}, 'thickness': 0.8, 'value': 54.06}
                            }
                        ))
                        fig_gauge.update_layout(paper_bgcolor='#121927', plot_bgcolor='#121927',
                                                font={'color': "#FFFFFF"}, height=320,
                                                margin=dict(t=40, b=0, l=10, r=10))
                        fig_gauge.update_traces(number={'font': {'size': 64, 'weight': 'bold'}})
                        st.plotly_chart(fig_gauge, use_container_width=True)

                        if model_decision == "GECİKME RİSKİ":
                            st.error(
                                f"⚠️ **KRİTİK GECİKME ALARMI:** Bu siparişin gecikme olasılığı %{risk_prob:.1f} olarak hesaplanmıştır. Belirlenen cerrahi eşik (T=%54.1) aşıldığı için kargo riskli gruptadır.")
                        else:
                            st.success(
                                f"✅ **GÜVENLİ SEVKİYAT HATTI:** Siparişin zamanında teslim edilme ihtimali yüksektir (Risk: %{risk_prob:.1f}). Standart operasyonel süreç devam ettirilebilir.")

                        st.markdown("### 📜 Yapay Zeka Operasyonel Teşhis Notları")
                        col_risk, col_adv = st.columns(2)
                        with col_risk:
                            st.markdown("**🚨 Gecikmeyi Tetikleyen Kritik Riskler:**")
                            if res_data["kritik_riskler"]:
                                for item in res_data["kritik_riskler"]: st.info(
                                    f"👉 {item['faktor']} nedeniyle risk **% {item['etki']}** arttı.")
                            else:
                                st.write("Gecikmeyi tetikleyen büyük bir makro risk saptanmadı.")

                        with col_adv:
                            st.markdown("**✅ Operasyonel Avantajlar (Hızlandırıcılar):**")
                            if res_data["operasyonel_avantajlar"]:
                                for item in res_data["operasyonel_avantajlar"]: st.success(
                                    f"👉 {item['faktor']} sayesinde teslimat **% {item['etki']}** hızlandı.")
                            else:
                                st.write("Teslimatı hızlandıran ekstra bir operasyonel kaldıraç saptanmadı.")

                        st.markdown("### 📊 SHAP Model Karar Terazisi (Açıklanabilirlik)")
                        raw_shap = res_data["raw_shap_values"]
                        feat_names = res_data["feature_names"]

                        sorted_idx = np.argsort(np.abs(raw_shap))[::-1][:10]
                        plot_features = [feat_names[i] for i in sorted_idx]
                        plot_shap = [raw_shap[i] for i in sorted_idx]

                        fig_waterfall = go.Figure(go.Bar(
                            x=plot_shap, y=plot_features, orientation='h',
                            marker=dict(color=['#ff4757' if val > 0 else '#2ed573' for val in plot_shap])
                        ))
                        fig_waterfall.update_layout(
                            title="Top 10 Karar Faktörünün Olasılığa Net Etkisi (SHAP)", title_font={'size': 20},
                            paper_bgcolor='#1a2333', plot_bgcolor='#1a2333', font={'color': "#FFFFFF"},
                            xaxis=dict(tickfont=dict(size=15)), yaxis=dict(tickfont=dict(size=15)),
                            height=450, margin=dict(t=50, b=50, l=180, r=20)
                        )
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                    else:
                        st.error(f"FastAPI Hatası: {res_data.get('detail')}")
                except Exception as e:
                    st.error(f"Backend sunucusu ile bağlantı kurulamadı. main.py çalışıyor mu? Hata: {str(e)}")
        else:
            st.info(
                "ℹ️ Sol panelden operasyonel sipariş girdilerini set edip 'Risk Analizini Başlat' butonuna basarak yapay zeka kararlarını tetikleyebilirsiniz.")

# ==========================================================================
# 📈 2. SAYFA: LSTM GLOBAL TALEP & AMBAR STOK PROJEKSİYONU (KUSURSUZ ENTEGRASYON)
# ==========================================================================
elif page == "📈 LSTM Global Talep & Stok Tahmini":

    st.markdown("""
        <div class="premium-header-card">
            <div class="premium-title">📈 LSTM Küresel Proaktif Talep & Ambar Yönetim Radarı</div>
            <div class="premium-subtitle">
                Seçtiğiniz tarihten ileriye doğru 30 günlük S&OP dengesini simüle edin. Stokların eksiye düşme riskini ve kaçan ciro maliyetlerini anlık izleyin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_s_input, col_s_result = st.columns([1, 1.2], gap="large")

    with col_s_input:
        st.subheader("📦 Ambar Stok Kontrolü")
        selected_date = st.date_input("📅 Projeksiyon Başlangıç Tarihi", datetime.date(2018, 1, 1))
        initial_stock = st.number_input("🛢️ Başlangıç Depo Stok Seviyesi ($)", min_value=1000.0, value=150000.0)

        st.subheader("📢 Operasyonel Şok Senaryoları")
        campaign_shock = st.slider("📈 Tahmini Kampanya / Talep Şoku Oranı (%)", -50, 100, 0)
        holding_cost_rate = st.slider("📉 Günlük Atıl Stok Holding Cost Oranı (‰ - On Binde)", 1, 50, 5)

        st.markdown("<br>", unsafe_allow_html=True)
        trigger_lstm = st.button("GLOBAL 30 GÜNLÜK TAHMİNİ TETİKLE")

    with col_s_result:
        st.subheader("🔮 Canlı S&OP Simülasyon Çıktıları")

        if trigger_lstm:
            payload_lstm = {
                "start_date": str(selected_date),
                "campaign_shock": float(campaign_shock),
                "holding_cost_rate": float(holding_cost_rate)
            }

            with st.spinner("⏳ S&OP Bilançosu dinamik olarak hesaplanıyor..."):
                try:
                    response = requests.post(LSTM_API_URL, json=payload_lstm)
                    res_data = response.json()

                    if response.status_code == 200:
                        forecast_list = res_data["forecast_30_days"]
                        dinamik_esik = res_data["dinamik_esik_degeri"]
                        capa_degeri = res_data["secilen_tarih_capasi"]

                        df_forecast = pd.DataFrame(forecast_list)
                        df_forecast["tarih"] = pd.to_datetime(df_forecast["tarih"])

                        # 🎯 GERÇEKÇİ S&OP AMBAR AKIŞ MATEMATİĞİ (Eksi Sınır Korumalı)
                        stock_trend = []
                        lost_sales_trend = []
                        current_stock = initial_stock

                        for sale in df_forecast["tahmin_edilen_satis"]:
                            if current_stock >= sale:
                                current_stock = current_stock - sale
                                lost_sales = 0.0
                            else:
                                lost_sales = sale - current_stock
                                current_stock = 0.0

                            stock_trend.append(current_stock)
                            lost_sales_trend.append(lost_sales)

                        df_forecast["simule_kalan_stok"] = stock_trend
                        df_forecast["kacan_ciro_riski"] = lost_sales_trend

                        # S&OP Dinamik Tedarik Karar Bildirimleri
                        stok_durumu_list = []
                        tedarik_emri_list = []
                        for idx, row in df_forecast.iterrows():
                            if row["simule_kalan_stok"] == 0:
                                stok_durumu_list.append("❌ STOK TÜKENDİ")
                                tedarik_emri_list.append("🚨 ACİL TEDARİK: Üretimi Tetikle")
                            elif row["simule_kalan_stok"] < (initial_stock * 0.20):
                                stok_durumu_list.append("⚠️ KRİTİK SEVİYE")
                                tedarik_emri_list.append("📦 GÜVENLİK STOKU: Sipariş Geç")
                            else:
                                stok_durumu_list.append("✅ OPTİMAL")
                                tedarik_emri_list.append("👍 SÜREÇ STABİL: Maliyet Optimize")

                        df_forecast["final_stok_durumu"] = stok_durumu_list
                        df_forecast["final_tedarik_emri"] = tedarik_emri_list

                        # Finansal KPI Kartları Üst Panel
                        kpi_1, kpi_2, kpi_3 = st.columns(3)
                        total_predicted_demand = df_forecast["tahmin_edilen_satis"].sum()
                        total_lost_sales = df_forecast["kacan_ciro_riski"].sum()

                        kpi_1.metric("🔮 30 Günlük Toplam Talep Baskısı", f"{total_predicted_demand:,.2f} $")
                        kpi_2.metric("🚨 Toplam Kaçan Ciro Riski", f"{total_lost_sales:,.2f} $",
                                     delta="- Finansal Kayıp" if total_lost_sales > 0 else "0 $")
                        kpi_3.metric("🌍 Doğrulanan Tarihsel Çapa Verisi", f"{capa_degeri:,.2f} $")

                        # --- GRAFİK 1: CİRO RİTMİ VE GÜVEN KORİDORU ---
                        fig_line = go.Figure()
                        fig_line.add_trace(go.Scatter(
                            x=df_forecast["tarih"], y=df_forecast["tahmin_edilen_satis"],
                            mode='lines+markers', name='Dinamik LSTM Tahmini (Ciro Ritmi)',
                            line=dict(color='#00d2d3', width=3)
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=df_forecast["tarih"], y=df_forecast["ust_limit"],
                            mode='lines', name='%95 Güven Koridoru (Üst)',
                            line=dict(color='rgba(255,255,255,0.15)', dash='dash')
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=df_forecast["tarih"], y=df_forecast["alt_limit"],
                            mode='lines', name='%95 Güven Koridoru (Alt)',
                            line=dict(color='rgba(255,255,255,0.15)', dash='dash'),
                            fill='tonexty', fillcolor='rgba(0, 210, 211, 0.05)'
                        ))
                        fig_line.update_layout(
                            title="30 Günlük Küresel Ciro Dalgalanması & Volatilite Koridoru",
                            paper_bgcolor='#121927', plot_bgcolor='#1a2333', font={'color': "#FFFFFF"},
                            height=260, margin=dict(t=40, b=20, l=40, r=20), showlegend=True
                        )
                        st.plotly_chart(fig_line, use_container_width=True)

                        # --- GRAFİK 2: AMBAR STOK ERİTME EĞRİSİ ---
                        fig_stock = go.Figure()
                        fig_stock.add_trace(go.Scatter(
                            x=df_forecast["tarih"], y=df_forecast["simule_kalan_stok"],
                            mode='lines+markers', name='Fiziksel Kalan Stok',
                            line=dict(color='#ff9f43', width=4)
                        ))
                        fig_stock.update_layout(
                            title="Ambar Stok Seviyesi Azalma Eğrisi (Sıfır Sınır Korumalı)",
                            paper_bgcolor='#121927', plot_bgcolor='#1a2333', font={'color': "#FFFFFF"},
                            height=220, margin=dict(t=40, b=20, l=40, r=20),
                            yaxis=dict(range=[0, max(initial_stock * 1.1, 10000.0)])
                        )
                        st.plotly_chart(fig_stock, use_container_width=True)

                        # --- BİLANÇO VE S&OP KARAR DESTEK TABLOSU ---
                        st.markdown("### 📋 Küresel S&OP Karar Destek ve Bilanço Çizelgesi")
                        df_display = df_forecast.copy()
                        df_display["Tarih"] = df_display["tarih"].dt.strftime("%Y-%m-%d")
                        df_display["Talep Tahmini ($)"] = df_display["tahmin_edilen_satis"].map("{:,.2f}".format)
                        df_display["Simüle Kalan Stok ($)"] = df_display["simule_kalan_stok"].map("{:,.2f}".format)
                        df_display["Kaçan Ciro ($)"] = df_display["kacan_ciro_riski"].map("{:,.2f}".format)
                        df_display["Holding Cost ($)"] = df_display["holding_cost"].map("{:,.2f}".format)
                        df_display["Durum"] = df_display["final_stok_durumu"]
                        df_display["Yapay Zeka S&OP Tedarik Emri"] = df_display["final_tedarik_emri"]

                        st.dataframe(
                            df_display[["Tarih", "Talep Tahmini ($)", "Simüle Kalan Stok ($)", "Kaçan Ciro ($)",
                                        "Holding Cost ($)", "Durum", "Yapay Zeka S&OP Tedarik Emri"]],
                            use_container_width=True
                        )

                    else:
                        st.error(f"FastAPI LSTM Motor Hatası: {res_data.get('detail')}")
                except Exception as e:
                    st.error(f"Backend sunucusuna erişilemedi. Hata: {str(e)}")
        else:
            st.info(
                "ℹ️ Sol panelden projeksiyon tarihini, ambar kapasitesini ve senaryo parametrelerini girip 'Global 30 Günlük Forecast'i Tetikle' butonuna basarak yaşayan LSTM simülasyonunu başlatabilirsiniz.")