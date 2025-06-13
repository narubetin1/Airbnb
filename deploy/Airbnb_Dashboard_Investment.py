#streamlit run Airbnb_Dashboard_Investment.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

DB_NAME  = "airbnb_db"
COLLECTION_NAME = "listings"

URI = 'mongodb+srv://Pha22:Pha22@cluster0.okkqifc.mongodb.net/'
client = MongoClient(URI)
collection = client[DB_NAME][COLLECTION_NAME]

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Airbnb Investment Dashboard", layout="wide")

# -------------------------------
# LOAD AND CLEAN DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("080668one_hot_amenities.csv")
    #df = pd.DataFrame(list(collection.find()))

    # ลบ _id ถ้าไม่ใช้
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df['instant_bookable'] = df['instant_bookable'].map({'t': True, 'f': False})
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': True, 'f': False})
    df['room_type'] = df['room_type'].str.strip().str.lower().str.title()
    if 'property_type' in df.columns:
        conditions = [
        df['property_type'].str.lower().str.contains('apartment', na=False),
        df['property_type'].str.lower().str.contains('house', na=False),
        df['property_type'].str.lower().str.contains('condominium', na=False) | df['property_type'].str.lower().str.contains('condo', na=False),
        df['property_type'].str.lower().str.contains('hotel', na=False),
        df['property_type'].str.lower().str.contains('hostel', na=False)
        ]   
    choices = ['Apartment', 'House', 'Condo', 'Hotel', 'Hostel']
    df['property_grouped'] = np.select(conditions, choices, default='Other')
    return df

df = load_data()
st.title("📊 Airbnb Data Analysis for Investment in Bangkok")

# -------------------------------
# Remove Outliers by IQR
# -------------------------------
def remove_outliers_grouped(df, target_col='price'):
    def iqr_filter(group):
        q1 = group[target_col].quantile(0.25)
        q3 = group[target_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[target_col] >= lower) & (group[target_col] <= upper)]
    
    return df.groupby(['neighbourhood', 'room_type', 'property_grouped', 'bedrooms'], group_keys=False).apply(iqr_filter)

df_clean = remove_outliers_grouped(df)

# -------------------------------
# LOAD AREA DATA
# -------------------------------
area_df = pd.read_csv("bangkok_50_districts_area_cleaned.csv")

# -------------------------------
# 1️⃣ Room Type → Property Group
# -------------------------------
st.subheader("🏠 Proportion of Room Types Available for Rent (Bangkok)")

# สร้าง Pie Chart รวมทุก Room Type
room_counts = df_clean['room_type'].value_counts().reset_index()
room_counts.columns = ['room_type', 'count']
fig_pie = px.pie(room_counts, values='count', names='room_type', title="Room Type Distribution in Bangkok")
st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# แสดงกราฟ Property Grouped ต่อจาก Pie Chart
# -------------------------------
st.markdown("### 🏘️ Property Types in Room Type:")

# selectbox
selected_room_type = st.selectbox("Select Room Type:", sorted(df_clean['room_type'].unique()))

# กรองตาม Room Type ที่เลือก
df_room = df_clean[df_clean['room_type'] == selected_room_type]

# สร้าง Bar Chart
grouped_counts = df_room['property_grouped'].value_counts().reset_index()
grouped_counts.columns = ['property_grouped', 'count']
fig_grouped_bar = px.bar(
    grouped_counts,
    x='property_grouped',
    y='count',
    title=f"Property Types in '{selected_room_type}' Listings",
    labels={'property_grouped': 'Property Type', 'count': 'Listings'}
)
st.plotly_chart(fig_grouped_bar, use_container_width=True)

# -------------------------------
# 2️⃣ Stacked Bar Chart: Number of Rooms per Type in Each Neighbourhood
# -------------------------------
st.subheader("📍 Number of Rooms per Type in Each Neighbourhood")
pivot = df_clean.groupby(['neighbourhood', 'room_type']).size().unstack(fill_value=0)
fig_stack = go.Figure()
for room_type in pivot.columns:
    fig_stack.add_trace(go.Bar(name=room_type, x=pivot.index, y=pivot[room_type]))
fig_stack.update_layout(
    barmode='stack',
    title="Number of Rooms per Type in Each Neighbourhood", 
    xaxis_title="Neighbourhood", 
    yaxis_title="Number of Rooms", 
    xaxis_tickangle=-45,
    height=600
)
st.plotly_chart(fig_stack, use_container_width=True)

# -------------------------------
# 3️⃣ Map: Average Price per Neighbourhood (by bedrooms + property group)
# -------------------------------
st.subheader("🗺️ Map of Average Price per Neighbourhood")

# เลือก Room Type
selected_room_type = st.selectbox("Select Room Type:", df_clean['room_type'].unique(), key="map_room_type")

# เลือก Property Group
selected_property_group = st.selectbox("Select Property Group:", sorted(df_clean['property_grouped'].unique()), key="map_property_group")

# กรองข้อมูลตาม Room Type และ Property Group
df_filtered_map = df_clean[
    (df_clean['room_type'] == selected_room_type) &
    (df_clean['property_grouped'] == selected_property_group)
]

# ถ้ามี column 'bedrooms' ให้เลือกเพิ่ม
if 'bedrooms' in df_filtered_map.columns:
    selected_bedroom = st.selectbox(
        "Select Number of Bedrooms:",
        sorted(df_filtered_map['bedrooms'].dropna().unique()),
        key="map_bedroom"
    )
    df_filtered_map = df_filtered_map[df_filtered_map['bedrooms'] == selected_bedroom]

    # คำนวณค่าเฉลี่ย
    df_grouped = df_filtered_map.groupby('neighbourhood')[['latitude', 'longitude', 'price']].mean().reset_index()

    # สร้างแผนที่
    fig_map = px.scatter_mapbox(
        df_grouped,
        lat="latitude",
        lon="longitude",
        size="price",
        color="price",
        hover_name="neighbourhood",
        color_continuous_scale="OrRd",
        size_max=15,
        zoom=10,
        mapbox_style="carto-positron",
        title=f"Average Price of {selected_room_type} ({selected_property_group}, {selected_bedroom} Bedrooms) per Neighbourhood"
    )
    fig_map.update_layout(height=800)
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("No bedroom information available in this dataset.")


# -------------------------------
# 4️⃣ Price Distribution per Neighbourhood (เชื่อมกับ Map section)
# -------------------------------
st.subheader("💵 Price Distribution per Neighbourhood")

# กรองข้อมูลจากตัวแปรที่ Map section ใช้
df_price_filtered = df_clean[
    (df_clean['room_type'] == selected_room_type) &
    (df_clean['bedrooms'] == selected_bedroom)
]
if selected_property_group != "All":
    df_price_filtered = df_price_filtered[df_price_filtered['property_grouped'] == selected_property_group]

# ให้เลือกเขต
selected_neigh = st.selectbox(
    "Select Neighbourhood to view price distribution:",
    sorted(df_price_filtered['neighbourhood'].unique()),
    key="dist_neigh"
)

# กรองข้อมูลในเขตนั้น
df_neigh = df_price_filtered[df_price_filtered['neighbourhood'] == selected_neigh]

# วาด histogram
fig_hist = px.histogram(
    df_neigh,
    x="price",
    nbins=50,
    title=f"Price Distribution in {selected_neigh} ({selected_room_type}, {selected_property_group}, {selected_bedroom} Bedrooms)",
    labels={"price": "Price (Baht)"}
)
st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------
# 5️⃣ Investment Location Analysis (Bubble Chart)
# -------------------------------
st.subheader("📈 Airbnb Investment Location Analysis")

# เลือก Room Type
room_option = st.selectbox(
    "Select Room Type for Investment Analysis:",
    sorted(df_clean['room_type'].unique()),
    key="inv_room_type"
)

# เลือก Property Group (มีตัวเลือก All)
property_group_list = ["All"] + sorted(df_clean['property_grouped'].dropna().unique())
property_option = st.selectbox(
    "Select Property Group for Investment Analysis:",
    property_group_list,
    key="inv_property_group"
)

# เลือก Bedrooms
bedroom_option = st.selectbox(
    "Select Number of Bedrooms for Investment Analysis:",
    sorted(df_clean['bedrooms'].dropna().unique()),
    key="inv_bedroom"
)

# กรองข้อมูลตาม Room Type และ Bedrooms
invest_df = df_clean[
    (df_clean['room_type'] == room_option) &
    (df_clean['bedrooms'] == bedroom_option)
]

# ถ้าเลือก Property Group แบบเฉพาะ กรองเพิ่ม
if property_option != "All":
    invest_df = invest_df[invest_df['property_grouped'] == property_option]

# ลบ outliers
invest_filtered = remove_outliers_grouped(invest_df)

# สรุปข้อมูลระดับเขต
summary_df = (
    invest_filtered.groupby('neighbourhood')
    .agg(avg_price=('price', 'mean'),
         avg_rating=('review_scores_rating', 'mean'))
    .join(invest_filtered.groupby('neighbourhood').size().rename('room_count'))
    .reset_index()
)

# รวมกับข้อมูลพื้นที่
summary_df = summary_df.merge(area_df, on='neighbourhood', how='left')
summary_df['density'] = summary_df['room_count'] / summary_df['area_km2']

# แสดงผล Bubble Chart
fig_invest = px.scatter(
    summary_df,
    x='density',
    y='avg_price',
    size='avg_rating',
    color='neighbourhood',
    hover_name='neighbourhood',
    hover_data={
        'room_count': True,
        'area_km2': True,
        'avg_price': True,
        'avg_rating': True,
        'density': True
    },
    title=f'Ideal Investment Locations ({room_option}, {property_option}, {bedroom_option} Bedrooms)',
    labels={
        'density': 'Rooms per Sq. Km.',
        'avg_price': 'Average Price per Night',
        'avg_rating': 'Average Review Rating'
    },
    height=650
)
fig_invest.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig_invest, use_container_width=True)

# -------------------------------
# 6️⃣ Word Cloud of Amenities (Filtered)
# -------------------------------
st.subheader("☁️ Word Cloud: Airbnb Amenities (Filtered)")

# ให้ผู้ใช้เลือก Room Type และ Property Grouped
col1, col2 = st.columns(2)
with col1:
    selected_wc_room_type = st.selectbox(
        "Select Room Type:", 
        sorted(df_clean['room_type'].unique()), 
        key="wc_room_type"
    )
with col2:
    selected_wc_prop_group = st.selectbox(
        "Select Property Group:", 
        sorted(df_clean['property_grouped'].unique()), 
        key="wc_property_group"
    )

# กรองข้อมูลตาม room_type และ property_grouped
df_wc = df_clean[
    (df_clean['room_type'] == selected_wc_room_type) &
    (df_clean['property_grouped'] == selected_wc_prop_group)
]

# คอลัมน์ amenities: AG ถึง BP (index 32–65)
try:
    amenity_cols = df_wc.columns[32:66]  # Adjust if your columns change
    word_freq = {
        col.replace('_', ' ').title(): df_wc[col].sum()
        for col in amenity_cols if df_wc[col].sum() > 0
    }

    if word_freq:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        fig_wc, ax = plt.subplots(figsize=(6, 3), dpi=200)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
    else:
        st.warning("No amenity data available for this combination.")
except Exception as e:
    st.error(f"Error generating Word Cloud: {e}")