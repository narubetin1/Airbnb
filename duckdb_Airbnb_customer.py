import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import duckdb
from geopy.distance import geodesic
 
st.set_page_config(page_title="Airbnb For Travelers", layout="wide")
st.title('AIRBNB FOR TRAVELERS')
 
##LOCAL_FILE = r'C:\Users\fah_b\Desktop\final-5001\080668one_hot_amenities_1.csv'
 
# Load landmarks
landmarks = pd.read_csv('bangkok_attractions_cleaned.csv')
 
@st.cache_data
def load_data():
    data = pd.read_csv('080668one_hot_amenities.csv')
    data = data.dropna(subset=['price', 'review_scores_rating', 'latitude', 'longitude'])
    data['price'] = data['price'].astype(str)
    return data
 
data = load_data()

st.subheader("🔍 FindMyPlace – Discover Your Perfect Airbnb Experience")

def build_where_clause(neighbourhoods, price_ranges, property_groups, room_types, amenities):
    conditions = []
 
    if neighbourhoods:
        neighbourhoods_escaped = [f"'{n}'" for n in neighbourhoods]
        conditions.append(f"neighbourhood IN ({', '.join(neighbourhoods_escaped)})")
 
    if price_ranges:
        price_conds = []
        for low, high in price_ranges:
            price_conds.append(f"(CAST(REPLACE(price, '$', '') AS DOUBLE) BETWEEN {low} AND {high})")
        conditions.append("(" + " OR ".join(price_conds) + ")")
 
    if property_groups:
        pg_escaped = [f"'{pg}'" for pg in property_groups]
        conditions.append(f"property_grouped IN ({', '.join(pg_escaped)})")
 
    if room_types:
        rt_escaped = [f"'{rt}'" for rt in room_types]
        conditions.append(f"room_type IN ({', '.join(rt_escaped)})")
 
    for amenity in amenities:
        conditions.append(f"{amenity} = 1")
 
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    else:
        return ""
 
# UI filters
selected_neighbourhood = []
if 'neighbourhood' in data.columns:
    selected_neighbourhood = st.multiselect("Select neighbourhood", data['neighbourhood'].dropna().unique())
 
price_ranges = [(1, 1000), (1001, 2000), (2001, 3000), (3001, 4000),
                (4001, 5000), (5001, 6000), (6001, 7000), (7001, 8000),
                (8001, 9000), (9001, 10000), (10001, 11000), (11001, 12000),
                (12001, 13000), (13001, 14000), (14001, 15000), (15001, 16000),
                (16001, 17000), (17001, 18000)]
select_all = st.checkbox("Select All Price Ranges")
selected_ranges = []
cols = st.columns(3)
for i, (low, high) in enumerate(price_ranges):
    col = cols[i % 3]
    label = f"{low:,} - {high:,} THB"
    checked = col.checkbox(label, key=f"price_range_{i}", value=select_all)
    if checked:
        selected_ranges.append((low, high))
 
property_groups = []
if 'property_type' in data.columns:
    conditions = [
        data['property_type'].str.lower().str.contains('apartment', na=False),
        data['property_type'].str.lower().str.contains('house', na=False),
        data['property_type'].str.lower().str.contains('condominium', na=False) | data['property_type'].str.lower().str.contains('condo', na=False),
        data['property_type'].str.lower().str.contains('hotel', na=False),
        data['property_type'].str.lower().str.contains('hostel', na=False)
    ]
    choices = ['Apartment', 'House', 'Condo', 'Hotel', 'Hostel']
    data['property_grouped'] = np.select(conditions, choices, default='Other')
 
    property_groups = data['property_grouped'].dropna().unique()
selected_property_groups = st.multiselect("Select Property Type Grouped", property_groups)
 
room_types = []
if 'room_type' in data.columns:
    room_types = data['room_type'].dropna().unique()
selected_room_types = st.multiselect("Select Room Type", room_types)
 
amenity_cols = [
    'Air conditioning', 'Bed linens', 'Breakfast', 'Coffee maker', 'Dedicated workspace',
    'Dishes and silverware', 'Dryer', 'Elevator', 'Essentials', 'Extra pillows and blankets',
    'Fire extinguisher', 'First aid kit', 'Free parking', 'Garden or backyard', 'Gym',
    'Hair dryer', 'Hangers', 'Heating', 'Host greets you', 'Hot water', 'Iron', 'Kitchen',
    'Lock on bedroom door', 'Lockbox', 'Luggage dropoff allowed', 'Microwave',
    'Patio or balcony', 'Pool', 'Refrigerator', 'Room-darkening shades', 'Shampoo',
    'Shower gel', 'Smoke alarm', 'TV', 'Washer', 'Wifi'
]

selected_amenities = st.multiselect("Select Amenities", amenity_cols)

invalid_amenities = [am for am in selected_amenities if am not in data.columns]
if invalid_amenities:
    st.error(f"These amenities are not in the data: {', '.join(invalid_amenities)}")
    st.stop()

# ✅ START: ตรงนี้คือจุดที่ต้องใส่ "ข้อ 1" แบบไม่ใช้ฟังก์ชัน
conditions = []

# neighbourhood
if selected_neighbourhood:
    hoods = [f"'{n}'" for n in selected_neighbourhood]
    conditions.append(f"neighbourhood IN ({', '.join(hoods)})")

# price range
if selected_ranges:
    price_conds = [f"(CAST(REPLACE(price, '$', '') AS DOUBLE) BETWEEN {low} AND {high})"
                   for (low, high) in selected_ranges]
    conditions.append("(" + " OR ".join(price_conds) + ")")

# property type
if selected_property_groups:
    props = [f"'{p}'" for p in selected_property_groups]
    conditions.append(f"property_grouped IN ({', '.join(props)})")

# room type
if selected_room_types:
    rooms = [f"'{r}'" for r in selected_room_types]
    conditions.append(f"room_type IN ({', '.join(rooms)})")

# amenities
for amenity in selected_amenities:
    safe_col = amenity.replace('"', '""')
    conditions.append(f'"{safe_col}" = 1')

# สร้าง where clause
if conditions:
    where_clause = "WHERE " + " AND ".join(conditions)
else:
    where_clause = ""
# ✅ END

# สร้าง query
query = f"""
SELECT *, CAST(REPLACE(price, '$', '') AS DOUBLE) AS price_num
FROM data
{where_clause}
"""
#st.code(query, language='sql')  # แสดง SQL เพื่อ debug

# Run DuckDB query
con = duckdb.connect()
con.register('data', data)
filtered_data = con.execute(query).df()

if filtered_data.empty:
    st.info("No data matches the selected filters. Please adjust your filter criteria.")
    st.stop()
 
st.markdown("### 📊 Overview Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg. Price (per night)", f"{filtered_data['price_num'].mean():,.2f} THB")
col2.metric("Avg. Review Score", f"{filtered_data['review_scores_rating'].mean():.1f}")
col3.metric("Total Listings", f"{filtered_data.shape[0]}")
 
def neighborhood_to_color(name):
    h = abs(hash(name))
    r = 50 + (h % 206)
    g = 50 + ((h // 206) % 206)
    b = 50 + ((h // (206*206)) % 206)
    return [r, g, b]
 
unique_hoods_all = data['neighbourhood'].dropna().unique()
hood_color_map = {hood: neighborhood_to_color(hood) for hood in unique_hoods_all}
 
st.subheader("🗺️ Explore Stays by Neighborhoods")
if all(col in filtered_data.columns for col in ['latitude', 'longitude', 'neighbourhood']):
    filtered_data['color'] = filtered_data['neighbourhood'].map(hood_color_map)
 
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=filtered_data['latitude'].mean(),
            longitude=filtered_data['longitude'].mean(),
            zoom=11,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_data,
                get_position='[longitude, latitude]',
                get_color='color',
                get_radius=150,
                pickable=True,
                auto_highlight=True,
            )
        ],
        tooltip={
            "html": "<b>Name:</b> {name}<br/>"
                    "<b>Price:</b> {price_num} THB<br/>"
                    "<b>Neighborhood:</b> {neighbourhood}<br/>"
                    "<b>Room Type:</b> {room_type}<br/>"
                    "<b>Review Score:</b> {review_scores_rating}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))
 
if not filtered_data.empty and 'neighbourhood' in filtered_data.columns:
    st.markdown("###### Summary by Neighbourhood")
 
    summary = filtered_data.groupby('neighbourhood').agg(
        avg_price = ('price_num', 'mean'),
        avg_review = ('review_scores_rating', 'mean'),
        total_listings = ('name', 'count')
    ).reset_index()
 
    summary = summary.sort_values('avg_price', ascending=False)
 
    summary_display = summary.rename(columns={
        'neighbourhood': 'Neighbourhood',
        'avg_price': 'Avg. Price (THB)',
        'avg_review': 'Avg. Review Score',
        'total_listings': 'Total Listings'
    })
    table_width = 1800
    table_height = 600
    st.dataframe(
    summary_display.style.format({
        'Avg. Price (THB)': '{:,.2f}',
        'Avg. Review Score': '{:.1f}',
        'Total Listings': '{:,}'
    }),
    width=table_width,
    height=table_height
    )

    fig = px.bar(summary, x='neighbourhood', y='avg_price',
                 labels={'neighbourhood': 'Neighbourhood', 'avg_price': 'Average Price (THB)'},
                 text=summary['avg_price'].apply(lambda x: f"{x:,.0f}"),
                 color_discrete_sequence=["#ce096c"])
 
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, yaxis=dict(title='Average Price (THB)'))
 
    st.plotly_chart(fig, use_container_width=True)
 
nights = st.slider("Number of nights", 1, 365, 1)
 
filtered_data['total_price'] = filtered_data['price_num'] * nights
 
st.subheader("📝 Overall Filtered Listings")
st.write(f"Total listings found: {filtered_data.shape[0]}")

filtered_data[amenity_cols] = filtered_data[amenity_cols].replace({1: '✔', 0: '✘'})
columns_to_show = ['name', 'neighbourhood', 'room_type', 'price', 'total_price'] + amenity_cols
columns_to_show = [col for col in columns_to_show if col in filtered_data.columns]
 
st.dataframe(filtered_data[columns_to_show])
 
lat_col = 'Latitude' if 'Latitude' in data.columns else 'latitude'
lng_col = 'Longitude' if 'Longitude' in data.columns else 'longitude'
 
if lat_col and lng_col:
    landmark_names = landmarks['Place'].dropna().unique().tolist()
    selected_landmark = st.selectbox("Selected Landmark", landmark_names, key="landmark_select")
 
    landmark_row = landmarks[landmarks['Place'] == selected_landmark].iloc[0]
    landmark_coords = (landmark_row['Latitude'], landmark_row['Longitude'])
 
    def safe_distance(x):
        try:
            if np.isnan(x[lat_col]) or np.isnan(x[lng_col]):
                return np.nan
            return geodesic((x[lat_col], x[lng_col]), landmark_coords).km
        except:
            return np.nan
 
    dist_col = f"dist_to_{selected_landmark}"
    filtered_data[dist_col] = filtered_data.apply(safe_distance, axis=1)
 
    st.markdown(f"### 🏨 Top 10 Locations near {selected_landmark}")
    top_nearest = filtered_data.dropna(subset=[dist_col]).sort_values(dist_col).head(10)
    st.dataframe(top_nearest[['name', 'neighbourhood', 'total_price', 'review_scores_rating', dist_col]])
 
    st.plotly_chart(px.scatter(
        filtered_data.dropna(subset=[dist_col]),
        x=dist_col,
        y="price",
        hover_name="name",
        labels={dist_col: f"Distance to {selected_landmark} (km)", "total_price": "Price (THB)"},
        title="Price vs Distance to Selected Landmark"
    ), use_container_width=True)
    
    #dist_col = "distance_to_landmark" ## add

    dist_str_col = f"{dist_col}_str"
    top_nearest[dist_str_col] = top_nearest[dist_col].apply(lambda x: f"{x:.2f} km" if pd.notna(x) else "N/A")
 
    tooltip = {
        "html": f"""
            <b>Name:</b> {{name}}<br/>
            <b>Price:</b> {{price}} THB<br/>
            <b>Neighborhood:</b> {{neighbourhood}}<br/>
            <b>Distance to {selected_landmark}:</b> {{{dist_str_col}}}
        """,
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    LANDMARK_ICON_URL = "https://cdn-icons-png.flaticon.com/512/684/684908.png"  ## add
    # --- Landmark Icon Data (ตำแหน่งหมุด landmark) ---
    landmark_icon_data = [{
        "lat": landmark_row['Latitude'],
        "lon": landmark_row['Longitude'],
        "icon_url": LANDMARK_ICON_URL
    }]

    # --- IconLayer สำหรับ landmark (หมุด) ---
    icon_layer = pdk.Layer(
        "IconLayer",
        data=landmark_icon_data,
        get_position='[lon, lat]',
        get_icon="icon_data",
        size_scale=12,
        pickable=True,
        icon_mapping={
            "icon_data": {
                "url": LANDMARK_ICON_URL,
                "width": 512,
                "height": 512,
                "anchorY": 512   # ชี้ที่ปลายหมุด
            }
        },
        get_size=50,
        get_icon_url="icon_url",
    )

    # --- ScatterplotLayer สำหรับบ้านหรือที่พัก ---
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        data=top_nearest,
        get_position='[longitude, latitude]',
        get_color=[0, 128, 255, 160],
        get_radius=200,
        pickable=True,
        auto_highlight=True,
    )




    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=landmark_row['Latitude'],
            longitude=landmark_row['Longitude'],
            zoom=13,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=top_nearest,
                get_position='[longitude, latitude]',
                get_color=[0, 128, 255, 160],
                get_radius=200,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=[{'lat': landmark_row['Latitude'], 'lon': landmark_row['Longitude']}],
                get_position='[lon, lat]',
                get_color=[255, 0, 0, 200],
                get_radius=300,
                pickable=False,
            )
        ],
        tooltip=tooltip
    ))