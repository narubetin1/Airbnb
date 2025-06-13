import streamlit as st
import pandas as pd
from pymongo import MongoClient

# ----- MongoDB Atlas Connection -----
client = MongoClient('mongodb+srv://Pha22:Pha22@cluster0.okkqifc.mongodb.net/')
db = client["airbnb_db"]
collection = db["listings"]

# ----- Predefined Amenities List -----
amenities_list = [
    "Air conditioning", "Bed linens", "Breakfast", "Coffee maker", "Dedicated workspace",
    "Dishes and silverware", "Dryer", "Elevator", "Essentials", "Extra pillows and blankets",
    "Fire extinguisher", "First aid kit", "Free parking", "Garden or backyard", "Gym",
    "Hair dryer", "Hangers", "Heating", "Host greets you", "Hot water", "Iron", "Kitchen",
    "Lock on bedroom door", "Lockbox", "Luggage dropoff allowed", "Microwave",
    "Patio or balcony", "Pool", "Refrigerator", "Room-darkening shades", "Shampoo",
    "Shower gel", "Smoke alarm", "TV", "Washer", "Wifi"
]

# ----- Clear Form Trigger -----
if "clear_form_triggered" in st.session_state and st.session_state.clear_form_triggered:
    st.session_state.clear_form_triggered = False
    for key in list(st.session_state.keys()):
        if key.startswith("amenity_") or key in (
            "name", "host_id", "property_type", "price", "minimum_nights", "latitude", "longitude", "other_amenities"
        ):
            st.session_state[key] = "" if isinstance(st.session_state.get(key), str) else 0
        if key == "select_all":
            st.session_state[key] = False
    st.rerun()

# ----- UI: Add Listing Form -----
st.title("🏠 Add New Airbnb Listing")

name = st.text_input("Name", key="name")
host_id = st.text_input("Host ID", key="host_id")

# ---- Safe loading of options ----
neighbourhoods = collection.distinct("neighbourhood")
if not isinstance(neighbourhoods, list) or len(neighbourhoods) == 0:
    neighbourhoods = ["Unknown"]
neighbourhoods = sorted([str(n) for n in neighbourhoods if n])

room_types = collection.distinct("room_type")
if not room_types:
    room_types = ["Entire Place", "Private Room"]
room_types = sorted([str(r) for r in room_types if r])

property_types = collection.distinct("property_type")
if not property_types:
    property_types = ["Apartment", "House", "Condo", "Other"]
property_types = sorted([str(p) for p in property_types if p])

# ----- Validate session_state before using selectbox -----
if "property_type" in st.session_state:
    if st.session_state.property_type not in property_types:
        st.session_state.property_type = property_types[0]

neighbourhood = st.selectbox("Neighbourhood", neighbourhoods, key="neighbourhood")
room_type = st.selectbox("Room Type", room_types, key="room_type")
property_type = st.selectbox("Property Type", property_types, key="property_type")

price = st.number_input("Price per Night", min_value=0, key="price")

# ----- Validate session state for minimum_nights -----
if "minimum_nights" in st.session_state:
    try:
        if int(st.session_state.minimum_nights) < 1:
            st.session_state.minimum_nights = 1
    except:
        st.session_state.minimum_nights = 1

minimum_nights = st.number_input("Minimum Nights", min_value=1, key="minimum_nights")

st.subheader("🧾 Amenities")
select_all = st.checkbox("Select All Amenities", value=st.session_state.get("select_all", False), key="select_all")
selected_amenities = []
cols = st.columns(3)

for i, amenity in enumerate(amenities_list):
    with cols[i % 3]:
        checked = st.checkbox(amenity, value=select_all, key=f"amenity_{amenity}")
        if checked:
            selected_amenities.append(amenity)

new_amenity_input = st.text_input("Other Amenities (comma separated)", key="other_amenities")
if new_amenity_input and new_amenity_input.strip():
    custom_amenities = [a.strip().title() for a in new_amenity_input.split(",") if a.strip()]
    selected_amenities.extend(custom_amenities)

latitude = st.number_input("Latitude", format="%.6f", key="latitude")
longitude = st.number_input("Longitude", format="%.6f", key="longitude")

submitted = st.button("➕ Add Listing")

# ----- Insert into MongoDB -----
if submitted:
    new_listing = {
        "name": name,
        "host_id": host_id,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "property_type": property_type,
        "price": price,
        "minimum_nights": minimum_nights,
        "amenities": list(set(selected_amenities)),
        "location": {
            "latitude": latitude,
            "longitude": longitude
        }
    }

    collection.insert_one(new_listing)
    st.success("✅ Listing added successfully!")

# ----- Clear Form Button -----
if st.button("Clear Form"):
    st.session_state.clear_form_triggered = True
    st.rerun()

# ----- Show recent entries -----
st.subheader("🗂️ Recently Added Listings")
latest = pd.DataFrame(list(collection.find().sort("_id", -1).limit(5)))
if not latest.empty:
    desired_cols = ["name", "neighbourhood", "room_type", "property_type", "price", "amenities"]
    available_cols = [col for col in desired_cols if col in latest.columns]
    if "amenities" in latest.columns:
        latest["amenities"] = latest["amenities"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    st.dataframe(latest[available_cols])
else:
    st.info("No listings found.")
