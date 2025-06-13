import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV # นำเข้า GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.cluster import KMeans # นำเข้า KMeans สำหรับการจัดกลุ่ม
import warnings
warnings.filterwarnings("ignore")
from pymongo import MongoClient

#MongoDB
DB_NAME  = "airbnb_db"
COLLECTION_NAME = "listings"

URI = 'mongodb+srv://Pha22:Pha22@cluster0.okkqifc.mongodb.net/'
client = MongoClient(URI)
collection = client[DB_NAME][COLLECTION_NAME]

# กำหนดการตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="📈 ML Airbnb Price Prediction", layout="wide") 
st.title("🤖 Airbnb Price Prediction (Feature Selection)") 

# ฟังก์ชันสำหรับโหลดและเตรียมข้อมูล โดยใช้ @st.cache_data เพื่อแคชข้อมูล
@st.cache_data
def load_data():
    #df = pd.read_csv("080668one_hot_amenities_1.csv")
    df = pd.DataFrame(list(collection.find()))

    # ลบ _id ถ้าไม่ใช้
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float) # แปลงคอลัมน์ราคาให้เป็นตัวเลข
    df['instant_bookable'] = df['instant_bookable'].map({'t': True, 'f': False}) # แปลง 't'/'f' เป็น True/False
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': True, 'f': False}) # แปลง 't'/'f' เป็น True/False
    df['room_type'] = df['room_type'].str.strip().str.lower().str.title() # ทำความสะอาดและจัดรูปแบบประเภทห้องพัก
    df['property_type'] = df['property_type'].astype(str).str.strip().str.title() # ทำความสะอาดและจัดรูปแบบประเภทที่พัก
    df.fillna(0, inplace=True) # เติมค่าว่าง (NaN) ทั้งหมดด้วย 0
    df['amenities_count'] = df.iloc[:, 32:66].sum(axis=1) # นับจำนวนสิ่งอำนวยความสะดวกที่มี (สมมติว่าเป็นคอลัมน์ 0/1)
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

def remove_outliers_grouped(df, target_col='price'):
    def iqr_filter(group):
        q1 = group[target_col].quantile(0.25)
        q3 = group[target_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[target_col] >= lower) & (group[target_col] <= upper)]
    
    return df.groupby(['neighbourhood', 'room_type', 'property_grouped', 'bedrooms'], group_keys=False).apply(iqr_filter)

df_raw = load_data()
df = remove_outliers_grouped(df_raw)
amenity_cols = ["Air conditioning", "Bed linens", "Breakfast", "Coffee maker", "Dedicated workspace",
    "Dishes and silverware", "Dryer", "Elevator", "Essentials", "Extra pillows and blankets",
    "Fire extinguisher", "First aid kit", "Free parking", "Garden or backyard", "Gym",
    "Hair dryer", "Hangers", "Heating", "Host greets you", "Hot water", "Iron", "Kitchen",
    "Lock on bedroom door", "Lockbox", "Luggage dropoff allowed", "Microwave",
    "Patio or balcony", "Pool", "Refrigerator", "Room-darkening shades", "Shampoo",
    "Shower gel", "Smoke alarm", "TV", "Washer", "Wifi"] # กำหนดคอลัมน์สิ่งอำนวยความสะดวก

# --- ส่วนติดต่อผู้ใช้ (UI) สำหรับรับข้อมูล ---
st.subheader("📌 Input for Prediction")
selected_neighbourhood = st.selectbox("Select Neighborhood:", sorted(df['neighbourhood'].dropna().unique()))
selected_room_type = st.selectbox("Select Room Type:", sorted(df['room_type'].dropna().unique()))
selected_property_grouped = st.selectbox("Select Property:", sorted(df['property_grouped'].dropna().unique()))
selected_bedrooms = st.number_input("Number of Bedrooms:", min_value=0, value=1)
min_nights = st.number_input("Minimum Nights:", min_value=1, value=1)

st.markdown("**Select Amenities:**")
select_all = st.checkbox("Select All")
selected_amenities = []
amenity_col1, amenity_col2 = st.columns(2)
with amenity_col1:
    for i, col in enumerate(amenity_cols[:len(amenity_cols)//2]):
        checked = st.checkbox(col.replace('_', ' ').title(), key=f"amenity1_{i}", value=select_all)
        if checked:
            selected_amenities.append(col)
with amenity_col2:
    for i, col in enumerate(amenity_cols[len(amenity_cols)//2:]):
        checked = st.checkbox(col.replace('_', ' ').title(), key=f"amenity2_{i}", value=select_all)
        if checked:
            selected_amenities.append(col)

# --- เตรียมข้อมูลสำหรับการทำ Feature Selection ---
# ฟีเจอร์หลัก (ไม่รวมสิ่งอำนวยความสะดวก)
base_cols = [
    'neighbourhood', 'room_type', 'property_grouped', 'bedrooms',
    'minimum_nights', 'instant_bookable', 'host_is_superhost', 'amenities_count'
]
used_cols = base_cols + list(amenity_cols) # รวมฟีเจอร์หลักกับคอลัมน์สิ่งอำนวยความสะดวก
X_raw = df[used_cols] # ข้อมูลฟีเจอร์ดิบ
y = df['price'] # คอลัมน์เป้าหมาย (ราคา)

# คอลัมน์ประเภท (Categorical) และตัวเลข (Numeric)
categorical_cols = ['neighbourhood', 'room_type', 'property_grouped']
numeric_cols = [c for c in used_cols if c not in categorical_cols]

# สร้าง Dummy Variables และคัดเลือกฟีเจอร์
X_dummy = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True) # ทำ One-hot encoding ให้คอลัมน์ประเภท
X_dummy = X_dummy.fillna(0) # เติมค่าว่างอีกครั้งหลังจากทำ dummy variables
selector = VarianceThreshold(threshold=0.01) # เลือกฟีเจอร์ที่มีความแปรปรวนสูงกว่าเกณฑ์
X_sel = selector.fit_transform(X_dummy)
feat_var = X_dummy.columns[selector.get_support()] # ชื่อฟีเจอร์ที่ผ่าน VarianceThreshold

k = min(25, len(feat_var)) # กำหนดจำนวนฟีเจอร์สูงสุดที่จะเลือก (ไม่เกิน 25)
kbest = SelectKBest(f_regression, k=k) # เลือก K ฟีเจอร์ที่ดีที่สุดตามความสัมพันธ์กับราคา
X_k = kbest.fit_transform(X_dummy[feat_var], y)
feat_selected = feat_var[kbest.get_support()] # ชื่อฟีเจอร์ที่ถูกเลือกสุดท้ายสำหรับใช้เทรนโมเดล

st.write("🔎 **Features used for training:**", list(feat_selected))

# --- เตรียม Dictionary สำหรับ Input ของผู้ใช้ และกรองเฉพาะฟีเจอร์ที่ถูกเลือก ---
input_dict = {
    'bedrooms': [selected_bedrooms],
    'minimum_nights': [min_nights],
    'instant_bookable': [True],
    'host_is_superhost': [False],
    'amenities_count': [len(selected_amenities)],
}
for col in amenity_cols:
    input_dict[col] = [1 if col in selected_amenities else 0]

# สำหรับ Dummy Variables ของคอลัมน์ประเภท (Categorical)
for c in categorical_cols:
    # ใช้ตัวแปรที่รับค่าจาก UI โดยตรง เพื่อความแข็งแกร่งของโค้ด
    if c == 'neighbourhood':
        selected_val = selected_neighbourhood
    elif c == 'room_type':
        selected_val = selected_room_type
    elif c == 'property_grouped':
        selected_val = selected_property_grouped
    else:
        selected_val = None # ไม่ควรเกิดขึ้นกับ categorical_cols ที่กำหนดไว้

    for v in df[c].dropna().unique():
        col_name_dummy = f"{c}_{v}"
        # เพิ่มเฉพาะคอลัมน์ Dummy Variable ที่ปรากฏอยู่ใน feat_selected
        if col_name_dummy in feat_selected:
            input_dict[col_name_dummy] = [1 if (selected_val == v) else 0]
        else:
            input_dict[col_name_dummy] = [0] # กำหนดเป็น 0 หากคอลัมน์นี้ไม่ได้ถูกเลือกใน feat_selected

# ตรวจสอบให้แน่ใจว่าคอลัมน์ทั้งหมดใน feat_selected มีอยู่ใน input_dict (ถ้าไม่มีให้เติม 0)
for col in feat_selected:
    if col not in input_dict:
        input_dict[col] = [0]

input_df = pd.DataFrame(input_dict)

# จัดเรียงคอลัมน์ของ input_df ให้ตรงกับลำดับและฟีเจอร์ที่อยู่ใน feat_selected
input_df = input_df.reindex(columns=feat_selected, fill_value=0)
X_train = X_dummy[feat_selected] # ข้อมูลสำหรับเทรนโมเดล (ใช้เฉพาะฟีเจอร์ที่ถูกเลือกแล้ว)

# --- เลือกโมเดลและทำการฝึก (Train) ---
model_choice = st.selectbox("Select Model:", ["Random Forest", "Linear Regression", "Decision Tree", "XGBoost"])

# เพิ่มตัวเลือกในการปรับจูน Hyperparameters
st.subheader("⚙️ Hyperparameter Tuning")
perform_tuning = st.checkbox("Perform Hyperparameter Tuning (may take time)")

if st.button("🚀 Predict Price"):
    with st.spinner("Processing..."): 
        model_to_fit = None # ตัวแปรสำหรับเก็บโมเดลที่จะใช้หลังจากปรับจูนหรือไม่ปรับจูน

        if model_choice == "Random Forest":
            if perform_tuning:
                st.write("Tuning Random Forest Regressor...") 
                # กำหนด Hyperparameter Grid สำหรับ GridSearchCV
                param_grid = {
                    'n_estimators': [50, 100, 150], # จำนวนต้นไม้
                    'max_depth': [5, 8, 10, None], # ความลึกสูงสุดของต้นไม้ (None คือไม่จำกัด)
                    'min_samples_split': [2, 5], # จำนวนตัวอย่างขั้นต่ำที่ต้องมีเพื่อแบ่งโหนด
                    'min_samples_leaf': [1, 2] # จำนวนตัวอย่างขั้นต่ำในใบ
                }
                # ใช้ GridSearchCV เพื่อค้นหา Hyperparameters ที่ดีที่สุด
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=42),
                    param_grid=param_grid,
                    cv=3, # จำนวนเท่าของ cross-validation
                    scoring='neg_mean_absolute_error', # ใช้ MAE (ค่าลบ)
                    n_jobs=-1, # ใช้ทุก core ของ CPU
                    verbose=1 # แสดงความคืบหน้า
                )
                grid_search.fit(X_train, y)
                model_to_fit = grid_search.best_estimator_
                st.write(f"Random Forest Best Hyperparameters: {grid_search.best_params_}")
            else:
                model_to_fit = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)

        elif model_choice == "Decision Tree":
            if perform_tuning:
                st.write("Tuning Decision Tree Regressor...") 
                param_grid = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 3, 5]
                }
                grid_search = GridSearchCV(
                    estimator=DecisionTreeRegressor(random_state=42),
                    param_grid=param_grid,
                    cv=3,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y)
                model_to_fit = grid_search.best_estimator_
                st.write(f"Decision Tree Best Hyperparameters: {grid_search.best_params_}")
            else:
                model_to_fit = DecisionTreeRegressor(max_depth=6, random_state=42)

        elif model_choice == "XGBoost":
            if perform_tuning:
                st.write("Tuning XGBoost Regressor...") 
                # เนื่องจาก XGBoost มี Hyperparameters เยอะและใช้เวลาจูนนาน
                # แนะนำให้ใช้ RandomizedSearchCV ในการค้นหาเบื้องต้น
                # หรือกำหนด param_grid ที่เล็กลงสำหรับ GridSearchCV
                param_distributions = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'gamma': [0, 0.1, 0.2]
                }
                # ใช้ RandomizedSearchCV เพื่อหาค่าที่ดีในเวลาที่เหมาะสมกว่า
                random_search = RandomizedSearchCV(
                    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
                    param_distributions=param_distributions,
                    n_iter=20, # จำนวนการทดลอง (ลดลงเพื่อความเร็ว)
                    cv=3,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )
                random_search.fit(X_train, y)
                model_to_fit = random_search.best_estimator_
                st.write(f"XGBoost Best Hyperparameters: {random_search.best_params_}")
            else:
                model_to_fit = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.15, objective='reg:squarederror', random_state=42)
        else: # Linear Regression ไม่จำเป็นต้องปรับจูน Hyperparameters มากนัก
            model_to_fit = LinearRegression()
            if perform_tuning:
                st.warning("Linear Regression does not significantly benefit from hyperparameter tuning.") 


        # ตรวจสอบว่าโมเดลถูกกำหนดแล้ว หากเกิดข้อผิดพลาดจะหยุดการทำงานตรงนี้
        if model_to_fit is None:
            st.error("Error: Could not create model. Please select a model and retry.") 
            st.stop() # ใช้ st.stop() เพื่อหยุดการทำงานของสคริปต์ใน Streamlit

        # ทำ Cross-Validation และทำนายด้วยโมเดลที่ได้
        scores = cross_val_score(model_to_fit, X_train, y, cv=3, scoring='neg_mean_absolute_error')
        mae = -np.mean(scores)
        model_to_fit.fit(X_train, y) # ฝึกโมเดลสุดท้ายด้วยข้อมูลทั้งหมด
        pred_price = model_to_fit.predict(input_df)[0]

        if perform_tuning:
            st.info("💡 Model has been hyperparameter tuned.") 

        st.success(f"✅ Predicted Price: {pred_price:,.0f} Baht/Night") 
        st.info(f"MAE from Cross-Validation: {mae:,.2f} Baht") 

        # --- เพิ่ม Metric: RMSE และ R² ---
        from sklearn.metrics import mean_squared_error, r2_score

        # ทำการทำนายทั้งหมดเพื่อวัดผล
        y_pred_all = model_to_fit.predict(X_train)

        rmse = np.sqrt(mean_squared_error(y, y_pred_all))
        r2 = r2_score(y, y_pred_all)

        st.info(f"RMSE on Training Data: {rmse:,.2f} Baht")
        st.info(f"R² on Training Data: {r2:.4f}")

        # --- ส่วน Clustering เฉพาะ Neighbourhood ---
        df_cluster = df[df['neighbourhood'] == selected_neighbourhood][used_cols].copy()
        df_cluster['price'] = df[df['neighbourhood'] == selected_neighbourhood]['price']

        # ข้อมูล Input ของผู้ใช้
        temp_input_cluster = pd.DataFrame(columns=used_cols)
        temp_input_cluster.loc[0, 'bedrooms'] = selected_bedrooms
        temp_input_cluster.loc[0, 'minimum_nights'] = min_nights
        temp_input_cluster.loc[0, 'instant_bookable'] = True
        temp_input_cluster.loc[0, 'host_is_superhost'] = False
        temp_input_cluster.loc[0, 'amenities_count'] = len(selected_amenities)
        temp_input_cluster.loc[0, 'neighbourhood'] = selected_neighbourhood
        temp_input_cluster.loc[0, 'room_type'] = selected_room_type
        temp_input_cluster.loc[0, 'property_grouped'] = selected_property_grouped
        for col in amenity_cols:
            temp_input_cluster.loc[0, col] = 1 if col in selected_amenities else 0
        temp_input_cluster['price'] = pred_price

        # รวมข้อมูลทั้งหมด
        combined_df = pd.concat([df_cluster, temp_input_cluster], ignore_index=True).fillna(0)

        # เตรียมข้อมูล Clustering
        cluster_pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')

        X_cluster = cluster_pre.fit_transform(combined_df)
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_cluster)

        # หา optimal k
        sse = []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            sse.append(km.inertia_)

        optimal_k = sse.index(min(sse)) + 2 if len(sse) >= 2 else 2
        

        # ทำ KMeans
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        combined_df['cluster'] = cluster_labels

        input_cluster_label = combined_df.iloc[-1]['cluster']
        df_cluster['cluster'] = cluster_labels[:len(df_cluster)]

        # หาคู่แข่ง
        competitors = df_cluster[df_cluster['cluster'] == input_cluster_label].copy()
        competitors['price'] = df.loc[competitors.index, 'price']
        competitors['name'] = df.loc[competitors.index, 'name']

        st.subheader("🏁 Competitors in the Same Cluster")
        st.write(f"🔍 Filtering competitors only in neighbourhood: `{selected_neighbourhood}`")
        st.info(f"🔢 Optimal number of clusters (K): {optimal_k}")
        st.dataframe(competitors[['name', 'neighbourhood', 'room_type', 'property_grouped', 'bedrooms', 'price']].reset_index(drop=True))
 
        # เพิ่มปุ่มสำหรับ Export ตาราง
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df_to_convert.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(competitors[['name', 'neighbourhood', 'room_type', 'property_grouped', 'bedrooms', 'price']].reset_index(drop=True))

        st.download_button(
            label="Download Competitors as CSV",
            data=csv,
            file_name='airbnb_competitors.csv',
            mime='text/csv',
        )