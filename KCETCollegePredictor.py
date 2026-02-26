import streamlit as st
import pandas as pd
import requests
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# =====================
# CONFIG & STYLE
# =====================
st.set_page_config(page_title="KCET Predictor Pro", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    .main-title {text-align:center; color:#0073e6; font-size:40px; font-weight:700;}
    .sub-title {text-align:center; color:#555; font-size:20px;}
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# USER AUTHENTICATION
# =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {"admin": {"password": "admin123", "cities": []}}

def signup_user(username, password):
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = {"password": password, "cities": []}
    return True

def login_user(username, password):
    user = st.session_state.users.get(username)
    if user and user["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = "🏠 Dashboard"
        return True
    return False

def logout_user():
    st.session_state.logged_in = False
    st.session_state.page = "🏠 Dashboard"

# =====================
# LOGIN / SIGNUP PAGE
# =====================
if not st.session_state.logged_in:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h2 class='main-title'>🎓 KCET College Predictor Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>AI + Cutoff Based Seat Prediction System</p>", unsafe_allow_html=True)
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])
        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login_user(username, password):
                    st.success(f"Welcome {username} 👋")
                    st.rerun()
                else:
                    st.error("Invalid username or password!")
        with tab2:
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if signup_user(new_user, new_pass):
                    st.success("Signup successful! Please log in.")
                else:
                    st.warning("Username already exists.")
    st.stop()

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    url = "https://github.com/VishnuSastryHK/KCETCollegePredictor/raw/master/CET_Database_Final2020.csv"
    content = requests.get(url).content
    return pd.read_csv(io.StringIO(content.decode('utf-8')))

df = load_data()

# =====================
# DASHBOARD NAVIGATION
# =====================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"

menu = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Dashboard", "📊 Cutoff Based", "🤖 ML Based", "👤 Profile", "ℹ️ About"],
    index=["🏠 Dashboard", "📊 Cutoff Based", "🤖 ML Based", "👤 Profile", "ℹ️ About"].index(st.session_state.page)
)
st.session_state.page = menu

st.sidebar.markdown("---")
st.sidebar.info(f"👋 Logged in as **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    logout_user()
    st.rerun()

# =====================
# DASHBOARD PAGE
# =====================
if menu == "🏠 Dashboard":
    st.markdown("<h2 class='main-title'>🎯 KCET Predictor Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Select a prediction mode from the sidebar to begin.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Colleges", df['College'].nunique())
    with col2:
        st.metric("Total Branches", df['Branch'].nunique())
    with col3:
        st.metric("Total Cities", df['Location'].nunique())

# =====================
# CUTOFF BASED PAGE
# =====================
elif menu == "📊 Cutoff Based":
    st.subheader("🎯 Cutoff-Based College Prediction")
    rank = st.number_input("Enter Your KCET Rank:", min_value=0, value=1000)
    branch_list = st.multiselect("Select Preferred Branches:", df['Branch'].unique())
    category = st.selectbox("Select Category:", df.columns[4:], index=15)
    preferred_cities = st.multiselect("Select Preferred Cities (optional):", df['Location'].unique())

    if st.button("🔍 Predict (Cutoff Based)"):
        seat_df = df[df['Branch'].isin(branch_list)].copy()
        if preferred_cities:
            seat_df = seat_df[seat_df['Location'].isin(preferred_cities)]
        seat_df['Cutoff'] = seat_df[category].astype(float)
        seat_df = seat_df[(seat_df['Cutoff'] != 0) & (rank < seat_df['Cutoff'])]

        if not seat_df.empty:
            st.success("✅ Colleges where you may get a seat:")
            st.dataframe(seat_df[['Branch', 'College', 'Location', 'CETCode', 'Cutoff']].sort_values('Cutoff'))
        else:
            st.warning("No matches found for the given rank and filters.")

    if st.button("🔙 Back to Dashboard"):
        st.session_state.page = "🏠 Dashboard"
        st.rerun()

# =====================
# ML BASED PAGE
# =====================
elif menu == "🤖 ML Based":
    st.subheader("🤖 ML-Based Admission Prediction")
    rank = st.number_input("Enter Your KCET Rank:", min_value=0, value=1000)
    branch_list = st.multiselect("Select Preferred Branches:", df['Branch'].unique())
    category = st.selectbox("Select Category:", df.columns[4:], index=15)
    college_list = st.multiselect("Preferred Colleges (optional):", df['College'].unique())
    preferred_cities = st.multiselect("Preferred Cities (optional):", df['Location'].unique())

    if st.button("🚀 Predict (ML Based)"):
        df_long = df.melt(
            id_vars=['College', 'Branch', 'Location', 'CETCode'],
            var_name='Category',
            value_name='Cutoff'
        )
        df_long['Admit'] = (df_long['Cutoff'] > 0) & (rank <= df_long['Cutoff'])
        df_long = df_long.dropna(subset=['College', 'Branch', 'Category', 'Cutoff'])

        features = ['College', 'Branch', 'Category', 'Cutoff']
        target = 'Admit'
        categorical_features = ['College', 'Branch', 'Category']

        ct = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )
        X = ct.fit_transform(df_long[features])
        y = df_long[target].astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        user_df = []
        for branch in branch_list:
            for col in (college_list if college_list else df['College'].unique()):
                user_df.append({
                    'College': col,
                    'Branch': branch,
                    'Category': category,
                    'Cutoff': rank
                })
        user_df = pd.DataFrame(user_df)

        if preferred_cities:
            user_df = user_df[user_df['College'].isin(
                df[df['Location'].isin(preferred_cities)]['College'].unique()
            )]

        user_X = ct.transform(user_df)
        user_df['Admission_Probability'] = model.predict_proba(user_X)[:, 1]

        st.success("✅ ML Prediction Results:")
        st.dataframe(user_df[['College', 'Branch', 'Category', 'Admission_Probability']].sort_values(
            by="Admission_Probability", ascending=False
        ))

    if st.button("🔙 Back to Dashboard"):
        st.session_state.page = "🏠 Dashboard"
        st.rerun()

# =====================
# PROFILE PAGE
# =====================
elif menu == "👤 Profile":
    st.subheader("👤 User Profile")
    user = st.session_state.users[st.session_state.username]
    st.write(f"**Username:** {st.session_state.username}")

    cities = st.multiselect("Preferred Cities:", df['Location'].unique(), default=user["cities"])
    if st.button("💾 Save Preferences"):
        st.session_state.users[st.session_state.username]["cities"] = cities
        st.success("✅ Preferences saved!")

    st.write("Your saved preferred cities:")
    if user["cities"]:
        st.write(", ".join(user["cities"]))
    else:
        st.info("No preferred cities saved yet.")

    if st.button("🔙 Back to Dashboard"):
        st.session_state.page = "🏠 Dashboard"
        st.rerun()

# =====================
# ABOUT PAGE
# =====================
elif menu == "ℹ️ About":
    st.markdown("""
    ### 📘 About KCET Predictor Pro
    - Built with **Streamlit + Scikit-learn**
    - Supports both **Cutoff-based** and **ML-based** predictions  
    - Includes **user profiles with city preferences**
    - Secure local **login system**
    - Smooth navigation with **Back to Dashboard** button  
    - Designed for Karnataka CET aspirants 🎓
    """)
