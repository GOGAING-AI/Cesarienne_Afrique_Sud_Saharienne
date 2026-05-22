# =============================================================================
# APPLICATION STREAMLIT — PRÉDICTION CÉSARIENNE D'URGENCE EN AFRIQUE
# =============================================================================
# Projet : Projet ML — Dataset DHS Afrique Subsaharienne
# Auteur : Gogaing Foguen Baudouin Legrand
# Niveau: Master 2 Recherche
# Ecole: Faculté des Sciences de Université de Douala
# Version : 1.0
#
# LANCEMENT :
#   streamlit run app_cesarienne.py
#
# DÉPENDANCES :
#   pour pouvoir bien mener ce projet veillez installer dans votre espace python les bibliothèques suivante:
#   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
#   pip install imbalanced-learn xgboost geopandas joblib plotly
#
# FICHIERS REQUIS dans le même dossier :
#     app_cesarienne.py          <-- ce fichier
#     ccs_ssa_fv_clean.csv       <-- dataset
#     modele_cesarienne_xgb.pkl  <-- modèle entraîné (optionnel*)
#     scaler_cesarienne.pkl      <-- scaler (optionnel*)
#     ne_110m_admin_0_countries/ <-- shapefile Natural Earth (optionnel**)
#
# * Si le modèle pkl n'existe pas, l'app l'entraîne automatiquement au démarrage
# ** Si le shapefile est absent, la carte utilisera plotly choropleth (online)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GÉNÉRALE DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CésaPredict — Afrique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé : thème médical professionnel ──────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #2980b9;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 5px solid #c0392b;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    .badge-danger {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 18px 28px;
        border-radius: 16px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 20px rgba(192,57,43,0.4);
    }
    .badge-safe {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 18px 28px;
        border-radius: 16px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 20px rgba(39,174,96,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES & CHEMINS
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH      = "ccs_ssa_fv_clean.csv"
MODEL_PATH     = "modele_cesarienne_xgb.pkl"
SCALER_PATH    = "scaler_cesarienne.pkl"
LE_PATH        = "le_v025.pkl"
COLS_PATH      = "model_columns.pkl"
SHAPEFILE_PATH = "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"

# Ordre pour les encodages ordinaux
ORDINAL_V106 = ['no education', 'primary', 'secondary', 'higher']
ORDINAL_V190 = ['poorest', 'poorer', 'middle', 'richer', 'richest']

# Lieux valides (filtrés, sans codes numériques)
LIEUX_VALIDES = [
    "respondent's home", "other home",
    "health center/post", "maternity",
    "provincial hospital", "central hospital",
    "municipal hospital", "private hospital/clinic",
    "medical center", "other public sector",
    "other private sector", "other"
]

# 21 pays du dataset
COUNTRIES = [
    'Angola', 'Benin', 'Burundi', 'Cameroon', 'Ethiopia', 'Gambia',
    'Guinea', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
    'Nigeria', 'Rwanda', 'Senegal', 'Sierra Leone', 'South Africa',
    'Tanzania', 'Uganda', 'Zambia', 'Zimbabwe'
]

# Correspondance noms DHS → noms Natural Earth (pour le shapefile)
COUNTRY_NAME_MAP = {
    "Tanzania": "United Republic of Tanzania",
}


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Chargement et nettoyage initial des données brutes."""
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    # Filtrage des modalités parasites (codes numériques dans m15 et v106)
    df = df[df['m15'].isin(LIEUX_VALIDES)]
    df = df[df['v106'].isin(ORDINAL_V106)]
    # Variable cible binaire pour les calculs d'exploration
    df['emergency'] = (df['m17'] == 'emergency cs').astype(int)
    return df


def preprocess_for_model(df_input, scaler, le_v025, model_cols):
    """
    Applique le même pipeline de prétraitement qu'à l'entraînement.
    Entrée : DataFrame avec colonnes brutes.
    Sortie : DataFrame aligné prêt pour model.predict().
    """
    df_p = df_input.copy()

    # 1. Encodage binaire v025 (rural / urban)
    df_p['v025'] = le_v025.transform(df_p['v025'])

    # 2. Encodage ordinal v106 (niveau d'éducation : 0→3)
    df_p['v106'] = df_p['v106'].map({v: i for i, v in enumerate(ORDINAL_V106)})

    # 3. Encodage ordinal v190 (richesse : 0→4)
    df_p['v190'] = df_p['v190'].map({v: i for i, v in enumerate(ORDINAL_V190)})

    # 4. Clamping pour corriger les valeurs aberrantes
    df_p['m13'] = df_p['m13'].clip(0, 12)
    df_p['m14'] = df_p['m14'].clip(1, 9)
    df_p['m19'] = df_p['m19'].clip(300, 6000)

    # 5. One-Hot Encoding des variables nominales
    ohe_cols = ['m15', 'm2n', 'm3n', 'b0', 'country']
    df_p = pd.get_dummies(df_p, columns=ohe_cols, drop_first=True)

    # 6. Alignement des colonnes avec celles du modèle entraîné
    for col in model_cols:
        if col not in df_p.columns:
            df_p[col] = 0
    df_p = df_p[model_cols]

    # 7. Normalisation des variables numériques continues
    num_cols = ['m13', 'm14', 'm19']
    df_p[num_cols] = scaler.transform(df_p[num_cols])

    return df_p


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """
    Charge le modèle s'il existe, sinon l'entraîne depuis le dataset.
    Retourne : (model, scaler, le_v025, model_columns)
    """
    if (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
            and os.path.exists(LE_PATH) and os.path.exists(COLS_PATH)):
        model      = joblib.load(MODEL_PATH)
        scaler     = joblib.load(SCALER_PATH)
        le_v025    = joblib.load(LE_PATH)
        model_cols = joblib.load(COLS_PATH)
        return model, scaler, le_v025, model_cols

    # ── Entraînement automatique ──────────────────────────────────────────
    st.info("  Modèle introuvable — entraînement en cours (~2 min)...")

    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df = df[df['m15'].isin(LIEUX_VALIDES)]
    df = df[df['v106'].isin(ORDINAL_V106)]

    df['target'] = (df['m17'] == 'emergency cs').astype(int)
    df.drop(columns=['m17'], inplace=True)

    # Encodages
    le_v025 = LabelEncoder()
    df['v025'] = le_v025.fit_transform(df['v025'])
    df['v106'] = df['v106'].map({v: i for i, v in enumerate(ORDINAL_V106)})
    df['v190'] = df['v190'].map({v: i for i, v in enumerate(ORDINAL_V190)})
    df['m13']  = df['m13'].clip(0, 12)
    df['m14']  = df['m14'].clip(1, 9)
    df['m19']  = df['m19'].clip(300, 6000)

    df = pd.get_dummies(df, columns=['m15', 'm2n', 'm3n', 'b0', 'country'], drop_first=True)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train[['m13', 'm14', 'm19']] = scaler.fit_transform(X_train[['m13', 'm14', 'm19']])

    # SMOTE pour équilibrer les classes (ratio 34:1)
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    # XGBoost avec scale_pos_weight pour compenser le déséquilibre résiduel
    scale_w = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_w,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    model.fit(X_sm, y_sm)

    model_cols = list(X.columns)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(le_v025,    LE_PATH)
    joblib.dump(model_cols, COLS_PATH)

    return model, scaler, le_v025, model_cols


def make_africa_map_plotly(country_stats):
    """
    Carte choroplèthe interactive de l'Afrique avec Plotly.
    Ne nécessite aucun fichier shapefile local.
    """
    fig = px.choropleth(
        country_stats,
        locations="country",
        locationmode="country names",
        color="rate_pct",
        color_continuous_scale="Reds",
        scope="africa",
        hover_name="country",
        hover_data={"rate_pct": ":.2f", "total": True, "emergency": True},
        labels={"rate_pct": "Taux urgence (%)", "total": "Total", "emergency": "Urgences"},
        title="Taux de Césarienne d'Urgence par Pays — Afrique Subsaharienne",
    )
    fig.update_layout(
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor="#555",
            showland=True, landcolor="#f0f0e8",
            showocean=True, oceancolor="#c8e6f5",
            showcountries=True, countrycolor="#999",
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Taux (%)", ticksuffix="%", len=0.7, thickness=15),
        height=560,
        font=dict(family="Georgia, serif", size=13)
    )
    return fig


def make_africa_map_geopandas(country_stats):
    """
    Carte haute résolution avec GeoPandas + Matplotlib.
    Nécessite le shapefile Natural Earth dans SHAPEFILE_PATH.
    """
    import geopandas as gpd
    world  = gpd.read_file(SHAPEFILE_PATH)
    africa = world[world['CONTINENT'] == 'Africa'].copy()

    # Normalisation des noms de pays pour le merge
    cs_norm = country_stats.copy()
    cs_norm['country_norm'] = cs_norm['country'].replace(COUNTRY_NAME_MAP)

    africa = africa.merge(
        cs_norm[['country_norm', 'rate_pct']],
        how='left', left_on='NAME', right_on='country_norm'
    )

    fig, ax = plt.subplots(figsize=(12, 9))
    africa.plot(
        column='rate_pct', ax=ax, legend=True,
        cmap='Reds',
        missing_kwds={"color": "#d0d0d0", "label": "Non disponible"},
        legend_kwds={"label": "Taux urgence (%)", "shrink": 0.6, "format": "%.1f%%"}
    )
    ax.set_title("Taux de Césarienne d'Urgence — Afrique Subsaharienne",
                 fontsize=15, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CAS DE TESTS PRÉDÉFINIS
# Chaque cas représente un profil clinique contrasté pour valider le modèle
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "🔴 Risque très élevé — Rwanda, rural, sans soins",
        "description": "Mère rurale sans éducation, aucune CPN, accouchement à domicile, bébé très léger.",
        "expected": "Urgence",
        "params": {
            "v025": "rural", "v106": "no education", "v190": "poorest",
            "m13": 0, "m14": 9, "m19": 1800,
            "m15": "respondent's home", "m2n": "no/somecare",
            "m3n": "yes: no assistance", "b0": "single birth", "country": "Rwanda"
        }
    },
    {
        "label": "🔴 Risque élevé — Afrique du Sud, naissance multiple",
        "description": "Naissance gémellaire (1er jumeau), CPN insuffisante, poids faible.",
        "expected": "Urgence",
        "params": {
            "v025": "urban", "v106": "primary", "v190": "poorer",
            "m13": 2, "m14": 8, "m19": 2200,
            "m15": "provincial hospital", "m2n": "no/somecare",
            "m3n": "no: some assistance", "b0": "1st of multiple", "country": "South Africa"
        }
    },
    {
        "label": "🟡 Risque modéré — Cameroun, CPN tardive",
        "description": "Mère urbaine avec éducation primaire, CPN tardive, poids limite.",
        "expected": "Intermédiaire",
        "params": {
            "v025": "urban", "v106": "primary", "v190": "middle",
            "m13": 3, "m14": 6, "m19": 2500,
            "m15": "health center/post", "m2n": "yes/nocare",
            "m3n": "no: some assistance", "b0": "single birth", "country": "Cameroon"
        }
    },
    {
        "label": "🟡 Risque modéré — Malawi, zone rurale",
        "description": "Mère avec éducation primaire, CPN insuffisante, centre de santé.",
        "expected": "Intermédiaire",
        "params": {
            "v025": "rural", "v106": "primary", "v190": "poorer",
            "m13": 4, "m14": 5, "m19": 2900,
            "m15": "health center/post", "m2n": "yes/nocare",
            "m3n": "no: some assistance", "b0": "single birth", "country": "Malawi"
        }
    },
    {
        "label": "🟢 Risque faible — Nigeria, mère éduquée et suivie",
        "description": "Mère urbaine diplômée, suivi complet, hôpital privé, poids normal.",
        "expected": "Normale",
        "params": {
            "v025": "urban", "v106": "higher", "v190": "richest",
            "m13": 8, "m14": 2, "m19": 3400,
            "m15": "private hospital/clinic", "m2n": "yes/nocare",
            "m3n": "no: some assistance", "b0": "single birth", "country": "Nigeria"
        }
    },
    {
        "label": "🟢 Risque très faible — Sénégal, conditions optimales",
        "description": "Suivi excellent, maternité, poids normal, zone urbaine aisée.",
        "expected": "Normale",
        "params": {
            "v025": "urban", "v106": "secondary", "v190": "richer",
            "m13": 10, "m14": 1, "m19": 3600,
            "m15": "maternity", "m2n": "yes/nocare",
            "m3n": "no: some assistance", "b0": "single birth", "country": "Senegal"
        }
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ET DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("⏳ Chargement des données..."):
    df = load_data()

with st.spinner("🤖 Chargement du modèle..."):
    model, scaler, le_v025, model_cols = load_or_train_model()

# Statistiques globales par pays
country_stats = df.groupby('country').agg(
    total=('emergency', 'count'),
    emergency=('emergency', 'sum')
).reset_index()
country_stats['rate']     = country_stats['emergency'] / country_stats['total']
country_stats['rate_pct'] = (country_stats['rate'] * 100).round(2)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #c0392b 100%);
            padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);'>
    <h1 style='color: white; margin: 0; font-size: 2.2rem; font-family: Georgia, serif;'>
         CésaPredict — Afrique Subsaharienne
    </h1>
    <p style='color: #f5c6c6; margin: 8px 0 0 0; font-size: 1.05rem;'>
        Prédiction &amp; Analyse de la Césarienne d'Urgence · 21 pays · 238 281 accouchements
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SAISIE PATIENT
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style='text-align:center; margin-bottom:16px;'>
    <span style='font-size:2.5rem;'>🩺</span>
    <h2 style='color:white; margin:4px 0; font-family: Georgia, serif;'>Saisie Patient</h2>
    <p style='color:#a8d8ea; font-size:0.85rem;'>Paramètres pour la prédiction</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**📍 Contexte géographique & socio-économique**")
pays     = st.sidebar.selectbox("🌍 Pays", COUNTRIES, index=12)
milieu   = st.sidebar.selectbox("🏘️ Milieu", ["rural", "urban"])
educ     = st.sidebar.selectbox("🎓 Éducation", ORDINAL_V106)
richesse = st.sidebar.selectbox("💰 Richesse", ORDINAL_V190)

st.sidebar.markdown("---")
st.sidebar.markdown("**🤰 Suivi prénatal**")
nb_cpn    = st.sidebar.slider("📋 Nb de CPN", 0, 12, 4)
mois_cpn  = st.sidebar.slider("📅 Mois 1re CPN", 1, 9, 4,
                               help="1 = très précoce · 9 = très tardif")
soins_pre = st.sidebar.selectbox("💊 Soins prénataux", ["yes/nocare", "no/somecare"])

st.sidebar.markdown("---")
st.sidebar.markdown("**🏥 Accouchement**")
lieu       = st.sidebar.selectbox("📍 Lieu", LIEUX_VALIDES)
assistance = st.sidebar.selectbox("👩‍⚕️ Assistance",
                                   ["no: some assistance", "yes: no assistance"])
naissance  = st.sidebar.selectbox("👶 Type de naissance",
                                   ["single birth", "1st of multiple",
                                    "2nd of multiple", "3rd of multiple"])
poids      = st.sidebar.number_input("⚖️ Poids (g)", 300, 6000, 3100, 50)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮  LANCER LA PRÉDICTION",
                                  use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tableau de bord",
    "🗺️  Carte Afrique",
    "🔬 Analyses détaillées",
    "🔮 Prédiction",
    "🧪 Cas de tests"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — TABLEAU DE BORD
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-title">📈 Vue globale du dataset</div>',
                unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────
    total_obs = len(df)
    n_urgence = df['emergency'].sum()
    n_normale = total_obs - n_urgence
    taux_urg  = n_urgence / total_obs * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔢 Total", f"{total_obs:,}")
    c2.metric("✅ Normales", f"{n_normale:,}", f"{100-taux_urg:.1f}%")
    c3.metric("⚠️ Urgences", f"{n_urgence:,}", f"{taux_urg:.1f}%", delta_color="inverse")
    c4.metric("🌍 Pays", df['country'].nunique())
    c5.metric("⚡ Ratio déséquilibre", f"{n_normale/n_urgence:.0f}:1")

    st.markdown("---")

    # ── Barplot taux par pays ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Taux d\'urgence par pays</div>',
                unsafe_allow_html=True)

    cs_sorted = country_stats.sort_values('rate_pct', ascending=False)
    fig_bar = px.bar(cs_sorted, x='country', y='rate_pct',
                     color='rate_pct', color_continuous_scale='Reds',
                     text='rate_pct', height=400,
                     labels={'country': 'Pays', 'rate_pct': 'Taux urgence (%)'})
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          marker_line_color='white')
    fig_bar.update_layout(xaxis_tickangle=-45, showlegend=False,
                          coloraxis_showscale=False,
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Distribution cible + stats ────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-title">Distribution de la cible</div>',
                    unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[n_normale, n_urgence],
            names=['Normale', 'Urgence'],
            color_discrete_sequence=['#27ae60', '#e74c3c'],
            hole=0.45, height=320
        )
        fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.05])
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Statistiques descriptives</div>',
                    unsafe_allow_html=True)
        num_stats = df[['m13', 'm14', 'm19', 'emergency']].rename(columns={
            'm13': 'Nb CPN', 'm14': 'Mois 1re CPN',
            'm19': 'Poids (g)', 'emergency': 'Urgence'
        }).describe().T.round(2)
        st.dataframe(num_stats, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CARTE AFRIQUE
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-title">🗺️ Carte choroplèthe — Afrique Subsaharienne</div>',
                unsafe_allow_html=True)

    # Détection automatique GeoPandas vs Plotly
    geo_method = "plotly"
    if os.path.exists(SHAPEFILE_PATH):
        try:
            import geopandas as gpd
            geo_method = "geopandas"
            st.success("✅ Shapefile détecté — carte haute résolution (GeoPandas)")
        except ImportError:
            st.info("ℹ️  GeoPandas non installé — carte Plotly utilisée")
    else:
        st.info(
            "ℹ️  Shapefile absent → carte Plotly (interactive). "
            "Pour la version haute résolution, placez `ne_110m_admin_0_countries/` "
            "dans le même dossier."
        )

    if geo_method == "geopandas":
        fig_geo = make_africa_map_geopandas(country_stats)
        st.pyplot(fig_geo)
    else:
        fig_geo = make_africa_map_plotly(country_stats)
        st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown("---")

    # ── Tableau détaillé par pays ──────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Détail par pays</div>',
                unsafe_allow_html=True)

    def risk_badge(rate):
        if rate >= 7:    return "🔴 Très élevé"
        if rate >= 4:    return "🟠 Élevé"
        if rate >= 2.5:  return "🟡 Modéré"
        return "🟢 Faible"

    cs_display = country_stats[['country', 'total', 'emergency', 'rate_pct']].copy()
    cs_display.columns = ['Pays', 'Total cas', 'Urgences', 'Taux urgence (%)']
    cs_display['Niveau de risque'] = cs_display['Taux urgence (%)'].apply(risk_badge)
    cs_display = cs_display.sort_values('Taux urgence (%)', ascending=False).reset_index(drop=True)
    cs_display.index += 1
    st.dataframe(cs_display, use_container_width=True, height=540)

    # ── Graphiques complémentaires ─────────────────────────────────────────
    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Top 5 vs Bottom 5
        top5   = country_stats.nlargest(5, 'rate_pct').assign(groupe='Top 5 🔴')
        bot5   = country_stats.nsmallest(5, 'rate_pct').assign(groupe='Bottom 5 🟢')
        concat = pd.concat([top5, bot5])
        fig_tb = px.bar(concat, x='rate_pct', y='country', orientation='h',
                        color='groupe',
                        color_discrete_map={'Top 5 🔴': '#e74c3c', 'Bottom 5 🟢': '#27ae60'},
                        title="Top 5 / Bottom 5 pays",
                        labels={'rate_pct': 'Taux (%)', 'country': ''}, height=360)
        fig_tb.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_tb, use_container_width=True)

    with col_g2:
        # Volume vs taux
        fig_sc = px.scatter(country_stats, x='total', y='rate_pct',
                            size='emergency', color='rate_pct', hover_name='country',
                            color_continuous_scale='RdYlGn_r',
                            title="Volume vs Taux d'urgence",
                            labels={'total': 'Total cas', 'rate_pct': 'Taux (%)'}, height=360)
        fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSES DÉTAILLÉES
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-title">🔬 Analyses bivariées & multivariées</div>',
                unsafe_allow_html=True)

    analyse = st.selectbox("Choisir une analyse :", [
        "Distribution des variables numériques",
        "Taux d'urgence par variable catégorielle",
        "Poids de naissance vs Urgence",
        "Consultations prénatales vs Urgence",
        "Lieu d'accouchement vs Urgence",
        "Richesse × Éducation (heatmap)",
        "Corrélation des variables"
    ])

    if analyse == "Distribution des variables numériques":
        col_v = st.selectbox("Variable", ['m13 (Nb CPN)', 'm14 (Mois 1re CPN)', 'm19 (Poids g)'])
        var_code = col_v.split(" ")[0]
        label_map = {'m13': 'Nombre de consultations prénatales',
                     'm14': 'Mois de la 1ère CPN', 'm19': 'Poids de naissance (g)'}
        fig_h = px.histogram(
            df.sample(min(50000, len(df)), random_state=42),
            x=var_code, color='m17', barmode='overlay', opacity=0.7, nbins=50,
            color_discrete_map={'normal/elective cs': '#27ae60', 'emergency cs': '#e74c3c'},
            title=f"Distribution de {label_map[var_code]} selon le type de césarienne",
            labels={var_code: label_map[var_code], 'm17': 'Type'}, height=420
        )
        fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_h, use_container_width=True)

    elif analyse == "Taux d'urgence par variable catégorielle":
        cat_labels = {'v025': 'Milieu de résidence', 'v106': "Niveau d'éducation",
                      'v190': 'Richesse', 'm2n': 'Soins prénataux',
                      'm3n': 'Assistance', 'b0': 'Type de naissance'}
        cat_var = st.selectbox("Variable", list(cat_labels.keys()))
        rate_cat = (df.groupby(cat_var)['emergency'].mean() * 100).reset_index()
        rate_cat.columns = [cat_var, 'taux']
        fig_cat = px.bar(rate_cat.sort_values('taux', ascending=False),
                         x=cat_var, y='taux', color='taux',
                         color_continuous_scale='Reds', text='taux',
                         title=f"Taux urgence par {cat_labels[cat_var]}",
                         labels={cat_var: cat_labels[cat_var], 'taux': 'Taux (%)'}, height=400)
        fig_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_cat.update_layout(coloraxis_showscale=False,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cat, use_container_width=True)

    elif analyse == "Poids de naissance vs Urgence":
        fig_box = px.box(
            df.sample(min(30000, len(df)), random_state=42),
            x='m17', y='m19', color='m17', notched=True,
            color_discrete_map={'normal/elective cs': '#27ae60', 'emergency cs': '#e74c3c'},
            title="Distribution du poids de naissance selon le type de césarienne",
            labels={'m17': 'Type', 'm19': 'Poids (g)'}, height=430
        )
        fig_box.update_layout(showlegend=False,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

    elif analyse == "Consultations prénatales vs Urgence":
        cpn_rate  = (df.groupby('m13')['emergency'].mean() * 100).reset_index()
        cpn_count = df.groupby('m13').size().reset_index(name='count')
        cpn_data  = cpn_rate.merge(cpn_count, on='m13')
        cpn_data.columns = ['nb_cpn', 'taux', 'count']
        fig_cpn = px.scatter(cpn_data, x='nb_cpn', y='taux', size='count',
                              color='taux', color_continuous_scale='RdYlGn_r',
                              trendline='lowess',
                              title="Taux d'urgence selon le nombre de CPN",
                              labels={'nb_cpn': 'Nb CPN', 'taux': 'Taux (%)'}, height=420)
        fig_cpn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cpn, use_container_width=True)

    elif analyse == "Lieu d'accouchement vs Urgence":
        lieu_rate = (df.groupby('m15')['emergency'].mean() * 100).reset_index()
        lieu_rate.columns = ['lieu', 'taux']
        fig_lieu = px.bar(lieu_rate.sort_values('taux'), x='taux', y='lieu',
                          orientation='h', color='taux', color_continuous_scale='Reds',
                          text='taux',
                          title="Taux urgence selon le lieu d'accouchement",
                          labels={'lieu': 'Lieu', 'taux': 'Taux (%)'}, height=430)
        fig_lieu.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_lieu.update_layout(coloraxis_showscale=False,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lieu, use_container_width=True)

    elif analyse == "Richesse × Éducation (heatmap)":
        pivot = df.groupby(['v190', 'v106'])['emergency'].mean().unstack() * 100
        ord_190 = [c for c in ORDINAL_V190 if c in pivot.index]
        ord_106 = [c for c in ORDINAL_V106 if c in pivot.columns]
        pivot = pivot.loc[ord_190, ord_106]
        fig_hm = px.imshow(pivot, text_auto='.1f', color_continuous_scale='Reds',
                           title="Taux urgence (%) : Richesse × Éducation",
                           labels=dict(x="Éducation", y="Richesse", color="Taux (%)"),
                           height=380)
        fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hm, use_container_width=True)

    elif analyse == "Corrélation des variables":
        num_df = df[['m13', 'm14', 'm19', 'emergency']].rename(columns={
            'm13': 'Nb CPN', 'm14': 'Mois 1re CPN', 'm19': 'Poids (g)', 'emergency': 'Urgence'
        })
        fig_corr = px.imshow(num_df.corr(), text_auto='.3f',
                             color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
                             title="Matrice de corrélation", height=380)
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRÉDICTION INDIVIDUELLE
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-title">🔮 Prédiction individualisée</div>',
                unsafe_allow_html=True)

    # Résumé des paramètres saisis dans la sidebar
    with st.expander("📋 Paramètres saisis", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.write(f"🌍 **Pays :** {pays}")
            st.write(f"🏘️ **Milieu :** {milieu}")
            st.write(f"🎓 **Éducation :** {educ}")
            st.write(f"💰 **Richesse :** {richesse}")
        with pc2:
            st.write(f"📋 **Nb CPN :** {nb_cpn}")
            st.write(f"📅 **Mois 1re CPN :** {mois_cpn}")
            st.write(f"💊 **Soins pré :** {soins_pre}")
        with pc3:
            st.write(f"📍 **Lieu :** {lieu}")
            st.write(f"👩‍⚕️ **Assistance :** {assistance}")
            st.write(f"👶 **Naissance :** {naissance}")
            st.write(f"⚖️ **Poids :** {poids} g")

    if predict_btn:
        # Construire le DataFrame patient et prédire
        patient = pd.DataFrame([{
            'v025': milieu, 'v106': educ, 'v190': richesse,
            'm13': nb_cpn, 'm14': mois_cpn, 'm19': poids,
            'm15': lieu,   'm2n': soins_pre, 'm3n': assistance,
            'b0': naissance, 'country': pays
        }])

        with st.spinner("🔄 Calcul en cours..."):
            time.sleep(0.4)
            patient_proc = preprocess_for_model(patient, scaler, le_v025, model_cols)
            pred  = model.predict(patient_proc)[0]
            proba = model.predict_proba(patient_proc)[0]
            prob_urgence = proba[1] * 100
            prob_normale = proba[0] * 100

        st.markdown("---")
        st.subheader("🎯 Résultat de la prédiction")

        # Jauge de probabilité
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_urgence,
            title={'text': "Probabilité de Césarienne d'Urgence",
                   'font': {'size': 16, 'family': 'Georgia, serif'}},
            delta={'reference': 2.86,
                   'increasing': {'color': '#e74c3c'},
                   'decreasing': {'color': '#27ae60'}},
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if pred == 1 else "#27ae60", 'thickness': 0.3},
                'steps': [
                    {'range': [0, 5],    'color': '#d5f5e3'},
                    {'range': [5, 20],   'color': '#fdebd0'},
                    {'range': [20, 50],  'color': '#fadbd8'},
                    {'range': [50, 100], 'color': '#f1948a'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': 2.86   # taux moyen du dataset = référence
                }
            }
        ))
        fig_gauge.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                                 font=dict(family="Georgia, serif"))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Verdict coloré
        if pred == 1:
            st.markdown(f"""
            <div class="badge-danger">
                ⚠️ RISQUE DE CÉSARIENNE D'URGENCE<br>
                <span style='font-size:1rem; font-weight:400;'>
                    Probabilité urgence : {prob_urgence:.1f}%  ·  Normale : {prob_normale:.1f}%
                </span>
            </div>
            <div style='background:#fdf2f8; border:1px solid #e74c3c; border-radius:10px;
                        padding:16px; margin-top:14px;'>
                <b style='color:#c0392b;'>📌 Recommandations cliniques :</b>
                <ul style='color:#444; margin-top:8px;'>
                    <li>Orienter vers un centre de référence obstétricale</li>
                    <li>Monitoring continu du rythme cardiaque fœtal</li>
                    <li>Préparer le bloc opératoire et l'équipe chirurgicale</li>
                    <li>Obtenir le consentement éclairé de la patiente</li>
                    <li>Vérifier la disponibilité de produits sanguins</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="badge-safe">
                ✅ CÉSARIENNE NORMALE / PROGRAMMÉE<br>
                <span style='font-size:1rem; font-weight:400;'>
                    Probabilité urgence : {prob_urgence:.1f}%  ·  Normale : {prob_normale:.1f}%
                </span>
            </div>
            <div style='background:#f0fdf4; border:1px solid #27ae60; border-radius:10px;
                        padding:16px; margin-top:14px;'>
                <b style='color:#27ae60;'>📌 Suivi recommandé :</b>
                <ul style='color:#444; margin-top:8px;'>
                    <li>Poursuivre le suivi prénatal standard</li>
                    <li>Programmer la césarienne selon le protocole habituel</li>
                    <li>Maintenir la surveillance du poids fœtal</li>
                    <li>Préparer le plan d'accouchement avec la patiente</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Comparaison avec la moyenne du pays sélectionné
        taux_ref_row = country_stats[country_stats['country'] == pays]
        if not taux_ref_row.empty:
            taux_ref = taux_ref_row['rate_pct'].values[0]
            delta    = prob_urgence - taux_ref
            icon     = "🔴" if delta > 0 else "🟢"
            st.markdown(f"""
            ---
            **📊 Comparaison avec la moyenne nationale de {pays} :**  
            Probabilité individuelle : **{prob_urgence:.1f}%**  
            Taux moyen {pays} : **{taux_ref:.2f}%**  
            Écart : **{"+" if delta >= 0 else ""}{delta:.1f} points** {icon}
            """)
    else:
        st.info("👈  Renseignez les paramètres dans la **sidebar** puis cliquez sur "
                "**LANCER LA PRÉDICTION**.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — CAS DE TESTS
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-title">🧪 Cas de tests du modèle</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Six profils contrastés pour valider le comportement du modèle, "
        "du risque **très élevé** au risque **très faible**."
    )
    st.markdown("---")

    run_all = st.button("▶️  Exécuter tous les tests", type="primary")

    # Exécution et affichage de chaque test
    for i, tc in enumerate(TEST_CASES):
        with st.expander(f"**{tc['label']}**", expanded=(i == 0)):
            col_info, col_params, col_res = st.columns([2, 2, 1.5])

            with col_info:
                st.markdown(f"**Description :** {tc['description']}")
                st.markdown(f"**Résultat attendu :** `{tc['expected']}`")

            with col_params:
                p = tc['params']
                st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| Pays | {p['country']} |
| Milieu | {p['v025']} |
| Éducation | {p['v106']} |
| Richesse | {p['v190']} |
| Nb CPN | {p['m13']} |
| Mois 1re CPN | {p['m14']} |
| Poids (g) | {p['m19']} |
| Lieu | {p['m15']} |
| Soins pré | {p['m2n']} |
| Assistance | {p['m3n']} |
| Type naissance | {p['b0']} |
                """)

            with col_res:
                try:
                    pt = pd.DataFrame([p])
                    pt_proc = preprocess_for_model(pt, scaler, le_v025, model_cols)
                    pred_t  = model.predict(pt_proc)[0]
                    proba_t = model.predict_proba(pt_proc)[0]
                    p_urg   = proba_t[1] * 100
                    p_nor   = proba_t[0] * 100

                    label_p  = "⚠️ URGENCE"  if pred_t == 1 else "✅ NORMALE"
                    c_bg     = "#fdf2f8"      if pred_t == 1 else "#f0fdf4"
                    c_brd    = "#e74c3c"      if pred_t == 1 else "#27ae60"
                    c_txt    = "#c0392b"      if pred_t == 1 else "#27ae60"

                    st.markdown(f"""
<div style='background:{c_bg}; border:2px solid {c_brd};
            border-radius:12px; padding:16px; text-align:center;'>
    <div style='font-size:1.4rem; font-weight:700; color:{c_txt};'>{label_p}</div>
    <div style='margin-top:8px; color:#555; font-size:0.9rem;'>
        Urgence : <b>{p_urg:.1f}%</b><br>Normale : <b>{p_nor:.1f}%</b>
    </div>
</div>
                    """, unsafe_allow_html=True)

                    # Concordance attendu / prédit
                    exp_code = 1 if tc['expected'] == "Urgence" else 0
                    if tc['expected'] == "Intermédiaire":
                        match_txt, match_icon = "Cas ambigu", "🟡"
                    elif pred_t == exp_code:
                        match_txt, match_icon = "Concordance !", "✅"
                    else:
                        match_txt, match_icon = "Discordance", "⚠️"

                    st.markdown(f"""
<div style='text-align:center; margin-top:8px; font-size:0.85rem; color:#666;'>
    {match_icon} {match_txt}
</div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erreur : {e}")

    # ── Tableau récapitulatif de tous les tests ────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Récapitulatif des 6 tests</div>',
                unsafe_allow_html=True)

    rows = []
    for tc in TEST_CASES:
        try:
            pt = pd.DataFrame([tc['params']])
            pt_proc = preprocess_for_model(pt, scaler, le_v025, model_cols)
            pred_t  = model.predict(pt_proc)[0]
            proba_t = model.predict_proba(pt_proc)[0]
            rows.append({
                "Profil": tc['label'].split("—")[1].strip()[:40],
                "Attendu": tc['expected'],
                "Prédit": "Urgence" if pred_t == 1 else "Normale",
                "Prob. urgence (%)": round(proba_t[1] * 100, 1),
                "Prob. normale (%)": round(proba_t[0] * 100, 1),
            })
        except Exception as ex:
            rows.append({"Profil": tc['label'], "Attendu": tc['expected'],
                         "Prédit": f"Erreur: {ex}", "Prob. urgence (%)": 0, "Prob. normale (%)": 0})

    df_res = pd.DataFrame(rows)
    st.dataframe(df_res, use_container_width=True)

    # Graphique des probabilités pour les 6 cas
    fig_tests = px.bar(
        df_res, x='Profil', y='Prob. urgence (%)',
        color='Prob. urgence (%)', color_continuous_scale='RdYlGn_r',
        text='Prob. urgence (%)',
        title="Probabilité de césarienne d'urgence — Cas de tests",
        height=400
    )
    fig_tests.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                             marker_line_color='white')
    fig_tests.update_layout(xaxis_tickangle=-20, coloraxis_showscale=False,
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_tests, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.82rem; padding:12px 0;'>
    🏥 <b>CésaPredict</b> — Application de prédiction de la césarienne d'urgence<br>
    Dataset : DHS Programme · 21 pays · 238 281 observations · Modèle : XGBoost + SMOTE<br>
    <i>⚠️ Usage académique uniquement — ne remplace pas un avis médical professionnel</i>
</div>
""", unsafe_allow_html=True)
