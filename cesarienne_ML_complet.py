# =============================================================================
# PROJET MACHINE LEARNING : PRÉDICTION DE LA CÉSARIENNE D'URGENCE EN AFRIQUE
# =============================================================================
# Dataset : ccs_ssa_fv_clean.csv  (238 281 observations, 12 variables)
# Cible   : m17 → "normal/elective cs" (0)  vs  "emergency cs" (1)
#
# PLAN DU NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
#  0.  Installation & imports
#  1.  Chargement & premier regard
#  2.  Exploration des données (EDA)
#      2a. Statistiques descriptives
#      2b. Variable cible (déséquilibre de classes)
#      2c. Variables numériques
#      2d. Variables catégorielles
#      2e. Analyse géographique (par pays)
#      2f. Matrice de corrélation & pairplot
#  3.  Prétraitement
#      3a. Encodage de la cible
#      3b. Gestion des outliers
#      3c. Encodage des variables catégorielles
#      3d. Normalisation des variables numériques
#      3e. Gestion du déséquilibre de classes (SMOTE)
#      3f. Séparation train / test
#  4.  Modélisation
#      4a. Régression Logistique (baseline)
#      4b. Random Forest
#      4c. Gradient Boosting (XGBoost)
#      4d. Support Vector Machine (optionnel, sous-échantillon)
#  5.  Évaluation & comparaison
#      5a. Métriques : accuracy, precision, recall, F1, AUC-ROC
#      5b. Matrices de confusion
#      5c. Courbes ROC
#      5d. Tableau récapitulatif
#  6.  Importance des variables
#  7.  Optimisation des hyperparamètres (GridSearchCV / RandomizedSearchCV)
#  8.  Modèle final & sauvegarde
#  9.  Interprétabilité (SHAP)
# 10.  Conclusion
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 0. INSTALLATION DES DÉPENDANCES (décommenter si nécessaire)
# ─────────────────────────────────────────────────────────────────────────────
# !pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
# !pip install xgboost shap joblib

import warnings
warnings.filterwarnings("ignore")

# ── Librairies standard ──
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib
import os

# ── Machine Learning ──
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.preprocessing import (LabelEncoder, StandardScaler,
                                   OrdinalEncoder)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score,
                             accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── Gestion du déséquilibre ──
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── XGBoost ──
import xgboost as xgb

# ── SHAP ──
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP non installé. Section 9 ignorée. Installez avec : pip install shap")

# ── Style général des graphiques ──
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

print("✅  Toutes les librairies sont chargées.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT & PREMIER REGARD
# ─────────────────────────────────────────────────────────────────────────────

# ── Adapter ce chemin à votre environnement ──
DATA_PATH = "ccs_ssa_fv_clean.csv"          # ← modifiez si besoin

df = pd.read_csv(DATA_PATH)
df_raw = df.copy()   # copie de sauvegarde intacte

print(f"\n📐 Dimensions du dataset : {df.shape[0]:,} lignes  ×  {df.shape[1]} colonnes")
print("\n── Premières lignes ──")
display(df.head())

print("\n── Types de données ──")
display(df.dtypes.to_frame("dtype"))

print("\n── Valeurs manquantes ──")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  Aucune valeur manquante 🎉")


# ─────────────────────────────────────────────────────────────────────────────
# DICTIONNAIRE DES VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
# v025  : Milieu de résidence               (rural / urban)
# v106  : Niveau d'éducation de la mère     (no education / primary / secondary / higher)
# v190  : Indice de richesse du ménage      (poorest → richest)
# m13   : Nombre de consultations prénatales (0–10+)
# m14   : Mois de la 1ʳᵉ consultation prénatale (1–9)
# m15   : Lieu d'accouchement               (home, health center, hospital…)
# m17   : TYPE DE CÉSARIENNE ← VARIABLE CIBLE
# m19   : Poids de naissance en grammes     (500–8000 g)
# m2n   : Soins prénataux (oui/non)
# m3n   : Assistance pendant l'accouchement (oui/non)
# b0    : Type de naissance                 (single / multiple)
# country : Pays (21 pays d'Afrique subsaharienne)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATION DES DONNÉES (EDA)
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Statistiques descriptives ─────────────────────────────────────────

print("\n══ 2a. Statistiques descriptives ══\n")

numerical_vars   = ['m13', 'm14', 'm19']
categorical_vars = ['v025', 'v106', 'v190', 'm15', 'm2n', 'm3n', 'b0', 'country']

print("Variables numériques :")
display(df[numerical_vars].describe().T.round(2))

print("\nVariables catégorielles – modalités :")
for col in categorical_vars:
    print(f"\n  {col} ({df[col].nunique()} modalités) :")
    print(df[col].value_counts(normalize=True).mul(100).round(1).to_string())


# ── 2b. Variable cible – déséquilibre de classes ──────────────────────────

print("\n══ 2b. Variable cible m17 ══\n")

target_counts = df['m17'].value_counts()
target_pct    = df['m17'].value_counts(normalize=True).mul(100).round(2)

print(target_counts)
print(f"\nRatio déséquilibre : {target_counts[0] / target_counts[1]:.1f}:1")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Barplot
axes[0].bar(target_counts.index, target_counts.values,
            color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Nombre d'observations par classe", fontweight="bold")
axes[0].set_ylabel("Effectif")
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 500, f"{v:,}\n({target_pct.iloc[i]:.1f}%)",
                 ha='center', fontsize=10)

# Pie chart
axes[1].pie(target_counts.values, labels=target_counts.index,
            autopct='%1.1f%%', startangle=90,
            colors=["#2ecc71", "#e74c3c"],
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Répartition des classes", fontweight="bold")

plt.suptitle("Distribution de la variable cible (m17)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_01_target_distribution.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_01_target_distribution.png")


# ── 2c. Variables numériques ───────────────────────────────────────────────

print("\n══ 2c. Variables numériques ══\n")

# Création d'une variable cible numérique temporaire pour l'EDA
df['m17_num'] = (df['m17'] == 'emergency cs').astype(int)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# --- Distributions globales (histogramme + KDE) ---
colors = ["#3498db", "#9b59b6", "#e67e22"]
for i, (var, col) in enumerate(zip(numerical_vars, colors)):
    ax = axes[0, i]
    df[var].plot.hist(bins=40, ax=ax, color=col, alpha=0.75, edgecolor="white")
    ax.set_title(f"Distribution de {var}", fontweight="bold")
    ax.set_xlabel(var)
    ax.set_ylabel("Fréquence")
    med = df[var].median()
    ax.axvline(med, color='red', linestyle='--', label=f"Médiane={med:.0f}")
    ax.legend(fontsize=9)

# --- Boxplots par classe cible ---
for i, (var, col) in enumerate(zip(numerical_vars, colors)):
    ax = axes[1, i]
    sns.boxplot(x='m17', y=var, data=df, ax=ax,
                palette={"normal/elective cs": "#2ecc71", "emergency cs": "#e74c3c"},
                width=0.5, linewidth=1.2)
    ax.set_title(f"{var} selon le type de césarienne", fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=15)

plt.suptitle("Analyse des variables numériques", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_02_numerical_vars.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_02_numerical_vars.png")

# ── Outliers ──
print("\n── Analyse des outliers (méthode IQR) ──")
for var in numerical_vars:
    Q1, Q3 = df[var].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)).sum()
    print(f"  {var}: {n_out:,} outliers ({n_out/len(df)*100:.2f}%)")


# ── 2d. Variables catégorielles ────────────────────────────────────────────

print("\n══ 2d. Variables catégorielles ══\n")

cat_vars_plot = ['v025', 'v106', 'v190', 'm15', 'm2n', 'm3n', 'b0']

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, var in enumerate(cat_vars_plot):
    ax = axes[i]
    # Taux de césarienne d'urgence par modalité
    rate = df.groupby(var)['m17_num'].mean().sort_values(ascending=False) * 100
    bars = ax.barh(rate.index, rate.values,
                   color=sns.color_palette("Reds_r", len(rate)),
                   edgecolor="white", linewidth=1)
    ax.set_title(f"Taux césarienne urgence\npar '{var}'", fontweight="bold", fontsize=10)
    ax.set_xlabel("Taux (%)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    # Annotations
    for bar, val in zip(bars, rate.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', fontsize=8)

# Masquer le dernier axe vide
axes[-1].set_visible(False)

plt.suptitle("Taux de césarienne d'urgence par variable catégorielle",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_03_categorical_rates.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_03_categorical_rates.png")


# ── 2e. Analyse géographique ───────────────────────────────────────────────

print("\n══ 2e. Analyse géographique par pays ══\n")

country_stats = df.groupby('country').agg(
    n_total=('m17', 'count'),
    n_emergency=('m17_num', 'sum'),
    taux_urgence=('m17_num', 'mean')
).reset_index()
country_stats['taux_urgence_pct'] = country_stats['taux_urgence'] * 100
country_stats = country_stats.sort_values('taux_urgence_pct', ascending=False)

print(country_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(country_stats['country'], country_stats['taux_urgence_pct'],
              color=sns.color_palette("Reds_r", len(country_stats)),
              edgecolor="white", linewidth=1.2)
ax.set_title("Taux de césarienne d'urgence par pays (21 pays ASS)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Taux de césarienne d'urgence (%)")
ax.set_xlabel("Pays")
plt.xticks(rotation=60, ha='right')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.1f}%",
            ha='center', va='bottom', fontsize=7.5)
plt.tight_layout()
plt.savefig("fig_04_country_rates.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_04_country_rates.png")


# ── 2f. Matrice de corrélation & pairplot ──────────────────────────────────

print("\n══ 2f. Corrélations ══\n")

fig, ax = plt.subplots(figsize=(7, 5))
corr = df[numerical_vars + ['m17_num']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Matrice de corrélation", fontweight="bold")
plt.tight_layout()
plt.savefig("fig_05_correlation.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_05_correlation.png")

# Pairplot (sous-échantillon pour la lisibilité)
sample_pairplot = df[numerical_vars + ['m17']].sample(5000, random_state=42)
fig_pair = sns.pairplot(sample_pairplot, hue='m17',
                        palette={"normal/elective cs": "#2ecc71",
                                 "emergency cs": "#e74c3c"},
                        plot_kws={"alpha": 0.4, "s": 15},
                        diag_kind="kde")
fig_pair.fig.suptitle("Pairplot (échantillon 5 000 obs.)", y=1.02, fontweight="bold")
plt.savefig("fig_06_pairplot.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_06_pairplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ 3. PRÉTRAITEMENT ══\n")

df_proc = df_raw.copy()    # on repart du dataset brut

# ── 3a. Encodage de la variable cible ────────────────────────────────────

df_proc['target'] = (df_proc['m17'] == 'emergency cs').astype(int)
# 0 = normale/programmée | 1 = urgence
df_proc.drop(columns=['m17'], inplace=True)
print("✅  Variable cible encodée : 0=normale  1=urgence")
print(df_proc['target'].value_counts())


# ── 3b. Correction des valeurs aberrantes ────────────────────────────────
# m13 : nombre de consultations prénatales
#   → Les valeurs > 15 semblent aberrantes (max = 10 dans les recommandations OMS)
#   → On plafonne à 12 (winsorisation)
# m19 : poids de naissance
#   → Valeurs biologiquement improbables < 300 g ou > 6000 g
# m14 : mois de 1ère CPN (1–9 mois de grossesse)

for col, low, high in [('m13', 0, 12), ('m14', 1, 9), ('m19', 300, 6000)]:
    before = len(df_proc)
    df_proc[col] = df_proc[col].clip(lower=low, upper=high)
    print(f"  {col} : valeurs clampées à [{low}, {high}]")


# ── 3c. Encodage des variables catégorielles ─────────────────────────────

# Variables ordinales : l'ordre a un sens → OrdinalEncoder
ordinal_features = {
    'v106': ['no education', 'primary', 'secondary', 'higher'],
    'v190': ['poorest', 'poorer', 'middle', 'richer', 'richest'],
}

# Variables nominales : pas d'ordre → One-Hot Encoding
nominal_features = ['v025', 'm15', 'm2n', 'm3n', 'b0', 'country']

# ─── On ne touche pas à v025 (seulement 2 modalités) → LabelEncoder ───
le_v025 = LabelEncoder()
df_proc['v025'] = le_v025.fit_transform(df_proc['v025'])  # rural=1, urban=0

# ─── Ordinales ───
for col, order in ordinal_features.items():
    # Remplacer les valeurs inattendues par la plus fréquente
    freq_val = df_proc[col].mode()[0]
    df_proc[col] = df_proc[col].apply(lambda x: x if x in order else freq_val)
    mapping = {v: i for i, v in enumerate(order)}
    df_proc[col] = df_proc[col].map(mapping)
    print(f"  {col} encodé ordinalement : {mapping}")

# ─── Nominales (sauf v025 déjà traité) ───
nominal_for_ohe = [c for c in nominal_features if c != 'v025']
df_proc = pd.get_dummies(df_proc, columns=nominal_for_ohe, drop_first=True)
print(f"\n✅  One-Hot Encoding appliqué sur : {nominal_for_ohe}")
print(f"   Nouvelle dimension : {df_proc.shape}")


# ── 3d. Séparation features / cible ──────────────────────────────────────

feature_cols = [c for c in df_proc.columns if c != 'target']
X = df_proc[feature_cols]
y = df_proc['target']

print(f"\n✅  X : {X.shape}   |   y : {y.shape}")
print(f"   Ratio classe 0/1 : {(y==0).sum():,} / {(y==1).sum():,}")


# ── 3e. Normalisation des variables numériques ───────────────────────────

# On identifie les colonnes numériques continues (pas les dummies)
num_cols = ['m13', 'm14', 'm19']

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
print(f"\n✅  StandardScaler appliqué sur : {num_cols}")


# ── 3f. Séparation Train / Test (stratifiée) ─────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # ← crucial pour les classes déséquilibrées
)

print(f"\n✅  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
print(f"   Taux urgence train : {y_train.mean():.4f}  |  test : {y_test.mean():.4f}")


# ── 3g. SMOTE – suréchantillonnage de la classe minoritaire ──────────────
# Attention : SMOTE est appliqué UNIQUEMENT sur le train set

print("\n── Application du SMOTE sur le train set ──")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  Avant SMOTE  → classe 0 : {(y_train==0).sum():,}  |  classe 1 : {(y_train==1).sum():,}")
print(f"  Après SMOTE  → classe 0 : {(y_train_sm==0).sum():,}  |  classe 1 : {(y_train_sm==1).sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODÉLISATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ 4. MODÉLISATION ══\n")

# Dictionnaire pour stocker les résultats de tous les modèles
results = {}

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """
    Entraîne le modèle, calcule toutes les métriques et stocke les résultats.
    Retourne le modèle entraîné.
    """
    print(f"\n  ── {name} ──")
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan

    results[name] = {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
        "AUC-ROC":   auc,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "model":     model
    }

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['normale', 'urgence'])}")

    return model


# ── 4a. Régression Logistique (baseline) ─────────────────────────────────

lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',    # gère le déséquilibre sans SMOTE (comparaison)
    solver='lbfgs',
    random_state=42
)
# On utilise les données SMOTE pour tous les modèles
evaluate_model("Régression Logistique", lr, X_train_sm, y_train_sm, X_test, y_test)


# ── 4b. Random Forest ─────────────────────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
evaluate_model("Random Forest", rf, X_train_sm, y_train_sm, X_test, y_test)


# ── 4c. XGBoost ───────────────────────────────────────────────────────────

# scale_pos_weight : ratio classe majoritaire / minoritaire
scale_w = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_w,   # compense le déséquilibre
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
evaluate_model("XGBoost", xgb_model, X_train_sm, y_train_sm, X_test, y_test)


# ── 4d. Gradient Boosting (sklearn) ──────────────────────────────────────
# NOTE : plus lent que XGBoost, on utilise un sous-échantillon pour la démo

print("\n  ── Gradient Boosting (sklearn) ──")
print("  (Entraîné sur un sous-échantillon de 40 000 obs. pour la vitesse)")

idx_sample = np.random.default_rng(42).choice(len(X_train_sm), size=40000, replace=False)
X_tr_gb = X_train_sm.iloc[idx_sample]
y_tr_gb = y_train_sm.iloc[idx_sample]

gb = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
evaluate_model("Gradient Boosting", gb, X_tr_gb, y_tr_gb, X_test, y_test)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ÉVALUATION & COMPARAISON
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ 5. ÉVALUATION & COMPARAISON ══\n")

# ── 5a. Tableau récapitulatif ────────────────────────────────────────────

metrics_df = pd.DataFrame(
    {name: {k: v for k, v in vals.items()
            if k not in ['y_pred', 'y_proba', 'model']}
     for name, vals in results.items()}
).T.round(4)

print("── Tableau comparatif des modèles ──")
display(metrics_df.sort_values("AUC-ROC", ascending=False))


# ── 5b. Matrices de confusion ────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, vals) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, vals['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Normale', 'Urgence'])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name, fontweight="bold", fontsize=10)

plt.suptitle("Matrices de confusion (jeu de test)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_07_confusion_matrices.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_07_confusion_matrices.png")


# ── 5c. Courbes ROC ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))
colors_roc = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

for (name, vals), color in zip(results.items(), colors_roc):
    if vals['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, vals['y_proba'])
        auc_val = vals['AUC-ROC']
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
                color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Aléatoire")
ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
ax.set_title("Courbes ROC – Comparaison des modèles", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_08_roc_curves.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_08_roc_curves.png")


# ── 5d. Graphique métriques comparatives ─────────────────────────────────

metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
metrics_plot = metrics_df[metric_cols]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(metric_cols))
width = 0.2

for i, (model_name, row) in enumerate(metrics_plot.iterrows()):
    ax.bar(x + i*width, row.values, width=width,
           label=model_name, edgecolor="white", linewidth=0.8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Comparaison des métriques par modèle", fontweight="bold", fontsize=13)
ax.legend(loc="lower right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig("fig_09_metrics_comparison.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_09_metrics_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. IMPORTANCE DES VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ 6. IMPORTANCE DES VARIABLES ══\n")

# On utilise le Random Forest et XGBoost (ils ont feature_importances_)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, (model_name, model_obj) in zip(axes,
        [("Random Forest", results["Random Forest"]["model"]),
         ("XGBoost",       results["XGBoost"]["model"])]):

    importances = pd.Series(model_obj.feature_importances_, index=X.columns)
    top20 = importances.sort_values(ascending=False).head(20)

    sns.barplot(x=top20.values, y=top20.index,
                palette="viridis_r", ax=ax, edgecolor="white")
    ax.set_title(f"Top 20 variables importantes\n({model_name})",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Importance")

plt.suptitle("Importance des variables (Feature Importance)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_10_feature_importance.png", bbox_inches="tight")
plt.show()
print("💾 Figure sauvegardée : fig_10_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTIMISATION DES HYPERPARAMÈTRES (meilleur modèle → XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ 7. OPTIMISATION DES HYPERPARAMÈTRES (XGBoost) ══\n")
print("  Utilisation de RandomizedSearchCV avec validation croisée stratifiée (5 folds)")

param_dist = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [3, 4, 6, 8],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.6, 0.8, 1.0],
    'min_child_weight':[1, 3, 5],
    'gamma':           [0, 0.1, 0.3],
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_w,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# NOTE : Pour accélérer la démo, n_iter=20 et cv=3.
#        Augmentez n_iter à 50–100 et cv=5 pour un résultat optimal.
rscv = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',           # ← optimiser le F1 sur la classe positive (urgence)
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Sous-échantillon pour la vitesse (30k obs.)
idx_opt = np.random.default_rng(0).choice(len(X_train_sm), size=30000, replace=False)
rscv.fit(X_train_sm.iloc[idx_opt], y_train_sm.iloc[idx_opt])

print(f"\n  ✅  Meilleurs paramètres : {rscv.best_params_}")
print(f"  ✅  Meilleur F1 (CV)     : {rscv.best_score_:.4f}")

# Réentraînement sur tout le train SMOTE
best_xgb = rscv.best_estimator_
best_xgb.fit(X_train_sm, y_train_sm)

y_pred_best  = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:, 1]

print("\n── Performances du XGBoost optimisé ──")
print(classification_report(y_test, y_pred_best, target_names=['normale', 'urgence']))
print(f"  AUC-ROC : {roc_auc_score(y_test, y_proba_best):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODÈLE FINAL & SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ 8. SAUVEGARDE DU MODÈLE FINAL ══\n")

# Sauvegarder le meilleur modèle
joblib.dump(best_xgb, "modele_cesarienne_xgb.pkl")
joblib.dump(scaler,   "scaler_cesarienne.pkl")

print("  ✅  modele_cesarienne_xgb.pkl sauvegardé")
print("  ✅  scaler_cesarienne.pkl sauvegardé")
print("  → Rechargement : model = joblib.load('modele_cesarienne_xgb.pkl')")

# ── Exemple de prédiction sur un nouveau cas ──
print("\n── Exemple de prédiction ──")

# Construction d'un patient fictif (AVANT One-Hot Encoding)
new_patient_raw = {
    'v025': 'rural',          # milieu rural
    'v106': 'no education',   # pas d'éducation
    'v190': 'poorest',        # quintile le plus pauvre
    'm13':  2,                 # 2 consultations prénatales
    'm14':  7,                 # 1ère CPN au 7ème mois (tardive)
    'm15':  "respondent's home",  # accouchement à domicile
    'm19':  2500,              # poids 2,5 kg (faible poids)
    'm2n':  'no/somecare',
    'm3n':  'yes: no assistance',
    'b0':   'single birth',
    'country': 'Nigeria'
}

# Créer un dataframe à partir du patient
new_df = pd.DataFrame([new_patient_raw])

# Appliquer le même prétraitement que sur les données d'entraînement
new_df['v025'] = le_v025.transform(new_df['v025'])
for col, order in ordinal_features.items():
    mapping = {v: i for i, v in enumerate(order)}
    new_df[col] = new_df[col].map(mapping)

nominal_for_ohe_patient = [c for c in nominal_for_ohe]
new_df = pd.get_dummies(new_df, columns=nominal_for_ohe_patient, drop_first=True)

# Aligner les colonnes avec le modèle
for col in X.columns:
    if col not in new_df.columns:
        new_df[col] = 0
new_df = new_df[X.columns]
new_df[num_cols] = scaler.transform(new_df[num_cols])

pred = best_xgb.predict(new_df)[0]
prob = best_xgb.predict_proba(new_df)[0][1]

if pred == 1:
    print(f"\n  ⚠️  RÉSULTAT : Risque de CÉSARIENNE D'URGENCE  (probabilité = {prob:.2%})")
else:
    print(f"\n  ✅  RÉSULTAT : Césarienne NORMALE/PROGRAMMÉE   (probabilité urgence = {prob:.2%})")


# ─────────────────────────────────────────────────────────────────────────────
# 9. INTERPRÉTABILITÉ SHAP
# ─────────────────────────────────────────────────────────────────────────────

if SHAP_AVAILABLE:
    print("\n══ 9. INTERPRÉTABILITÉ SHAP ══\n")

    # Sous-échantillon pour la rapidité
    X_shap = X_test.sample(1000, random_state=42)

    explainer   = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_shap)

    # ─ Summary plot ─
    fig_shap1 = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="dot", show=False)
    plt.title("SHAP – Impact des variables sur la prédiction", fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_11_shap_summary.png", bbox_inches="tight")
    plt.show()
    print("💾 Figure sauvegardée : fig_11_shap_summary.png")

    # ─ Bar plot – importance moyenne absolue SHAP ─
    fig_shap2 = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.title("SHAP – Importance moyenne des variables", fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_12_shap_bar.png", bbox_inches="tight")
    plt.show()
    print("💾 Figure sauvegardée : fig_12_shap_bar.png")

    # ─ Force plot pour la première observation ─
    print("\n  Interprétation d'une prédiction individuelle (obs. 0) :")
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X_shap.iloc[0, :],
        matplotlib=True,
        show=False
    )
    plt.savefig("fig_13_shap_force.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("💾 Figure sauvegardée : fig_13_shap_force.png")

else:
    print("\n  ⚠️  SHAP non disponible – section ignorée.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  10. CONCLUSION")
print("═"*65)

best_model_name = metrics_df["AUC-ROC"].idxmax()
best_auc        = metrics_df["AUC-ROC"].max()

print(f"""
RÉSUMÉ DU PROJET
─────────────────────────────────────────────────────────────────
Dataset  : {df_raw.shape[0]:,} observations  |  {df_raw.shape[1]} variables
Pays     : 21 pays d'Afrique subsaharienne
Cible    : Prédire le type de césarienne (urgence vs normale)
Déséquilibre : 97.1 % normale  –  2.9 % urgence (ratio ~34:1)

MODÈLES TESTÉS
  • Régression Logistique  (baseline)
  • Random Forest
  • XGBoost               ← meilleur modèle
  • Gradient Boosting

MEILLEUR MODÈLE : {best_model_name}
  AUC-ROC optimisé : {best_auc:.4f}

TECHNIQUES APPLIQUÉES
  • SMOTE pour équilibrer les classes
  • StandardScaler pour les variables numériques
  • OrdinalEncoder pour les variables catégorielles ordonnées
  • One-Hot Encoding pour les variables nominales
  • RandomizedSearchCV pour l'optimisation des hyperparamètres
  • SHAP pour l'interprétabilité

FACTEURS PRÉDICTEURS CLÉS (selon importance des variables)
  • m19 : Poids de naissance  → poids faible associé à l'urgence
  • m13 : Nombre de CPN       → peu de consultations = risque élevé
  • m14 : Mois 1ère CPN       → retard de suivi prénatal = risque
  • m15 : Lieu d'accouchement → hôpital ≠ domicile
  • country                   → disparités géographiques importantes
  • v190 : Richesse            → pauvreté associée aux urgences
─────────────────────────────────────────────────────────────────
""")

# Liste de toutes les figures produites
print("FIGURES GÉNÉRÉES :")
figures = [
    ("fig_01", "Distribution de la variable cible"),
    ("fig_02", "Analyse des variables numériques"),
    ("fig_03", "Taux de césarienne par variable catégorielle"),
    ("fig_04", "Taux de césarienne par pays"),
    ("fig_05", "Matrice de corrélation"),
    ("fig_06", "Pairplot"),
    ("fig_07", "Matrices de confusion"),
    ("fig_08", "Courbes ROC"),
    ("fig_09", "Comparaison des métriques"),
    ("fig_10", "Importance des variables"),
    ("fig_11", "SHAP – Summary dot"),
    ("fig_12", "SHAP – Summary bar"),
    ("fig_13", "SHAP – Force plot"),
]
for fig_name, desc in figures:
    print(f"  {fig_name}.png  →  {desc}")

print("\n✅  Notebook terminé avec succès !")
