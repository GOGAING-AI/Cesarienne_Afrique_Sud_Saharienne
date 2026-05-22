# 🏥 CésarienneAI — Prédiction de la Césarienne d'Urgence en Afrique Subsaharienne

> **Projet de Machine Learning · AIMS / Groupe 3**
> Modélisation prédictive de la césarienne d'urgence à partir des données DHS
> de 21 pays d'Afrique Subsaharienne — 238 281 accouchements

---

## 📋 Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Dataset](#2-dataset)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Installation et lancement](#4-installation-et-lancement)
5. [Pipeline Machine Learning](#5-pipeline-machine-learning)
6. [Modèles mathématiques](#6-modèles-mathématiques)
7. [Résultats et performances](#7-résultats-et-performances)
8. [Application Streamlit](#8-application-streamlit)
9. [Outils et technologies](#9-outils-et-technologies)
10. [Limites et perspectives](#10-limites-et-perspectives)
11. [Références](#11-références)
12. [Auteurs](#12-auteurs)

---

## 1. Contexte et problématique

### 1.1 Contexte médical

La césarienne d'urgence est une intervention chirurgicale vitale déclenchée lorsque la vie de la mère ou du nouveau-né est en danger immédiat. En Afrique subsaharienne, les inégalités d'accès aux soins, la faiblesse des infrastructures médicales et les disparités socio-économiques rendent la **détection précoce des grossesses à risque** d'une importance capitale.

Contrairement à une césarienne programmée — planifiée à l'avance pour des raisons médicales connues — la césarienne d'urgence survient de façon imprévisible et requiert une mobilisation immédiate du personnel soignant et du plateau technique. Dans des contextes où les blocs opératoires sont rares et les délais de prise en charge longs, **anticiper ces situations peut sauver des vies**.

### 1.2 Problématique scientifique

> **Peut-on prédire, à partir d'informations disponibles dès la consultation prénatale, si une femme accouchant par césarienne sera une urgence ou une intervention programmée ?**

Il s'agit d'un problème de **classification binaire supervisée** :

| Classe | Label | Description |
|--------|-------|-------------|
| 0 | `normal/elective cs` | Césarienne normale ou programmée |
| 1 | `emergency cs` | Césarienne d'urgence |

### 1.3 Défi principal : le déséquilibre des classes

Le dataset présente un déséquilibre extrême de **34:1** — seulement 2.86% des observations correspondent à des césariennes d'urgence. Un modèle naïf qui prédit systématiquement "normale" obtiendrait 97.14% d'accuracy sans aucune utilité clinique. La gestion de ce déséquilibre est le défi central de ce projet.

---

## 2. Dataset

### 2.1 Source

Les données proviennent des **Enquêtes Démographiques et de Santé (DHS — Demographic and Health Surveys)**, programme financé par l'USAID et reconnu comme la référence mondiale en matière de données de santé maternelle et infantile dans les pays en développement.

### 2.2 Caractéristiques générales

| Attribut | Valeur |
|----------|--------|
| Fichier | `ccs_ssa_fv_clean.csv` |
| Observations | 238 281 |
| Variables | 12 |
| Valeurs manquantes | 0 (dataset pré-nettoyé) |
| Pays couverts | 21 pays d'Afrique Subsaharienne |
| Variable cible | `m17` (type de césarienne) |

### 2.3 Description des variables

| Variable | Type | Description | Modalités / Plage |
|----------|------|-------------|-------------------|
| `v025` | Catégorielle nominale | Milieu de résidence | `rural`, `urban` |
| `v106` | Catégorielle **ordinale** | Niveau d'éducation de la mère | `no education` → `higher` |
| `v190` | Catégorielle **ordinale** | Indice de richesse du ménage | `poorest` → `richest` |
| `m13` | Numérique entier | Nombre de consultations prénatales (CPN) | 0 – 10 |
| `m14` | Numérique entier | Mois de la 1ère consultation prénatale | 1 – 9 |
| `m15` | Catégorielle nominale | Lieu d'accouchement | Domicile, hôpital, maternité… |
| `m17` | **Variable cible** | Type de césarienne | `normal/elective cs`, `emergency cs` |
| `m19` | Numérique continu | Poids de naissance (grammes) | 500 – 8 000 g |
| `m2n` | Catégorielle nominale | Suivi des soins prénataux | `yes/nocare`, `no/somecare` |
| `m3n` | Catégorielle nominale | Assistance pendant l'accouchement | `no: some assistance`, `yes: no assistance` |
| `b0` | Catégorielle nominale | Type de naissance | `single birth`, `1st/2nd/3rd of multiple` |
| `country` | Catégorielle nominale | Pays | 21 pays ASS |

### 2.4 Pays du dataset et taux de césarienne d'urgence

| Pays | Taux urgence | Pays | Taux urgence |
|------|-------------|------|-------------|
| Rwanda | **8.90%** | Senegal | 2.78% |
| South Africa | **8.37%** | Liberia | 2.75% |
| Malawi | 5.04% | Cameroon | 2.19% |
| Tanzania | 3.84% | Mauritania | 2.09% |
| Zimbabwe | 3.90% | Angola | 1.85% |
| Uganda | 3.76% | Ethiopia | 1.59% |
| Zambia | 3.27% | Mali | 1.51% |
| Burundi | 3.24% | Nigeria | 1.47% |
| Benin | 3.02% | Guinea | 1.45% |
| Gambia | 2.20% | Madagascar | **1.15%** |
| Sierra Leone | 2.42% | — | — |

---

## 3. Architecture du projet

```
cesarienne-ai/
│
├── 📄 ccs_ssa_fv_clean.csv          # Dataset source (DHS)
│
├── 🐍 cesarienne_ML_complet.py      # Notebook ML complet (10 étapes)
│
├── 🌐 app_cesarienne.py             # Application Streamlit
│
├── 🤖 modele_cesarienne_xgb.pkl     # Modèle XGBoost sérialisé (généré)
├── ⚖️  scaler_cesarienne.pkl         # StandardScaler sérialisé (généré)
│
├── 🗺️  ne_110m_admin_0_countries/    # Shapefile Natural Earth (optionnel)
│   └── ne_110m_admin_0_countries.shp
│
├── 📊 fig_01_target_distribution.png
├── 📊 fig_02_numerical_vars.png
├── 📊 fig_03_categorical_rates.png
├── 📊 fig_04_country_rates.png
├── 📊 fig_05_correlation.png
├── 📊 fig_06_pairplot.png
├── 📊 fig_07_confusion_matrices.png
├── 📊 fig_08_roc_curves.png
├── 📊 fig_09_metrics_comparison.png
├── 📊 fig_10_feature_importance.png
├── 📊 fig_11_shap_summary.png        # (si SHAP installé)
├── 📊 fig_12_shap_bar.png            # (si SHAP installé)
├── 📊 fig_13_shap_force.png          # (si SHAP installé)
│
└── 📖 README.md                     # Ce fichier
```

---

## 4. Installation et lancement

### 4.1 Prérequis

- Python 3.9 ou supérieur
- pip

### 4.2 Installation des dépendances

```bash
# Cloner ou télécharger le projet
cd cesarienne-ai

# Installer toutes les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn \
            imbalanced-learn xgboost joblib plotly geopandas \
            streamlit shap
```

### 4.3 Étape 1 — Générer le modèle (notebook ML)

Exécuter `cesarienne_ML_complet.py` dans Jupyter ou VSCode.
Ce script génère `modele_cesarienne_xgb.pkl` et `scaler_cesarienne.pkl`.

```bash
jupyter notebook cesarienne_ML_complet.py
# ou
python cesarienne_ML_complet.py
```

### 4.4 Étape 2 — Lancer l'application Streamlit

```bash
streamlit run app_cesarienne.py
```

Ouvrir le navigateur à l'adresse : `http://localhost:8501`

### 4.5 Carte géographique (optionnel)

Pour utiliser la carte GeoPandas avec shapefile local :

1. Télécharger Natural Earth 110m Admin 0 :
   👉 https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
2. Extraire dans `ne_110m_admin_0_countries/`
3. La carte Plotly (choroplèthe interactive) fonctionne **sans** shapefile

---

## 5. Pipeline Machine Learning

Le projet suit rigoureusement les 10 étapes standard d'un projet de Machine Learning en production.

### Étape 1 — Chargement et inspection

Chargement du fichier CSV, vérification des types, identification des valeurs manquantes et premier regard statistique sur les distributions. Le dataset est déjà propre (0 valeur manquante).

### Étape 2 — Analyse exploratoire (EDA)

Exploration approfondie en 6 sous-sections :

- **Distribution de la cible** : visualisation du déséquilibre 34:1
- **Variables numériques** : histogrammes, boxplots par classe, détection d'outliers par méthode IQR
- **Variables catégorielles** : taux de césarienne d'urgence par modalité pour chaque variable
- **Analyse géographique** : classement des 21 pays, carte choroplèthe
- **Corrélations** : heatmap de la matrice de corrélation
- **Pairplot** : relations bivariées sur échantillon de 5 000 observations

**13 figures** sont produites et sauvegardées automatiquement.

### Étape 3 — Prétraitement

Le prétraitement suit un ordre précis et reproductible :

#### 3a. Encodage de la variable cible
```python
df['target'] = (df['m17'] == 'emergency cs').astype(int)
# 0 = normale/programmée | 1 = urgence
```

#### 3b. Correction des outliers (Winsorisation / Clamping)
Les valeurs aberrantes sont bornées plutôt que supprimées pour préserver les effectifs :

| Variable | Borne inférieure | Borne supérieure | Justification |
|----------|-----------------|-----------------|---------------|
| `m13` | 0 | 12 | Recommandation OMS : ≤ 12 CPN |
| `m14` | 1 | 9 | Mois de grossesse 1 à 9 |
| `m19` | 300 | 6 000 | Poids biologiquement plausible |

#### 3c. Encodage des variables catégorielles

Trois stratégies selon la nature de chaque variable :

**Variables ordinales** — `OrdinalEncoder` (l'ordre a un sens médical) :
```
v106 : no education(0) < primary(1) < secondary(2) < higher(3)
v190 : poorest(0) < poorer(1) < middle(2) < richer(3) < richest(4)
```

**Variable binaire** — `LabelEncoder` :
```
v025 : urban(0), rural(1)
```

**Variables nominales** — `One-Hot Encoding` (drop_first=True) :
```
m15, m2n, m3n, b0, country → création de colonnes indicatrices
```

#### 3d. Normalisation
`StandardScaler` appliqué aux variables numériques continues (`m13`, `m14`, `m19`) :
```
x_normalisé = (x - μ) / σ
```

#### 3e. Séparation Train / Test
Division stratifiée pour préserver le ratio des classes :
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Train : 190 624 obs. | Test : 47 657 obs.
```

#### 3f. SMOTE — Gestion du déséquilibre

**SMOTE (Synthetic Minority Over-sampling Technique)** génère synthétiquement de nouvelles observations de la classe minoritaire en interpolant entre les k-plus-proches-voisins existants :

```
x_new = x_i + λ · (x_nn - x_i),   λ ∈ [0, 1]
```

Appliqué **uniquement sur le train set** pour éviter la fuite d'information :

| Classe | Avant SMOTE | Après SMOTE |
|--------|------------|------------|
| Normale (0) | 185 160 | 185 160 |
| Urgence (1) | 5 464 | 185 160 |

### Étape 4 — Modélisation

Quatre modèles entraînés et comparés :

| Modèle | Rôle | Particularité |
|--------|------|---------------|
| Régression Logistique | Baseline | Simple, interprétable |
| Random Forest | Ensembliste | Robuste, importance des variables |
| XGBoost | **Modèle final** | Meilleur AUC, boosting |
| Gradient Boosting | Référence sklearn | Validation croisée |

### Étape 5 — Évaluation

Métriques calculées sur le **jeu de test uniquement** (données jamais vues) :

- **Accuracy** : proportion globale de prédictions correctes
- **Precision** : parmi les urgences prédites, combien sont vraies ?
- **Recall** : parmi les vraies urgences, combien sont détectées ? *(métrique critique)*
- **F1-Score** : moyenne harmonique precision/recall
- **AUC-ROC** : aire sous la courbe ROC, métrique principale

> Le **Recall** est la métrique la plus importante médicalement : manquer une urgence (faux négatif) est bien plus grave que déclencher une fausse alerte (faux positif).

### Étape 6 — Importance des variables

`feature_importances_` extraites de Random Forest et XGBoost. Les 5 variables les plus importantes identifiées :

1. `m19` — Poids de naissance
2. `m13` — Nombre de consultations prénatales
3. `m14` — Mois de la 1ère consultation
4. `country_*` — Appartenance géographique
5. `v190` — Indice de richesse

### Étape 7 — Optimisation des hyperparamètres

`RandomizedSearchCV` avec validation croisée stratifiée à 3 folds, optimisé sur le F1-Score (classe urgence) :

```python
param_dist = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':        [3, 4, 6, 8],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0, 0.1, 0.3]
}
```

### Étape 8 — Sauvegarde

```python
joblib.dump(best_xgb, "modele_cesarienne_xgb.pkl")
joblib.dump(scaler,   "scaler_cesarienne.pkl")
```

### Étape 9 — Interprétabilité SHAP

**SHAP (SHapley Additive exPlanations)** explique chaque prédiction individuelle en décomposant la contribution de chaque variable selon la théorie des jeux coopératifs (valeurs de Shapley) :

```
f(x) = E[f(x)] + Σᵢ φᵢ
```

Trois visualisations produites : summary dot plot, bar chart des importances moyennes, force plot individuel.

### Étape 10 — Conclusion et déploiement

Synthèse des résultats, facteurs prédicteurs clés et déploiement via l'application Streamlit.

---

## 6. Modèles mathématiques

### 6.1 Régression Logistique (baseline)

Modèle linéaire généralisé utilisant la **fonction sigmoïde** pour transformer une combinaison linéaire de features en probabilité :

$$P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \sum_{j=1}^{p} \beta_j x_j)}}$$

Les coefficients $\beta_j$ sont estimés par **maximisation de la log-vraisemblance** :

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**Décision** : $\hat{y} = \mathbf{1}[P(Y=1 \mid \mathbf{x}) \geq 0.5]$

### 6.2 Random Forest

Algorithme d'apprentissage ensembliste basé sur $B$ arbres de décision indépendants. Chaque arbre $h_b$ est entraîné par **bootstrap** sur un sous-ensemble aléatoire de features. La prédiction finale est un **vote majoritaire** :

$$\hat{y} = \text{mode}\left\{ h_b(\mathbf{x}) \right\}_{b=1}^{B}$$

Chaque nœud de chaque arbre est divisé en minimisant l'**impureté de Gini** :

$$G = 1 - \sum_{k=0}^{1} p_k^2$$

### 6.3 XGBoost — Modèle principal retenu

XGBoost (eXtreme Gradient Boosting — Chen & Guestrin, 2016) construit les arbres **séquentiellement**, chaque arbre $f_t$ corrigeant les erreurs résiduelles du précédent.

**Prédiction à l'itération t :**

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(\mathbf{x}_i)$$

**Fonction objectif à minimiser :**

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\left(y_i,\ \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$

**Terme de régularisation** (contrôle la complexité de l'arbre) :

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Où $T$ est le nombre de feuilles, $w_j$ les poids foliaires, $\gamma$ la pénalité sur le nombre de feuilles, $\lambda$ la régularisation L2.

En développant $\ell$ en **approximation de Taylor au second ordre** autour de $\hat{y}^{(t-1)}$ :

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

Avec $g_i = \partial_{\hat{y}} \ell(y_i, \hat{y}^{(t-1)})$ (gradient) et $h_i = \partial^2_{\hat{y}} \ell(y_i, \hat{y}^{(t-1)})$ (hessien).

**Gestion du déséquilibre** via le paramètre `scale_pos_weight` :

$$\text{scale\_pos\_weight} = \frac{|\text{classe majoritaire}|}{|\text{classe minoritaire}|} = \frac{231\,464}{6\,817} \approx 34$$

Cela revient à **multiplier le gradient des exemples positifs** (urgences) par 34, forçant le modèle à pénaliser davantage les erreurs sur la classe minoritaire.

---

## 7. Résultats et performances

### 7.1 Tableau comparatif des modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Régression Logistique | ~0.88 | ~0.15 | ~0.72 | ~0.25 | ~0.87 |
| Random Forest | ~0.95 | ~0.28 | ~0.60 | ~0.38 | ~0.88 |
| **XGBoost** | **~0.96** | **~0.32** | **~0.62** | **~0.42** | **~0.89** |
| Gradient Boosting | ~0.94 | ~0.25 | ~0.58 | ~0.35 | ~0.87 |

> Les valeurs exactes dépendent des résultats de l'optimisation RandomizedSearchCV.

### 7.2 Interprétation des métriques

- **Accuracy élevée mais trompeuse** : reflet du déséquilibre, pas de la vraie performance
- **Recall ~62%** sur la classe urgence : le modèle détecte plus de 6 urgences sur 10
- **AUC-ROC ~0.89** : excellente capacité discriminante, bien supérieure au hasard (0.5)
- **F1-Score** : compromis precision/recall, métrique principale d'optimisation

### 7.3 Facteurs prédicteurs clés

D'après SHAP et l'importance des variables :

| Rang | Variable | Interprétation médicale |
|------|----------|------------------------|
| 1 | `m19` (poids naissance) | Faible poids → risque accru de complications |
| 2 | `m13` (nb CPN) | Peu de consultations → grossesse non suivie |
| 3 | `m14` (mois 1ère CPN) | CPN tardive → problèmes non détectés précocement |
| 4 | `country` | Disparités systémiques entre pays |
| 5 | `v190` (richesse) | Pauvreté → accès limité aux soins de qualité |

---

## 8. Application Streamlit

### 8.1 Structure de l'interface

L'application est organisée en **4 onglets** :

#### 🔮 Onglet Prédiction
Interface de saisie des paramètres d'une patiente avec :
- **Jauge dynamique** de probabilité de risque
- **Barres de probabilité** par classe (normale vs urgence)
- **5 métriques globales** du dataset
- **Analyse automatique des facteurs de risque** basée sur les seuils OMS

#### 📊 Onglet Analyse des données
Visualisations interactives (Plotly) du dataset complet :
- Distribution de la variable cible (donut chart)
- Histogramme du poids de naissance par classe
- Boxplots des variables numériques
- Taux d'urgence par variable catégorielle (6 graphiques)
- Matrice de corrélation interactive

#### 🗺️ Onglet Carte de l'Afrique
Deux modes de visualisation :
- **Plotly choroplèthe** : carte interactive, fonctionne sans shapefile, hover/zoom
- **GeoPandas/Matplotlib** : utilise le shapefile Natural Earth local, annotations des taux par pays

Complétée par un barplot de classement des 21 pays.

#### 🧪 Onglet Cas de test
6 profils patients prédéfinis couvrant les extrêmes du dataset :

| Profil | Type attendu | Description |
|--------|-------------|-------------|
| Profil 1 | URGENCE | Femme rurale, non éduquée, très pauvre, domicile, aucun suivi |
| Profil 2 | URGENCE | Urbaine, éducation primaire, hôpital, naissance multiple |
| Profil 3 | URGENCE | Rwanda (taux national 8.9%), rural, peu de CPN |
| Profil 4 | NORMALE | Urbaine, éducation supérieure, aisée, hôpital central, suivi complet |
| Profil 5 | NORMALE | Maternité, éduquée, CPN précoces, assistance complète |
| Profil 6 | URGENCE | Grossesse multiple, faible poids, aucune assistance qualifiée |

### 8.2 Pipeline de prédiction en temps réel

```
Saisie utilisateur (sidebar)
        ↓
LabelEncoder (v025)
        ↓
OrdinalEncoder (v106, v190)
        ↓
One-Hot Encoding (m15, m2n, m3n, b0, country)
        ↓
Alignement des colonnes avec le modèle entraîné
        ↓
StandardScaler (m13, m14, m19)
        ↓
XGBoost.predict_proba()
        ↓
Affichage jauge + analyse des facteurs de risque
```

---

## 9. Outils et technologies

### 9.1 Langage et environnement

| Outil | Version | Usage |
|-------|---------|-------|
| Python | ≥ 3.9 | Langage principal |
| Jupyter Notebook | ≥ 6.0 | Développement et exploration |

### 9.2 Manipulation des données

| Bibliothèque | Usage |
|-------------|-------|
| **pandas** | Chargement, manipulation, agrégation du dataset |
| **NumPy** | Calculs numériques, opérations matricielles |

### 9.3 Visualisation

| Bibliothèque | Usage |
|-------------|-------|
| **Matplotlib** | Graphiques statiques, figures du notebook |
| **Seaborn** | Heatmaps, pairplots, boxplots stylisés |
| **Plotly** | Graphiques interactifs dans Streamlit (choroplèthe, jauge, barres) |
| **GeoPandas** | Lecture du shapefile, carte choroplèthe avec matplotlib |

### 9.4 Machine Learning

| Bibliothèque | Usage |
|-------------|-------|
| **scikit-learn** | Preprocessing, modèles (LR, RF, GB), métriques, cross-validation |
| **XGBoost** | Gradient Boosting optimisé, modèle final |
| **imbalanced-learn** | SMOTE pour l'équilibrage des classes |
| **SHAP** | Interprétabilité des prédictions (valeurs de Shapley) |
| **joblib** | Sérialisation/désérialisation des modèles |

### 9.5 Application web

| Bibliothèque | Usage |
|-------------|-------|
| **Streamlit** | Framework de l'application interactive |

### 9.6 Données géographiques

| Ressource | Usage |
|-----------|-------|
| **Natural Earth 110m** | Shapefile des frontières africaines |
| **ISO 3166-1 alpha-3** | Codes pays pour la carte Plotly choroplèthe |

---

## 10. Limites et perspectives

### 10.1 Limites actuelles

**Limites méthodologiques :**
- SMOTE génère des observations synthétiques qui peuvent ne pas refléter la réalité clinique
- Les variables DHS sont auto-déclarées par les femmes interrogées — biais de mémorisation possible
- Certaines variables importantes sont absentes : antécédents obstétricaux, complications pendant la grossesse, données biologiques (tension artérielle, glycémie)
- Le modèle apprend des **associations statistiques**, pas des relations causales

**Limites du dataset :**
- Données transversales (cross-sectional) : une seule mesure par femme, pas de suivi longitudinal
- Hétérogénéité entre pays dans les pratiques de collecte DHS
- Représentativité limitée à la période de collecte des enquêtes (2010–2020 selon les pays)

**Limites de déploiement :**
- Le modèle n'a pas été validé prospectivement dans un contexte clinique réel
- Les performances peuvent varier significativement selon les pays non représentés dans le dataset

### 10.2 Perspectives d'amélioration

**À court terme :**
- Intégrer des variables d'antécédents médicaux (parité, complications antérieures)
- Tester des modèles de deep learning (TabNet, transformers tabulaires) sur des jeux d'entraînement plus larges
- Calibration des probabilités (Platt scaling, isotonic regression) pour des probabilités plus fiables

**À moyen terme :**
- Validation prospective dans au moins 2 pays du dataset
- Développement d'une API REST pour intégration dans des systèmes d'information hospitaliers
- Interface mobile (PWA) adaptée aux agents de santé communautaires

**À long terme :**
- Extension aux pays d'Afrique du Nord et centrale
- Intégration de données cliniques en temps réel (IoT, dossiers médicaux électroniques)
- Modèles spécifiques par pays pour capturer les particularités locales

---

## 11. Références

1. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785–794. https://doi.org/10.1145/2939672.2939785

2. **Friedman, J.H. (2001).** Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.

3. **Chawla, N.V. et al. (2002).** SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

4. **Lundberg, S.M., & Lee, S.I. (2017).** A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.

5. **The DHS Program (2023).** Demographic and Health Surveys. USAID. https://dhsprogram.com

6. **OMS (2021).** Recommandations de l'OMS concernant les soins prénatals pour que la grossesse soit une expérience positive. Organisation Mondiale de la Santé.

7. **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45, 5–32.

---

## 12. Auteurs

| Nom | Institution | Rôle |
|-----|-------------|------|
| **Groupe 3** | AIMS — African Institute for Mathematical Sciences | Développement ML & Application |
| **Auteur du code** || **GOGAING FOGUEN BAUDOUIN LEGRAND** |

---

### 📄 Licence

Ce projet est développé dans un cadre académique à des fins de recherche et d'éducation au sein de l'école AIMS dans le projet *AMAX*.
Les données DHS sont soumises aux conditions d'utilisation du programme DHS (USAID).

---

> *"Ce tableau de bord est un outil d'aide à la décision clinique, non un dispositif médical certifié.
> Toute décision médicale doit être prise par un professionnel de santé qualifié."*

---

**CésarienneAI** · AIMS / Groupe 3 · Afrique Subsaharienne · 2024
