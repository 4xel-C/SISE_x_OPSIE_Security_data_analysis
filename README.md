# SecurityView — Analyse de logs réseau pour la cybersécurité

> Projet académique **SISE × OPSIE 2026** — Détection d'intrusion et monitoring réseau par analyse de données firewall

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Mistral AI](https://img.shields.io/badge/LLM-Mistral_AI-FF7000?style=flat-square&logo=mistral&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/Packaging-uv-DE5FE9?style=flat-square)
![Challenge](https://img.shields.io/badge/Challenge-48h-red?style=flat-square&logo=lightning&logoColor=white)

---

<img width="2178" height="1348" alt="image" src="https://github.com/user-attachments/assets/92009eb0-5ab9-4df3-b31a-cbe401ae3a3a" />


## Contexte

**SecurityView** a été développé dans le cadre d'un **challenge de 48 heures** organisé conjointement par les masters **SISE** (Statistique et Informatique pour la Science des donnEes) et **OPSIE** (Optimisation et Pilotage des Systèmes d'Information en Entreprise). Le projet a été conçu, développé et livré sous contrainte de temps forte, avec l'objectif de produire une solution fonctionnelle et complète en deux jours.

Les équipes de sécurité (SOC) sont confrontées à des volumes massifs de logs réseau qu'il est impossible d'analyser manuellement. **SecurityView** automatise la détection d'IPs suspectes, identifie des patterns d'attaque et assiste les analystes SOC dans leur travail de monitoring et de réponse aux incidents.

L'application ingère des logs de firewall bruts, les enrichit par feature engineering, puis les soumet à plusieurs couches d'analyse :

- **Visualisation** des comportements réseau et indicateurs de compromission
- **Clustering** pour identifier des groupes d'IPs au comportement homogène (scanners, bots, attaquants ciblés…)
- **Modèles supervisés et non-supervisés** pour la détection automatique d'anomalies
- **Analyse LLM** (Mistral AI) pour interpréter les résultats et générer des conclusions exploitables par un SOC

---

## Pipeline d'analyse

### 1. Parsing et feature engineering

Les logs bruts (CSV, provenant d'un firewall) sont agrégés par IP source. Chaque IP est caractérisée par **16 features** calculées automatiquement :

| Feature | Description |
|---|---|
| `deny_rate` | Ratio de requêtes bloquées — signal principal d'une IP hostile |
| `unique_dst_ratio` | Ratio d'IPs destinations uniques — détecte le **scan horizontal** |
| `unique_port_ratio` | Ratio de ports uniques — détecte le **scan vertical** |
| `requests_per_second` | Vélocité du trafic — détecte les comportements automatisés/burst |
| `deny_rules_hit` | Nombre de règles de blocage distinctes touchées |
| `sensitive_ports_ratio` | Ratio d'accès vers ports critiques (SSH, RDP, SMB, MySQL…) |
| `activity_duration_s` | Durée d'activité en secondes |
| `access_nbr` | Nombre total de requêtes |
| `distinct_ipdst` | Nombre d'IPs de destination uniques |
| `distinct_portdst` | Nombre de ports de destination uniques |
| `permit_nbr` / `deny_nbr` | Nombre d'actions autorisées / bloquées |
| `most_triggered_rule` | Règle firewall la plus déclenchée |
| `sensitive_ports_nbr` | Nombre d'accès vers ports sensibles |

Les IPs sont également **géolocalisées** (ville, pays, coordonnées) via l'API ip-api.com pour la cartographie.

### 2. Visualisation

Page d'exploration du trafic réseau avec une vingtaine de graphiques interactifs :
- Top IPs sources par volume et taux de blocage
- Distribution des ports et protocoles
- Timeline deny/permit pour identifier les pics d'attaque
- Détection de scans horizontaux vs verticaux
- Heatmap géographique des sources d'attaque
- Classement des règles firewall les plus déclenchées

### 3. Clustering non-supervisé

Exploration des groupes comportementaux avec 4 algorithmes au choix :

| Algorithme | Usage |
|---|---|
| **KMeans** | Segmentation en k groupes homogènes |
| **CAH** (Clustering Agglomératif Hiérarchique) | Dendrogramme, pas besoin de fixer k |
| **HDBSCAN** | Détection de clusters de densité variable + outliers |
| **Isolation Forest** | Détection d'anomalies par isolation |

La projection dimensionnelle (PCA ou UMAP) permet de visualiser les clusters en 2D/3D. Un commentaire LLM interprète automatiquement les groupes détectés.

<img width="2142" height="1302" alt="image" src="https://github.com/user-attachments/assets/5795611b-b1cf-438d-840b-6604f1e4d673" />


### 4. Pipeline MCP — Analyse IA en 4 étapes

Pipeline séquentiel assisté par **Mistral AI** pour une analyse complète orientée SOC :

```
Étape 1 — Analyse descriptive
  └─ Statistiques globales, top IPs, corrélations, commentaire narratif

Étape 2 — Modèle supervisé
  └─ Random Forest ou Régression Logistique (modèles pré-entraînés)
  └─ Classification Normal / Suspect pour chaque IP

Étape 3 — Modèle non-supervisé
  └─ Isolation Forest ou LOF avec optimisation automatique des hyperparamètres
  └─ Détection d'outliers avec scoring d'anomalie

Étape 4 — Consolidation SOC
  └─ Union des IPs suspectes des deux modèles
  └─ Priorisation des IPs flaggées par les deux approches
  └─ Conclusion SOC + recommandations (blocage, investigation, escalade)
  └─ Export PDF du rapport complet
```

---

## Stack technique

| Catégorie | Technologies |
|---|---|
| Interface | Streamlit |
| Data | Pandas, NumPy, PyArrow |
| Machine Learning | Scikit-learn, HDBSCAN, UMAP-learn |
| Visualisation | Plotly, Folium, Seaborn, Matplotlib |
| LLM | Mistral AI (`mistral-small-latest`) |
| Base de données | SQLAlchemy, PyMySQL (Aiven MySQL / MariaDB SkySQL) |
| Export | fpdf2 (PDF avec polices Unicode DejaVu) |
| Packaging | uv, pyproject.toml |
| Déploiement | Docker |

---

## Installation

### Avec Docker (recommandé)

**1. Cloner le projet**
```bash
git clone <url-du-repo>
cd SISE_x_OPSIE_Security_data_analysis
```

**2. Configurer les variables d'environnement**
```bash
cp .env.example .env
# Remplir les valeurs dans .env
```

**3. Build de l'image**
```bash
docker build -t securityview .
```

**4. Lancer le container**
```bash
docker run -p 8501:8501 securityview
```

L'application est accessible sur [http://localhost:8501](http://localhost:8501).

---

### Sans Docker (développement local)

```bash
# Installer les dépendances
uv sync

# Lancer l'application
uv run streamlit run app.py
```

> **Prérequis** : Python 3.13+ et [uv](https://github.com/astral-sh/uv) installé.

---

## Configuration

Copier `.env.example` en `.env` et renseigner les valeurs :

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `AIVEN_HOST` / `AIVEN_PORT` | Connexion au service MySQL Aiven |
| `AIVEN_USER` / `AIVEN_PASSWORD` / `AIVEN_DB` | Credentials Aiven |
| `AIVEN_SSL_CA` | Chemin vers le certificat CA Aiven |
| `MARIADB_*` | Connexion MariaDB SkySQL (alternative) |
| `DATAFILE` | Nom du fichier CSV de logs à analyser |
| `MISTRAL_API_KEY` | Clé API Mistral AI |

---

## Structure du projet

```
.
├── app.py                      # Point d'entrée Streamlit + navigation
├── pages/
│   ├── visualisation.py        # Graphiques et heatmap géographique
│   ├── clustering.py           # Clustering interactif (KMeans, CAH, HDBSCAN, IF)
│   ├── prediction.py           # Modèles supervisés
│   └── mcp.py                  # Pipeline d'analyse SOC en 4 étapes
├── features/
│   ├── parser.py               # Agrégation par IP + feature engineering
│   └── clustering.py           # Algorithmes de clustering
├── services/
│   ├── data_manager.py         # Chargement des données (singleton)
│   ├── charts.py               # Génération de tous les graphiques
│   ├── clustering_service.py   # Orchestration du clustering
│   ├── analysis_pipeline.py    # Pipeline MCP (4 étapes + export PDF)
│   └── mistral_client.py       # Client Mistral AI
├── models/                     # Modèles pré-entraînés (pickle)
│   ├── rf_classifier.pkl
│   └── logistic_regression.pkl
├── data/                       # Fichiers de logs CSV
├── assets/
│   └── fonts/                  # Polices DejaVu pour l'export PDF
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Données attendues

L'application attend un fichier CSV de logs firewall avec au minimum les colonnes suivantes :

| Colonne | Description |
|---|---|
| `ipsrc` | IP source |
| `ipdst` | IP destination |
| `portdst` | Port de destination |
| `action` | Action firewall (`Permit` / `Deny`) |
| `regle` | Règle firewall déclenchée |
| `date` | Timestamp de l'événement |

Le nom du fichier est configuré via la variable `DATAFILE` dans `.env`.
