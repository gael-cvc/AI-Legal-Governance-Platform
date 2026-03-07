"""
models.py — Schémas Pydantic pour la validation et la sérialisation des données

POURQUOI PYDANTIC ?
FastAPI s'appuie entièrement sur Pydantic pour deux responsabilités critiques :

  1. VALIDATION DES ENTRÉES (requêtes HTTP)
     Quand un client envoie un POST /search, le JSON reçu est automatiquement
     validé contre SearchRequest. Si un champ requis est absent, si un type
     est incorrect (ex: k="beaucoup" au lieu de k=5), ou si une contrainte
     n'est pas respectée (ex: k=50 > max 20), FastAPI retourne automatiquement
     une erreur 422 Unprocessable Entity avec le détail exact du problème.
     Pas besoin d'écrire une seule ligne de validation manuelle.

  2. SÉRIALISATION DES SORTIES (réponses HTTP)
     Les objets Python retournés par nos endpoints sont automatiquement
     convertis en JSON selon le schéma de réponse défini (ex: SearchResponse).
     Pydantic s'assure que tous les champs sont présents et du bon type.

DOCUMENTATION AUTOMATIQUE :
  FastAPI génère le Swagger UI (http://localhost:8000/docs) directement depuis
  ces schémas. Les Field(description=...) et Field(examples=[...]) apparaissent
  dans l'interface interactive — le client peut tester l'API sans écrire de code.

STRUCTURE DES SCHÉMAS :
  ┌─ Enums ──────────────────────────────────────────────────────────────┐
  │  Regulation         → filtre par réglementation (GDPR, EU_AI_ACT...) │
  │  ResponseLanguage   → langue de la réponse LLM (fr, en)              │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Request ────────────────────────────────────────────────────────────┐
  │  SearchRequest      → corps du POST /search (question + options)     │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Response ───────────────────────────────────────────────────────────┐
  │  ChunkResult        → un chunk juridique (texte + métadonnées)       │
  │  SearchResponse     → réponse complète (LLM + sources + méta)        │
  │  HealthResponse     → statut API + stats FAISS                       │
  │  SuggestionsResponse→ liste de questions suggérées                   │
  └──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class Regulation(str, Enum):
    """
    Réglementations disponibles dans le corpus vectoriel.

    POURQUOI str + Enum (et pas juste Enum) ?
    En héritant de str, chaque membre de l'enum EST une string Python.
    Cela signifie que la valeur JSON est directement la string (ex: "GDPR")
    et non un entier ou un objet. Comparer :

        class Regulation(Enum):        # mauvais — JSON: {"regulation": "Regulation.GDPR"}
            GDPR = "GDPR"

        class Regulation(str, Enum):   # bien — JSON: {"regulation": "GDPR"}
            GDPR = "GDPR"

    C'est aussi ce qui permet à Pydantic d'accepter indifféremment
    Regulation.GDPR ou la string "GDPR" comme valeur valide dans SearchRequest.

    CES VALEURS CORRESPONDENT AUX CHAMPS "regulation" DES MÉTADONNÉES :
    Quand on fait request.regulation.value, on obtient la string exacte
    utilisée comme clé dans les métadonnées de chaque chunk FAISS.
    """
    GDPR             = "GDPR"
    EU_AI_ACT        = "EU_AI_ACT"
    CNIL             = "CNIL"
    EDPB             = "EDPB"
    DATA_GOVERNANCE  = "DATA_GOVERNANCE_ACT"


class SegmentType(str, Enum):
    """
    Type de segment juridique dans le corpus.

    Permet de restreindre la recherche à un type de texte précis :
    - ARTICLE  : texte normatif officiel (obligations, droits, interdictions)
    - RECITAL  : considérant interprétatif (contexte, intentions du législateur)
    - ANNEX    : annexe technique (listes, critères, formulaires)
    - FREETEXT : texte libre sans structure article (guidelines CNIL, EDPB)

    EXEMPLE D'USAGE :
    Un juriste qui veut uniquement les obligations légales formelles
    utilisera segment_type=ARTICLE pour exclure les recitals interprétatifs.
    Un analyste cherchant le contexte législatif utilisera RECITAL.
    """
    ARTICLE  = "article"
    RECITAL  = "recital"
    ANNEX    = "annex"
    FREETEXT = "freetext"


class ResponseLanguage(str, Enum):
    """
    Langue dans laquelle le LLM doit synthétiser sa réponse.

    Note : cela ne filtre pas les chunks — la recherche vectorielle est
    multilingue (all-MiniLM-L6-v2 gère l'anglais et le français).
    Seule la réponse générée par Claude change de langue.
    Les chunks sources sont toujours retournés dans leur langue d'origine.
    """
    FRENCH  = "fr"
    ENGLISH = "en"


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    """
    Schéma de validation du corps JSON envoyé à POST /search.

    FIELD(...) vs FIELD(default) :
      Field(...) = champ OBLIGATOIRE (le ... est la convention Pydantic pour "requis")
      Field(default=X) = champ OPTIONNEL avec valeur par défaut X

    CONTRAINTES DE VALIDATION :
      min_length, max_length → longueur de la string
      ge (greater or equal), le (lower or equal) → bornes numériques
      Ces contraintes sont vérifiées automatiquement par Pydantic avant
      que notre code soit exécuté. Si k=50, FastAPI retourne 422 avant
      même d'appeler notre fonction search().

    EXEMPLE DE REQUÊTE JSON MINIMALE (seul "question" est requis) :
      {
        "question": "What are the obligations of a data controller?"
      }

    EXEMPLE DE REQUÊTE JSON COMPLÈTE :
      {
        "question": "What are the obligations of a data controller?",
        "regulation": "GDPR",
        "k": 5,
        "language": "fr",
        "min_score": 0.4,
        "use_query_expansion": true,
        "use_reranking": true
      }
    """

    question: str = Field(
        ...,  # obligatoire
        min_length=10,    # évite les questions vides ou trop courtes ("?" passe pas)
        max_length=1000,  # évite les abus (prompt injection longue, etc.)
        description="Question juridique en langage naturel (min 10 caractères)",
        examples=["What are the obligations of a data controller under GDPR?"],
    )

    regulation: Optional[Regulation] = Field(
        default=None,
        description=(
            "Restreindre la recherche à une seule réglementation. "
            "None = cherche dans tout le corpus (GDPR + EU AI Act + CNIL + EDPB + DGA). "
            "Valeurs acceptées : GDPR, EU_AI_ACT, CNIL, EDPB, DATA_GOVERNANCE_ACT"
        ),
    )

    segment_type: Optional[SegmentType] = Field(
        default=None,
        description=(
            "Restreindre la recherche à un type de segment juridique. "
            "None = cherche dans tous les types. "
            "ARTICLE = texte normatif officiel uniquement. "
            "RECITAL = considérants interprétatifs uniquement. "
            "ANNEX = annexes techniques. "
            "FREETEXT = guidelines sans structure article (CNIL, EDPB)."
        ),
    )

    article_number: Optional[int] = Field(
        default=None,
        ge=1,
        le=999,
        description=(
            "Restreindre la recherche à un numéro d'article précis. "
            "None = pas de filtre par numéro. "
            "Ex: article_number=5 → retourne uniquement les chunks de l'Article 5. "
            "À combiner avec regulation pour un résultat précis : "
            "regulation=GDPR + article_number=5 = Article 5 GDPR uniquement."
        ),
    )

    language_filter: Optional[str] = Field(
        default=None,
        description=(
            "Restreindre la recherche aux chunks d'une langue source précise. "
            "None = cherche dans toutes les langues du corpus (EN + FR). "
            "Valeurs acceptées : 'en' (anglais), 'fr' (français). "
            "Exemple : language_filter='fr' retourne uniquement les chunks "
            "issus du RGPD FR, des recommandations CNIL en français, etc."
        ),
    )

    k: int = Field(
        default=5,
        ge=1,   # minimum 1 chunk
        le=20,  # maximum 20 chunks (au-delà, le contexte LLM serait trop long)
        description=(
            "Nombre de chunks à retourner dans les sources. "
            "En interne, on cherche k*2 candidats avant reranking, "
            "puis on garde les k meilleurs après reranking."
        ),
    )

    language: ResponseLanguage = Field(
        default=ResponseLanguage.FRENCH,
        description="Langue de la réponse synthétisée par le LLM (fr ou en). "
                    "N'affecte pas les chunks sources retournés.",
    )

    min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Seuil de similarité cosinus minimum (entre 0 et 1). "
            "Un chunk avec score < min_score est filtré même si FAISS le retourne. "
            "0.35 = seuil calibré sur notre corpus (voir search_test.py). "
            "Baisser si peu de résultats, monter pour plus de précision."
        ),
    )

    use_query_expansion: bool = Field(
        default=True,
        description=(
            "Activer l'expansion de requête (solution 1/3 d'amélioration). "
            "La question est enrichie avec des synonymes juridiques spécifiques "
            "avant d'être encodée en vecteur. Améliore les scores de ~10-15%."
        ),
    )

    use_reranking: bool = Field(
        default=True,
        description=(
            "Activer le reranking cross-encoder (solution 2/3). "
            "Après la recherche vectorielle, un modèle plus précis re-score "
            "chaque paire (question, chunk) ensemble. "
            "Ajoute ~100-200ms mais améliore significativement la pertinence."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ChunkResult(BaseModel):
    """
    Représente un chunk juridique retourné dans les sources de la réponse.

    Un chunk = une unité de texte issue de la couche silver (après chunking
    sémantique du pipeline d'ingestion). Chaque chunk contient :
    - Le texte brut du passage juridique
    - Les métadonnées du document source (règlement, article, année, page...)
    - Les scores de pertinence (similarité vectorielle + éventuellement reranking)

    SIMILARITY_SCORE vs RERANK_SCORE :
      similarity_score = similarité cosinus entre le vecteur de la question
                         et le vecteur du chunk (produit scalaire normalisé).
                         Calculé par FAISS. Entre 0 et 1.

      rerank_score     = score du cross-encoder (si reranking activé).
                         Mesure à quel point la question et le chunk sont
                         pertinents l'un par rapport à l'autre en les
                         lisant ensemble. Non normalisé (peut être négatif).
                         C'est ce score qui détermine l'ordre final.

    Optional[float] pour rerank_score car ce champ est None si use_reranking=False.
    """

    segment_id:       str    # ex: "GDPR_ART_5" — identifiant unique du segment source
    segment_type:     str    # "article", "recital", "section", "annex"...
    regulation:       str    # "GDPR", "EU_AI_ACT", "CNIL"...
    official_title:   str    # titre officiel du document source
    year:             int    # année de publication (ex: 2016 pour GDPR)
    jurisdiction:     str    # "EU", "FR"...
    source_file:      str    # nom du fichier PDF source
    page_start:       int    # numéro de la page dans le PDF original
    text:             str    # texte du chunk (max ~1200 chars selon chunker.py)
    similarity_score: float  # score FAISS (0.0 à 1.0, seuil min_score=0.35)
    rerank_score:     Optional[float] = None  # score cross-encoder (None si désactivé)


class SearchResponse(BaseModel):
    """
    Réponse complète de POST /search.

    STRUCTURE LOGIQUE :
      ┌─ Réponse LLM ───────────────────────────────────────────────────┐
      │  answer     → texte synthétisé avec citations [SOURCE X]        │
      │  citations  → liste dédupliquée des sources citées              │
      └─────────────────────────────────────────────────────────────────┘
      ┌─ Sources brutes ────────────────────────────────────────────────┐
      │  sources    → liste des ChunkResult utilisés par le LLM         │
      └─────────────────────────────────────────────────────────────────┘
      ┌─ Métadonnées de la requête ─────────────────────────────────────┐
      │  question            → question originale de l'utilisateur      │
      │  regulation_filter   → filtre appliqué (None si tout le corpus) │
      │  n_chunks_retrieved  → chunks retournés par FAISS               │
      │  n_chunks_used       → chunks après reranking (≤ retrieved)     │
      │  query_expanded      → bool, si l'expansion a été déclenchée    │
      │  expanded_query      → question reformulée (None si pas d'exp.) │
      │  processing_time_ms  → latence totale (expansion+FAISS+rank+LLM)│
      │  model_used          → modèle LLM utilisé (ou "fallback-no-llm")│
      └─────────────────────────────────────────────────────────────────┘

    POURQUOI INCLURE LES MÉTADONNÉES ?
    Elles permettent au client de :
    - Afficher le temps de traitement pour l'UX
    - Débugger si la qualité est faible (ex: query_expanded=False signale
      qu'aucun mot-clé n'a matché le dictionnaire d'expansion)
    - Voir si le filtre réglementation a réduit les résultats
    """

    # Réponse LLM
    answer:    str        # réponse synthétisée en langage naturel avec [SOURCE X]
    citations: list[str]  # ex: ["GDPR Article 5 (2016)", "EU_AI_ACT Recital 47 (2024)"]

    # Chunks sources
    sources: list[ChunkResult]

    # Métadonnées de la requête
    question:            str
    regulation_filter:   Optional[str]   # "GDPR" ou None
    n_chunks_retrieved:  int             # combien FAISS a retourné
    n_chunks_used:       int             # combien ont été envoyés au LLM (après reranking)
    query_expanded:      bool
    expanded_query:      Optional[str]   # question enrichie (None si pas d'expansion)
    processing_time_ms:  float
    model_used:          str             # "claude-sonnet-4-20250514" ou "fallback-no-llm"


class HealthResponse(BaseModel):
    """
    Réponse de GET /api/v1/health.

    CE QU'UN HEALTH CHECK DOIT RETOURNER :
    - status : "ok" si tout fonctionne, "degraded" si l'index n'est pas chargé.
      Les load balancers et systèmes de monitoring se basent sur ce champ.
    - Les stats de l'index pour confirmer que les données sont bien chargées.
    - L'uptime pour savoir depuis combien de temps l'API tourne
      (utile pour détecter des redémarrages inattendus).

    STATUS "degraded" (pas d'erreur 500) :
    On retourne quand même 200 avec status="degraded" si l'index FAISS
    n'est pas chargé. Pourquoi ne pas retourner 503 ?
    Parce que l'API elle-même fonctionne — c'est juste la donnée qui manque.
    Le 503 est retourné par l'endpoint /search si on tente une recherche
    sans index. Séparer les deux cas rend le monitoring plus clair.
    """

    status:         str        # "ok" ou "degraded"
    version:        str        # version du package api (ex: "1.0.0")
    index_loaded:   bool       # True si l'index FAISS est en mémoire
    n_vectors:      int        # nombre de vecteurs dans l'index (ex: 2016)
    n_regulations:  int        # nombre de réglementations distinctes indexées
    regulations:    list[str]  # liste des réglementations (ex: ["CNIL", "EDPB", "GDPR"...])
    uptime_seconds: float      # secondes depuis le démarrage de l'API


class SuggestionsResponse(BaseModel):
    """
    Réponse de GET /api/v1/search/suggestions.

    Fournit des questions d'exemple prêtes à l'emploi.

    UTILISATION FRONTEND :
    Le frontend peut appeler cet endpoint au chargement de la page pour
    afficher des boutons "Questions suggérées" cliquables. Quand l'utilisateur
    clique sur une suggestion, elle est pré-remplie dans le champ de recherche.
    Améliore l'UX en guidant les utilisateurs qui ne savent pas quoi demander.

    Si regulation=None → retourne un échantillon de toutes les réglementations.
    Si regulation="GDPR" → retourne uniquement les suggestions GDPR.
    """

    regulation:  Optional[str]  # réglementation filtrée, ou None si toutes
    suggestions: list[str]      # liste des questions suggérées
