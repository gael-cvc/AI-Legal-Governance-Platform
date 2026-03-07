"""
search.py — Pipeline RAG complet : query expansion → FAISS → reranking → LLM

QU'EST-CE QUE LE RAG (Retrieval Augmented Generation) ?
  Le RAG est une architecture qui combine deux approches :
  - La RECHERCHE (retrieval) : trouver les passages pertinents dans un corpus
  - La GÉNÉRATION : produire une réponse en langage naturel à partir de ces passages

  Sans RAG, un LLM comme Claude répond depuis sa mémoire d'entraînement :
  les informations peuvent être périmées, imprécises, ou hallusinées.
  Avec RAG, le LLM est contraint de répondre UNIQUEMENT depuis les textes
  qu'on lui fournit — les lois officielles dans notre cas.

  C'est particulièrement critique en droit : une citation incorrecte d'article
  peut avoir des conséquences juridiques. Le RAG garantit la traçabilité.

PIPELINE COMPLET D'UNE REQUÊTE /search :

  Question utilisateur
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ÉTAPE 1 : QUERY EXPANSION                              │
  │  Enrichit la question avec des synonymes juridiques.    │
  │  "data minimisation" → "data minimisation Article       │
  │  5(1)(c) personal data collection limited necessary"    │
  │  → guide le vecteur vers le bon espace sémantique       │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ÉTAPE 2 : RECHERCHE VECTORIELLE FAISS                  │
  │  Encode la question en vecteur 384D.                    │
  │  FAISS retourne les k*2 chunks les plus proches         │
  │  par similarité cosinus dans l'index de 2016 vecteurs.  │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ÉTAPE 3 : RERANKING CROSS-ENCODER                      │
  │  Un second modèle re-score chaque (question, chunk)     │
  │  en les lisant ENSEMBLE — bien plus précis qu'un        │
  │  simple produit scalaire entre vecteurs séparés.        │
  │  → élimine les faux positifs, remonte les vrais matches │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ÉTAPE 4 : GÉNÉRATION LLM (Claude)                      │
  │  Les top chunks rerankés sont injectés dans un prompt.  │
  │  Claude synthétise une réponse juridique précise        │
  │  avec citations [SOURCE X] obligatoires.                │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  Réponse JSON structurée (SearchResponse)

CHANGEMENT v1.1 — RERANKER SINGLETON :
  Avant : CrossEncoder() était instancié DANS rerank_chunks() à chaque requête.
          → chargement du modèle (~1s) + allocation mémoire répétée → crash Mac M4
  Après : LegalReranker est chargé UNE SEULE FOIS au démarrage dans main.py lifespan.
          rerank_chunks() reçoit l'instance en paramètre → aucun rechargement.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from .models import (
    ChunkResult,
    Regulation,
    SearchRequest,
    SearchResponse,
    SuggestionsResponse,
)

logger = logging.getLogger("api.search")
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

# PROBLÈME QUE L'EXPANSION RÉSOUT :
# Le modèle all-MiniLM-L6-v2 est un modèle généraliste entraîné sur du texte
# internet (Wikipedia, news, forums...). Il ne connaît pas les correspondances
# spécifiques du droit européen :
#
#   "data minimisation" → il ne sait pas que c'est l'Article 5(1)(c) GDPR
#   "data controller"   → il ne sait pas que c'est l'Article 24 GDPR
#   "high-risk AI"      → il ne sait pas que c'est l'Annexe III EU AI Act
#
# Conséquence : la question "data minimisation" produit un vecteur qui
# ressemble à "réduire les données" en général, pas au texte précis de l'art. 5.
#
# SOLUTION : avant d'encoder, on ajoute les termes techniques qui apparaissent
# dans le texte source. La question expansée génère un vecteur plus proche
# des chunks qui contiennent réellement ces termes juridiques précis.
#
# RÉSULTAT MESURÉ : amélioration des scores de similarité de ~10-15%.

QUERY_EXPANSIONS: dict[str, str] = {
    # ── GDPR : principes fondamentaux ─────────────────────────────────────────
    "data controller": (
        "data controller obligations responsibilities Article 24 GDPR "
        "implement appropriate technical organisational measures"
    ),
    "data minimisation": (
        "data minimisation Article 5(1)(c) GDPR personal data adequate "
        "relevant limited necessary purpose collection"
    ),
    "purpose limitation": (
        "purpose limitation Article 5(1)(b) GDPR collected specified explicit "
        "legitimate purposes not further processed incompatible"
    ),
    "lawful basis": (
        "lawful basis processing Article 6 GDPR consent contract legal obligation "
        "vital interests public task legitimate interest"
    ),
    "consent": (
        "consent Article 7 GDPR freely given specific informed unambiguous "
        "indication data subject withdrawable"
    ),

    # ── GDPR : droits des personnes ───────────────────────────────────────────
    "data subject rights": (
        "data subject rights GDPR Article 15 access Article 16 rectification "
        "Article 17 erasure Article 18 restriction Article 20 portability"
    ),
    "right to erasure": (
        "right to erasure right to be forgotten Article 17 GDPR delete personal "
        "data no longer necessary withdrawn consent"
    ),
    "automated decision": (
        "automated decision making Article 22 GDPR solely automated processing "
        "profiling legal significant effects right to object human review"
    ),
    "profiling": (
        "profiling Article 4(4) GDPR Article 22 automated processing personal "
        "data evaluate analyse predict behaviour"
    ),

    # ── GDPR : obligations organisationnelles ─────────────────────────────────
    "dpo": (
        "data protection officer DPO Article 37 designation mandatory Article 38 "
        "position Article 39 tasks advise monitor cooperate"
    ),
    "dpia": (
        "data protection impact assessment DPIA Article 35 GDPR systematic "
        "description purposes legitimate interests high risk prior consultation"
    ),
    "data breach": (
        "personal data breach notification Article 33 supervisory authority 72 hours "
        "Article 34 communication data subjects risk rights freedoms"
    ),
    "special categories": (
        "special categories personal data Article 9 GDPR sensitive health genetic "
        "biometric racial ethnic political religious processing prohibition"
    ),
    "data transfer": (
        "international transfer personal data third country Article 44 adequacy "
        "decision Article 45 Article 46 appropriate safeguards standard clauses"
    ),

    # ── EU AI Act ─────────────────────────────────────────────────────────────
    "high risk ai": (
        "high-risk AI systems Annex III Article 9 risk management Article 10 "
        "data governance Article 13 transparency Article 14 human oversight "
        "conformity assessment CE marking"
    ),
    "prohibited ai": (
        "prohibited AI practices Article 5 AI Act unacceptable risk "
        "subliminal manipulation vulnerability exploitation social scoring "
        "real-time biometric identification"
    ),
    "general purpose ai": (
        "general purpose AI models GPAI Article 51 systemic risk Article 52 "
        "transparency obligations Article 53 providers large language models"
    ),
    "transparency ai": (
        "transparency obligations Article 13 AI Act information users "
        "intended purpose capabilities limitations explainability"
    ),
    "conformity": (
        "conformity assessment Article 43 AI Act notified body CE marking "
        "declaration conformity harmonised standards technical documentation"
    ),

    # ── EDPB / CNIL ───────────────────────────────────────────────────────────
    "recommendation cnil": (
        "CNIL recommendations guidelines AI systems training data personal "
        "data protection privacy impact assessment"
    ),
    "edpb guidelines": (
        "EDPB European Data Protection Board guidelines recommendations "
        "opinions consistency mechanism adequacy decisions"
    ),
}


def expand_query(question: str) -> tuple[str, bool]:
    """
    Enrichit la question de l'utilisateur avec des termes juridiques spécifiques.

    ALGORITHME :
    1. On convertit la question en minuscules (comparaison insensible à la casse)
    2. On itère sur chaque mot-clé du dictionnaire QUERY_EXPANSIONS
    3. Si le mot-clé est trouvé dans la question, on ajoute le texte d'expansion
    4. Plusieurs mots-clés peuvent matcher → leurs expansions s'accumulent
    5. On retourne la question enrichie + un bool indiquant si une expansion a eu lieu

    EXEMPLE :
    Question : "What are the obligations for data minimisation under GDPR?"
    Match     : "data minimisation"
    Résultat  : question originale + "data minimisation Article 5(1)(c) GDPR..."

    LIMITES CONNUES :
    - Le dictionnaire est manuel → ne couvre pas tous les termes possibles
    - Solution future : expansion basée sur un modèle (BM25, graph de concepts)

    Retourne : (question_expansée, a_été_expansée)
    """
    question_lower   = question.lower()
    expansions_found: list[str] = []

    for keyword, expansion_text in QUERY_EXPANSIONS.items():
        if keyword in question_lower:
            expansions_found.append(expansion_text)
            logger.debug(f"  → match: '{keyword}'")

    if not expansions_found:
        return question, False

    expanded = question + " " + " ".join(expansions_found)
    logger.info(f"Query expansion : {len(expansions_found)} terme(s) ajouté(s)")
    return expanded, True


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : RERANKING
# ══════════════════════════════════════════════════════════════════════════════

def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_k: int,
    reranker,  # instance LegalReranker passée depuis le endpoint
) -> list[dict]:
    """
    Re-score et réordonne les chunks avec le cross-encoder singleton.

    CHANGEMENT v1.1 PAR RAPPORT À L'ANCIENNE VERSION :
    Avant : CrossEncoder() était instancié ici à chaque appel → crash Mac M4
            car PyTorch alloue et libère de la mémoire GPU/CPU à chaque fois.
    Après : on reçoit l'instance `reranker` déjà chargée depuis main.py.
            L'allocation mémoire se fait UNE SEULE FOIS au démarrage.

    DÉLÉGATION À LegalReranker.rerank() :
    Toute la logique de scoring est dans rag/reranker.py.
    Cette fonction est un simple wrapper qui :
    1. Vérifie si le reranker est disponible
    2. Délègue à reranker.rerank()
    3. Applique le fallback ordre FAISS si nécessaire

    POURQUOI LA QUESTION ORIGINALE (PAS L'EXPANSÉE) POUR LE RERANKING ?
    L'expansion sert à guider le vecteur FAISS vers les bons chunks.
    Le cross-encoder lit les textes directement — lui passer la question
    concise est plus efficace car il cherche des correspondances directes.

    Paramètres :
        question : question originale (non expansée)
        chunks   : candidats FAISS à re-scorer
        top_k    : nombre de chunks à retourner
        reranker : instance LegalReranker (peut être None si chargement échoué)

    Retourne :
        list[dict] rerankés par score décroissant, ou ordre FAISS si fallback
    """
    if reranker is None or not reranker.is_available:
        # Reranker non disponible → fallback ordre FAISS
        logger.warning("Reranker non disponible → fallback ordre FAISS")
        return chunks[:top_k]

    # Délègue à LegalReranker.rerank() — gère lui-même les erreurs et fallbacks
    return reranker.rerank(question=question, chunks=chunks, top_k=top_k)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : GÉNÉRATION LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, chunks: list[dict], language: str = "fr") -> str:
    """
    Construit le prompt structuré envoyé au LLM.

    PHILOSOPHIE DU PROMPT RAG :
    Un bon prompt RAG doit résoudre une tension fondamentale :
    - Le LLM a ses propres "connaissances" acquises à l'entraînement
    - On veut qu'il réponde UNIQUEMENT depuis les sources fournies

    Pour forcer cette contrainte, on :
    1. Donne les sources en premier (avant la question)
    2. Utilise des instructions explicites ("EXCLUSIVEMENT", "RÈGLES ABSOLUES")
    3. Demande des citations obligatoires [SOURCE X] → traçabilité totale
    4. Demande de signaler si les sources sont insuffisantes → honnêteté

    NUMÉROTATION DES SOURCES :
    Chaque source est préfixée [SOURCE 1], [SOURCE 2]... Le LLM est instruit
    de citer avec ce numéro dans sa réponse.

    LANGUE :
    Le prompt change selon request.language, mais les chunks restent en anglais.
    Claude gère très bien : comprendre des sources anglaises et répondre en français.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        reg    = chunk.get("regulation", "?")
        seg_id = chunk.get("segment_id", "?")
        year   = chunk.get("year", "?")
        text   = chunk.get("text", "")
        context_parts.append(f"[SOURCE {i}] {reg} ({year}) — {seg_id}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    if language == "fr":
        instructions = (
            "Tu es un expert juridique spécialisé en droit européen de la protection "
            "des données et de l'intelligence artificielle.\n\n"
            "En utilisant UNIQUEMENT les sources numérotées ci-dessous, réponds "
            "à la question de manière précise et structurée.\n\n"
            "RÈGLES ABSOLUES :\n"
            "- Base ta réponse EXCLUSIVEMENT sur les sources fournies\n"
            "- Cite chaque affirmation avec [SOURCE X] correspondant au numéro ci-dessus\n"
            "- Si les sources ne permettent pas de répondre complètement, indique-le explicitement\n"
            "- Utilise un langage juridique précis mais accessible\n"
            "- Structure ta réponse avec des paragraphes clairs\n"
            "- À la fin, liste les sources utilisées au format "
            "\"Règlement Article/Recital X (Année)\"\n\n"
            "SOURCES :\n"
        )
    else:
        instructions = (
            "You are a legal expert specialized in European data protection "
            "and artificial intelligence law.\n\n"
            "Using ONLY the numbered sources provided below, answer the question "
            "precisely and in a structured manner.\n\n"
            "ABSOLUTE RULES:\n"
            "- Base your answer EXCLUSIVELY on the provided sources\n"
            "- Cite each claim with [SOURCE X] matching the number above\n"
            "- If the sources are insufficient to answer completely, state this explicitly\n"
            "- Use precise but accessible legal language\n"
            "- Structure your answer with clear paragraphs\n"
            "- At the end, list the sources used as \"Regulation Article/Recital X (Year)\"\n\n"
            "SOURCES:\n"
        )

    question_label = "QUESTION :" if language == "fr" else "QUESTION:"
    return f"{instructions}{context}\n\n{question_label} {question}"


async def generate_answer(
    question: str,
    chunks: list[dict],
    language: str = "fr",
) -> tuple[str, list[str], str]:
    """
    Envoie le prompt au LLM Claude et parse la réponse.

    POURQUOI async ?
    L'appel à l'API Anthropic est une opération I/O (réseau).
    En Python async, pendant qu'on attend la réponse d'Anthropic (~2-3s),
    FastAPI peut traiter d'autres requêtes entrantes en parallèle.

    FALLBACK SANS LLM :
    Si ANTHROPIC_API_KEY n'est pas configurée ou si l'API est down,
    on génère une réponse structurée qui liste les chunks avec leurs scores.
    L'API reste 100% fonctionnelle pour valider la recherche vectorielle.

    Retourne : (texte_réponse, liste_citations, modèle_utilisé)
    """
    import os
    model_id = "claude-sonnet-4-20250514"
    api_key  = os.getenv("ANTHROPIC_API_KEY")

    try:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY absente du .env — mode fallback activé")

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = build_prompt(question, chunks, language)

        # Appel synchrone à l'API Anthropic.
        # max_tokens=2000 : suffisant pour une réponse juridique détaillée.
        message = client.messages.create(
            model      = model_id,
            max_tokens = 2000,
            messages   = [{"role": "user", "content": prompt}],
        )
        answer = message.content[0].text

        # ── Extraction des citations depuis la réponse ─────────────────────────
        # On cherche deux patterns :
        # 1. [SOURCE X]          → références numériques inline
        # 2. GDPR Article 5 (2016) → citations formelles en fin de réponse
        import re
        citation_pattern = (
            r'\[SOURCE \d+\]'
            r'|[A-Z_]+\s+(?:Article|Recital|Section)\s+'
            r'\d+(?:\(\d+\))?(?:\([a-z]\))?\s*\(\d{4}\)'
        )
        raw       = re.findall(citation_pattern, answer)
        citations = list(dict.fromkeys(raw))  # déduplique en préservant l'ordre

    except Exception as e:
        logger.warning(f"Génération LLM échouée : {e}")

        # ── FALLBACK : réponse structurée sans LLM ────────────────────────────
        model_id = "fallback-no-llm"

        if language == "fr":
            header = (
                f"**Résultats de recherche pour :** {question}\n\n"
                f"{len(chunks)} source(s) pertinente(s) identifiée(s). "
                f"Configurez `ANTHROPIC_API_KEY` dans `.env` pour une synthèse intelligente.\n\n"
            )
        else:
            header = (
                f"**Search results for:** {question}\n\n"
                f"{len(chunks)} relevant source(s) found. "
                f"Set `ANTHROPIC_API_KEY` in `.env` for intelligent synthesis.\n\n"
            )

        source_lines = []
        for c in chunks:
            score_info = f"score FAISS: {c.get('similarity_score', 0):.3f}"
            if c.get("rerank_score") is not None:
                score_info += f" | rerank: {c.get('rerank_score'):.3f}"
            source_lines.append(
                f"**{c.get('regulation')} — {c.get('segment_id')}** ({score_info})\n"
                f"{c.get('text', '')[:400]}..."
            )

        answer    = header + "\n\n".join(source_lines)
        citations = [
            f"{c.get('regulation')} {c.get('segment_id')} ({c.get('year')})"
            for c in chunks
        ]

    return answer, citations, model_id


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Recherche juridique intelligente (RAG)",
    description=(
        "Pipeline complet : query expansion → FAISS → reranking → LLM. "
        "Pose une question en langage naturel, reçois une réponse synthétisée "
        "avec citations et les chunks sources. "
        "Corpus : GDPR · EU AI Act · EDPB Guidelines · CNIL · Data Governance Act."
    ),
    tags=["Search"],
)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Orchestre les 4 étapes du pipeline RAG pour répondre à une question juridique.

    GESTION DES ERREURS :
    - 503 : index FAISS non chargé (lancer build_index.py d'abord)
    - 404 : aucun chunk ne dépasse le min_score
    - 422 : validation Pydantic échouée — auto FastAPI

    STRATÉGIE k*2 :
    On demande à FAISS le double de chunks nécessaires (k*2) pour donner
    au cross-encoder assez de candidats à reordonner.
    Ex: k=5 → FAISS retourne 10 → cross-encoder garde les 5 meilleurs.
    """
    # Récupère les singletons chargés au démarrage
    from .main import get_embedder, get_reranker, get_vector_store

    t_start  = time.perf_counter()
    store    = get_vector_store()
    embedder = get_embedder()
    reranker = get_reranker()  # peut être None si chargement échoué

    # ── Vérification de l'état du service ─────────────────────────────────────
    if store is None or store.n_vectors == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "Index FAISS non chargé. "
                "Lancez d'abord : python -m rag.build_index"
            ),
        )

    # ── ÉTAPE 1 : Query Expansion ──────────────────────────────────────────────
    if request.use_query_expansion:
        search_query, was_expanded = expand_query(request.question)
    else:
        search_query, was_expanded = request.question, False

    logger.info(
        f"[SEARCH] question='{request.question[:80]}' | "
        f"regulation={request.regulation} | k={request.k} | "
        f"expanded={was_expanded}"
    )

    # ── ÉTAPE 2 : Recherche vectorielle FAISS ─────────────────────────────────
    # k*2 candidats si reranking activé → marge pour le cross-encoder
    # k candidats si reranking désactivé → pas de marge nécessaire
    search_k     = request.k * 2 if request.use_reranking else request.k
    query_vector = embedder.encode_query(search_query)

    raw_chunks = store.search(
        query_vector      = query_vector,
        k                 = search_k,
        regulation_filter = request.regulation.value if request.regulation else None,
        min_score         = request.min_score,
    )

    n_retrieved = len(raw_chunks)
    logger.info(f"[FAISS] {n_retrieved} chunks récupérés (seuil={request.min_score})")

    if not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Aucun résultat avec score ≥ {request.min_score}. "
                f"Essayez de reformuler la question ou de baisser min_score."
            ),
        )

    # ── ÉTAPE 3 : Reranking ───────────────────────────────────────────────────
    if request.use_reranking and len(raw_chunks) > 1:
        final_chunks = rerank_chunks(
            question = request.question,  # question originale, pas l'expansée
            chunks   = raw_chunks,
            top_k    = request.k,
            reranker = reranker,          # singleton chargé au démarrage
        )
    else:
        final_chunks = raw_chunks[:request.k]

    n_used = len(final_chunks)

    # ── ÉTAPE 4 : Génération LLM ──────────────────────────────────────────────
    answer, citations, model_used = await generate_answer(
        question = request.question,
        chunks   = final_chunks,
        language = request.language.value,
    )

    # ── Construction de la réponse structurée ────────────────────────────────
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    sources = [
        ChunkResult(
            segment_id       = c.get("segment_id", ""),
            segment_type     = c.get("segment_type", ""),
            regulation       = c.get("regulation", ""),
            official_title   = c.get("official_title", ""),
            year             = c.get("year", 0),
            jurisdiction     = c.get("jurisdiction", ""),
            source_file      = c.get("source_file", ""),
            page_start       = c.get("page_start", 0),
            text             = c.get("text", ""),
            similarity_score = c.get("similarity_score", 0.0),
            rerank_score     = c.get("rerank_score"),
        )
        for c in final_chunks
    ]

    logger.info(
        f"[DONE] {elapsed_ms:.0f}ms — "
        f"FAISS: {n_retrieved} → reranking → LLM: {n_used} chunks utilisés"
    )

    return SearchResponse(
        answer             = answer,
        citations          = citations,
        sources            = sources,
        question           = request.question,
        regulation_filter  = request.regulation.value if request.regulation else None,
        n_chunks_retrieved = n_retrieved,
        n_chunks_used      = n_used,
        query_expanded     = was_expanded,
        expanded_query     = search_query if was_expanded else None,
        processing_time_ms = round(elapsed_ms, 1),
        model_used         = model_used,
    )


@router.get(
    "/search/suggestions",
    response_model=SuggestionsResponse,
    summary="Questions suggérées par réglementation",
    description=(
        "Retourne des questions juridiques types pour guider l'utilisateur. "
        "Filtrable par réglementation. "
        "Sans filtre, retourne un échantillon de toutes les réglementations."
    ),
    tags=["Search"],
)
async def get_suggestions(regulation: Optional[Regulation] = None) -> SuggestionsResponse:
    """
    Fournit des questions d'exemple prêtes à l'emploi.

    UTILISATION FRONTEND :
    Le frontend appelle cet endpoint au chargement pour afficher des boutons
    "Questions suggérées". Quand l'utilisateur clique, la question est
    pré-remplie dans le champ de recherche.
    """
    suggestions_by_regulation: dict[str, list[str]] = {
        "GDPR": [
            "What are the obligations of a data controller under GDPR?",
            "What constitutes valid consent under GDPR Article 7?",
            "What are the conditions for lawful processing under Article 6?",
            "What are the rights of data subjects under GDPR?",
            "When is a Data Protection Impact Assessment required?",
            "What are the notification requirements for a personal data breach?",
            "What is the principle of data minimisation under Article 5?",
        ],
        "EU_AI_ACT": [
            "What AI systems are classified as high-risk under the EU AI Act?",
            "What are the prohibited AI practices under Article 5?",
            "What conformity assessment is required for high-risk AI systems?",
            "What transparency obligations apply to AI systems?",
            "What are the requirements for general purpose AI models?",
            "Who bears compliance responsibility — the provider or the deployer?",
        ],
        "CNIL": [
            "Quelles sont les recommandations de la CNIL sur les systèmes d'IA?",
            "Comment la CNIL encadre-t-elle l'utilisation des données d'entraînement?",
            "Quelles sont les obligations RGPD pour les systèmes IA selon la CNIL?",
        ],
        "EDPB": [
            "What are the EDPB guidelines on automated decision-making?",
            "How does the EDPB define solely automated processing under Article 22?",
            "What safeguards are required for automated decisions with legal effects?",
        ],
        "DATA_GOVERNANCE_ACT": [
            "What are the obligations of data intermediaries under the DGA?",
            "How does the Data Governance Act regulate data altruism organisations?",
            "What conditions apply to re-use of public sector data under the DGA?",
        ],
    }

    if regulation:
        reg_key = regulation.value
        return SuggestionsResponse(
            regulation  = reg_key,
            suggestions = suggestions_by_regulation.get(reg_key, []),
        )

    # Sans filtre : 2 questions par réglementation pour un aperçu équilibré
    all_suggestions: list[str] = []
    for sug_list in suggestions_by_regulation.values():
        all_suggestions.extend(sug_list[:2])

    return SuggestionsResponse(
        regulation  = None,
        suggestions = all_suggestions,
    )
