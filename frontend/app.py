"""
frontend/app.py — Interface Streamlit pour AI Legal Governance Platform

LANCEMENT :
    streamlit run frontend/app.py

CONFIGURATION :
    L'app communique avec l'API FastAPI sur http://localhost:8000.
    Elle gère elle-même l'authentification JWT via /auth/token/demo.
    Configurable via les variables d'environnement ou le sidebar.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Optional

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Juris AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS PERSONNALISÉ ───────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Typography */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #1a1a2e !important;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 100%);
    border-bottom: 1px solid #e5e7eb;
    padding: 2rem 0 1.5rem;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header .subtitle {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.875rem;
    color: #6b7280;
    margin-top: 0.25rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Réponse */
.response-container {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #d1d5db;
    border-radius: 0 8px 8px 0;
    padding: 1.75rem;
    margin: 1.5rem 0;
    line-height: 1.75;
    font-size: 0.9375rem;
    color: #1f2937;
}

/* Sources */
.source-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    transition: border-color 0.15s;
}
.source-card:hover { border-color: #9ca3af; }
.source-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.5rem;
}
.source-meta {
    font-size: 0.8125rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.source-text {
    font-size: 0.8125rem;
    color: #4b5563;
    line-height: 1.6;
    font-style: italic;
}
.score-badge {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 3px;
    padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #374151;
    margin-right: 0.5rem;
}
.score-badge.rerank {
    background: #f0fdf4;
    border-color: #86efac;
    color: #166534;
}

/* Historique */
.history-item {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.875rem 1rem;
    margin: 0.375rem 0;
    cursor: pointer;
    transition: all 0.15s;
}
.history-item:hover {
    border-color: #1a1a2e;
    background: #f9fafb;
}
.history-question {
    font-size: 0.875rem;
    color: #1f2937;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.history-meta {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Métriques */
.metric-row {
    display: flex;
    gap: 0.75rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-pill {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #374151;
}
.metric-pill.success { background: #f0fdf4; border-color: #86efac; color: #166534; }
.metric-pill.warning { background: #fff7ed; border-color: #fed7aa; color: #9a3412; }
.metric-pill.info    { background: #f3f4f6; border-color: #d1d5db; color: #374151; }

/* Switch langue FR/EN */
.lang-switch {
    display: flex;
    gap: 4px;
    align-items: center;
}
.lang-btn {
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    border: 1px solid #d1d5db;
    cursor: pointer;
    transition: all 0.15s;
    background: white;
    color: #6b7280;
}
.lang-btn.active {
    background: #1a1a2e;
    color: white;
    border-color: #1a1a2e;
}



/* Guardrail */
.guardrail-ok      { color: #16a34a; font-size: 0.8125rem; }
.guardrail-low     { color: #d97706; font-size: 0.8125rem; }
.guardrail-high    { color: #dc2626; font-size: 0.8125rem; }

/* Disclaimer */
.disclaimer {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.875rem 1rem;
    margin-top: 1rem;
    font-size: 0.8125rem;
    color: #6b7280;
    font-style: italic;
    line-height: 1.6;
}

/* Sidebar */
.sidebar-section {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.8125rem;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.25rem 0 0.5rem;
    padding-bottom: 0.375rem;
    border-bottom: 1px solid #e5e7eb;
}

/* Status badge */
.status-ok  { color: #16a34a; font-weight: 600; }
.status-err { color: #dc2626; font-weight: 600; }

/* Séparateur élégant */
.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #9ca3af;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 1rem; }
.empty-state p { font-size: 0.9375rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────────

if "token" not in st.session_state:
    st.session_state.token = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "api_status" not in st.session_state:
    st.session_state.api_status = None


# ── FONCTIONS API ──────────────────────────────────────────────────────────────

def get_token() -> Optional[str]:
    """Obtient un token JWT demo depuis l'API."""
    try:
        r = requests.post(f"{API_BASE}/auth/token/demo", timeout=5)
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception:
        pass
    return None


def check_health() -> dict:
    """Vérifie l'état de l'API."""
    try:
        r = requests.get(f"{API_BASE}/api/v1/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def search(question: str, k: int, regulation: Optional[str],
           language: str, use_reranking: bool) -> dict:
    """Lance une recherche via l'API RAG."""
    if not st.session_state.token:
        st.session_state.token = get_token()

    headers = {
        "Authorization": f"Bearer {st.session_state.token}",
        "Content-Type": "application/json",
    }
    payload = {
        "question": question,
        "k": k,
        "language": language,
        "use_reranking": use_reranking,
        "use_query_expansion": True,
    }
    if regulation and regulation != "Tout le corpus":
        payload["regulation"] = regulation.replace(" ", "_").upper()

    r = requests.post(
        f"{API_BASE}/api/v1/search",
        headers=headers,
        json=payload,
        timeout=60,
    )
    return r.json(), r.status_code


def get_suggestions(regulation: Optional[str] = None) -> list:
    """Récupère les questions suggérées."""
    try:
        params = {}
        if regulation and regulation != "Tout le corpus":
            params["regulation"] = regulation.replace(" ", "_").upper()
        r = requests.get(f"{API_BASE}/api/v1/search/suggestions",
                        params=params, timeout=5)
        if r.status_code == 200:
            return r.json().get("suggestions", [])
    except Exception:
        pass
    return []


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    import os
    logo_path = os.path.join(os.path.dirname(__file__), "juris_ai_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem;">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem;
                        font-weight: 700; color: #1a1a2e; line-height: 1.3;">
                ⚖️ Juris AI
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Status API
    api_label = "Statut API" if st.session_state.get("ui_lang","fr") == "fr" else "API Status"
    st.markdown(f'<div class="sidebar-section">{api_label}</div>', unsafe_allow_html=True)
    if st.button("↻  Vérifier la connexion", use_container_width=True):
        health = check_health()
        st.session_state.api_status = health

    if st.session_state.api_status:
        h = st.session_state.api_status
        if h.get("status") == "ok":
            vs = h.get("vector_store", {})
            st.markdown(f'<span class="status-ok">● API connectée</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.75rem; color:#6b7280; margin-top:0.25rem; font-family:'JetBrains Mono',monospace;">
                {vs.get('n_vectors', '?')} vecteurs · dim {vs.get('dimension', '?')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">● API hors ligne</span>', unsafe_allow_html=True)

    # Filtres
    filters_label = "Filtres de recherche" if st.session_state.get("ui_lang","fr") == "fr" else "Search filters"
    st.markdown(f'<div class="sidebar-section">{filters_label}</div>', unsafe_allow_html=True)

    reg_label = "Regulation" if st.session_state.get("ui_lang","fr") == "en" else "Réglementation"
    regulation_options_fr = ["Tout le corpus", "GDPR", "EU AI Act", "CNIL", "EDPB", "DATA GOVERNANCE ACT"]
    regulation_options_en = ["All corpus", "GDPR", "EU AI Act", "CNIL", "EDPB", "DATA GOVERNANCE ACT"]
    regulation_options = regulation_options_en if st.session_state.get("ui_lang","fr") == "en" else regulation_options_fr
    regulation = st.selectbox(reg_label, regulation_options, index=0)

    response_lang_label = "Response language" if st.session_state.get("ui_lang","fr") == "en" else "Langue de la réponse"
    language = st.radio(response_lang_label, ["Français", "English"],
                        horizontal=True, index=0)
    lang_code = "fr" if language == "Français" else "en"

    k = st.slider("Nombre de sources (k)", min_value=3, max_value=10, value=5)

    use_reranking = st.toggle("Reranking cross-encoder", value=True,
                               help="Plus précis, +100-200ms de latence")

    # Historique
    hist_label = "Historique" if st.session_state.get("ui_lang","fr") == "fr" else "History"
    st.markdown(f'<div class="sidebar-section">{hist_label}</div>', unsafe_allow_html=True)

    if st.session_state.history:
        clear_label = "🗑 Clear history" if st.session_state.get("ui_lang","fr") == "en" else "🗑 Effacer l'historique"
        if st.button(clear_label, use_container_width=True):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()

        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            q_short = item["question"][:55] + "..." if len(item["question"]) > 55 else item["question"]
            reg_label = item.get("regulation_filter") or "Corpus"
            lat = item.get("processing_time_ms", 0)
            ts = item.get("timestamp", "")

            clicked = st.button(
                f"**{q_short}**\n\n`{reg_label}` · `{lat:.0f}ms`",
                key=f"hist_{i}",
                use_container_width=True,
            )
            if clicked:
                st.session_state.current_result = item
    else:
        st.markdown("""
        <div style="font-size:0.8125rem; color:#9ca3af; padding: 0.5rem 0;">
            Aucune recherche effectuée
        </div>
        """, unsafe_allow_html=True)


# ── HEADER PRINCIPAL ───────────────────────────────────────────────────────────

# Switch langue FR/EN en haut à droite
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "fr"

col_title, col_lang = st.columns([8, 1])
with col_title:
    if st.session_state.ui_lang == "fr":
        st.markdown("""
        <div class="main-header">
            <h1>Recherche Juridique Intelligente</h1>
            <div class="subtitle">GDPR · EU AI Act · Data Governance Act · EDPB · CNIL</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="main-header">
            <h1>Legal Intelligence Search</h1>
            <div class="subtitle">GDPR · EU AI Act · Data Governance Act · EDPB · CNIL</div>
        </div>
        """, unsafe_allow_html=True)

with col_lang:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    # CSS pour hover rouge sur les boutons de langue
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        padding: 4px 8px !important;
        min-height: 0 !important;
        height: 32px !important;
        border: 1px solid #d1d5db !important;
        background: white !important;
        color: #6b7280 !important;
        transition: all 0.15s !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        border-color: #dc2626 !important;
        color: #dc2626 !important;
        background: #fff5f5 !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        padding: 4px 8px !important;
        min-height: 0 !important;
        height: 32px !important;
        background: #1a1a2e !important;
        border: 1px solid #1a1a2e !important;
    }
    </style>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("FR", key="btn_fr",
                     type="primary" if st.session_state.ui_lang == "fr" else "secondary",
                     use_container_width=True):
            st.session_state.ui_lang = "fr"
            st.rerun()
    with c2:
        if st.button("EN", key="btn_en",
                     type="primary" if st.session_state.ui_lang == "en" else "secondary",
                     use_container_width=True):
            st.session_state.ui_lang = "en"
            st.rerun()


# ── ZONE DE RECHERCHE ──────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_input(
        label="Question juridique",
        placeholder="Ex: What are the obligations of a data controller under GDPR?" if st.session_state.get("ui_lang","fr") == "en" else "Ex: Quelles sont les obligations du responsable de traitement selon le RGPD ?",
        label_visibility="collapsed",
    )

with col_btn:
    search_btn_label = "Search →" if st.session_state.get("ui_lang","fr") == "en" else "Rechercher →"
    search_clicked = st.button(search_btn_label, type="primary", use_container_width=True)

# Questions suggérées
suggestions = get_suggestions(regulation if regulation != "Tout le corpus" else None)
if suggestions:
    sug_label = "💡 Suggested questions" if st.session_state.get("ui_lang","fr") == "en" else "💡 Questions suggérées"
    with st.expander(sug_label, expanded=False):
        cols = st.columns(2)
        for i, sug in enumerate(suggestions[:8]):
            if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                question = sug
                search_clicked = True


# ── EXÉCUTION DE LA RECHERCHE ─────────────────────────────────────────────────

if search_clicked and question and len(question.strip()) >= 10:

    with st.spinner("Analyse en cours…"):
        try:
            t0 = time.time()
            result, status_code = search(
                question=question,
                k=k,
                regulation=regulation if regulation != "Tout le corpus" else None,
                language=lang_code,
                use_reranking=use_reranking,
            )
            elapsed = time.time() - t0

            if status_code == 200:
                result["timestamp"] = datetime.now().strftime("%H:%M")
                result["question"] = question
                st.session_state.current_result = result
                # Ajouter à l'historique
                st.session_state.history.append(result)
                if len(st.session_state.history) > 50:
                    st.session_state.history = st.session_state.history[-50:]
            else:
                st.error(f"Erreur {status_code} : {result.get('detail', 'Erreur inconnue')}")
                st.session_state.current_result = None

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Impossible de contacter l'API. Vérifiez que le serveur tourne sur http://localhost:8000")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")

elif search_clicked and question and len(question.strip()) < 10:
    st.warning("La question doit contenir au moins 10 caractères.")


# ── AFFICHAGE DES RÉSULTATS ────────────────────────────────────────────────────

result = st.session_state.current_result

if result and result.get("answer"):

    # Métriques rapides
    lat = result.get("processing_time_ms", 0)
    n_ret = result.get("n_chunks_retrieved", 0)
    n_used = result.get("n_chunks_used", 0)
    model = result.get("model_used", "")
    expanded = result.get("query_expanded", False)
    reg_filter = result.get("regulation_filter") or "Corpus complet"

    st.markdown(f"""
    <div class="metric-row">
        <span class="metric-pill info">⏱ {lat:.0f} ms</span>
        <span class="metric-pill">📚 {n_ret} → {n_used} sources</span>
        <span class="metric-pill">🔍 {reg_filter}</span>
        <span class="metric-pill {'success' if expanded else ''}">
            {'✦ Query expanded' if expanded else '○ No expansion'}
        </span>
        <span class="metric-pill" style="font-size:0.65rem;">{model}</span>
    </div>
    """, unsafe_allow_html=True)

    # Réponse
    answer_raw = result.get("answer", "")

    # Séparer le disclaimer de la réponse principale
    disclaimer_text = ""
    main_answer = answer_raw
    if "---" in answer_raw and "⚖️" in answer_raw:
        parts = answer_raw.split("---\n", 1)
        if len(parts) == 2:
            main_answer = parts[0].strip()
            disclaimer_text = parts[1].strip()

    st.markdown(f'<div class="response-container">{main_answer}</div>',
                unsafe_allow_html=True)

    if disclaimer_text:
        st.markdown(f'<div class="disclaimer">{disclaimer_text}</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Sources
    sources = result.get("sources", [])
    if sources:
        sources_title = f"Sources used ({len(sources)})" if st.session_state.get("ui_lang","fr") == "en" else f"Sources utilisées ({len(sources)})"
        st.markdown(f"#### {sources_title}")

        for i, src in enumerate(sources, 1):
            seg_id    = src.get("segment_id", "—")
            reg       = src.get("regulation", "—")
            year      = src.get("year", "")
            text      = src.get("text", "")[:300] + "..." if src.get("text") else ""
            faiss_sc  = src.get("similarity_score", 0)
            rerank_sc = src.get("rerank_score")

            scores_html = f'<span class="score-badge">FAISS {faiss_sc:.3f}</span>'
            if rerank_sc is not None:
                scores_html += f'<span class="score-badge rerank">Rerank {rerank_sc:.3f}</span>'

            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">[SOURCE {i}] {seg_id}</div>
                <div class="source-meta">
                    {scores_html}
                    <span style="color:#9ca3af;">{reg} · {year}</span>
                </div>
                <div class="source-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # Citations extraites
    citations = result.get("citations", [])
    if citations:
        with st.expander("📎 Citations extraites de la réponse"):
            for c in citations:
                st.markdown(f"- `{c}`")

elif not result:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">⚖️</div>
        <p>
            Posez une question juridique en langage naturel.<br>
            Le système recherche dans <strong>2 016 chunks</strong> de textes réglementaires<br>
            et synthétise une réponse avec citations vérifiables.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Afficher quelques exemples directement
    _faq_title = "#### Frequently asked questions" if st.session_state.get("ui_lang","fr") == "en" else "#### Questions fréquentes"
    st.markdown(_faq_title)
    example_questions = [
        "What are the obligations of a data controller under GDPR?",
        "What AI systems are classified as high-risk under the EU AI Act?",
        "When is a Data Protection Impact Assessment required?",
        "What are the requirements for valid consent under GDPR?",
    ]
    cols = st.columns(2)
    for i, eq in enumerate(example_questions):
        if cols[i % 2].button(eq, key=f"ex_{i}", use_container_width=True):
            question = eq
            with st.spinner("Analyse en cours…"):
                try:
                    result_ex, sc = search(eq, k, regulation if regulation != "Tout le corpus" else None, lang_code, use_reranking)
                    if sc == 200:
                        result_ex["timestamp"] = datetime.now().strftime("%H:%M")
                        result_ex["question"] = eq
                        st.session_state.current_result = result_ex
                        st.session_state.history.append(result_ex)
                        st.rerun()
                except Exception:
                    st.error("API non disponible")
