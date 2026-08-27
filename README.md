# Assistant IA — RAG + MCP (test technique)

Assistant capable de répondre à des questions en combinant :
1. un **RAG** sur un corpus documentaire interne (notice de médicament),
2. un **serveur MCP** développé sur mesure exposant des outils de domaines distincts
   (recherche web, traitement de données), connecté à l'agent.

> Statut : projet en cours de construction dans le cadre d'un test technique limité dans le temps.
> Ce README est mis à jour au fur et à mesure des étapes (voir section "Avancement").

## Architecture (résumé)

```
Utilisateur
    │
    ▼
 Agent (LLM avec tool-calling, via Groq)
    │
    ├── outil "search_corpus"  → RAG sur la notice de médicament (Chroma + embeddings locaux)
    ├── outil "web_search"     → serveur MCP, recherche web (DuckDuckGo)
    └── outil "data_tool"      → serveur MCP, traitement de données local
```

Le routage RAG vs MCP est délégué au tool-calling natif du LLM : le modèle décide
lui-même quel(s) outil(s) appeler selon la question. Chaque décision est loggée pour
garder une trace du raisonnement de routage.

## Stack technique

- **LLM** : Groq (`llama-3.3-70b-versatile`), gratuit, sans carte bancaire
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2`, local, gratuit
- **Vector store** : ChromaDB (local, persistant)
- **Serveur MCP** : SDK officiel `mcp` (Python), transport stdio
- **Outil web** : DuckDuckGo (`ddgs`), sans clé API
- **Orchestration** : LangChain (chaînes RAG) + tool-calling Groq pour l'agent

Choix guidé par une contrainte : aucun budget disponible pour des API payantes.
Tout le stack tourne avec des services gratuits ou du calcul local.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
cp .env.example .env  # puis renseigner GROQ_API_KEY (gratuit sur console.groq.com)
```

## Lancement

### RAG seul (mode chat en ligne de commande)
```bash
cd rag
python rag.py
```

### Serveur MCP (indépendamment)
```bash
cd mcp_server
python server.py
```
*(instructions détaillées à venir dans `mcp_server/README.md`)*

### Agent complet (RAG + MCP orchestrés)
```bash
cd agent
python agent.py
```
*(à venir)*

## Corpus

`corpus/notice_doliprane_500mg.pdf` — notice patient officielle (source : Base de
Données Publique des Médicaments, ANSM), utilisée comme exemple de document interne
pour la démonstration du RAG.

## Points de l'objectif réalisés / non réalisés

| Point | Statut |
|---|---|
| RAG (ingestion, chunking, retrieval, citation, gestion du "je ne sais pas") | 🔄 en cours |
| Serveur MCP avec ≥2 outils de domaines distincts | 🔄 en cours |
| Routage RAG/MCP traçable | 🔄 en cours |
| Gestion des échecs (timeout, boucle d'appels) | ⏳ à faire si le temps le permet |
| 3e domaine d'outil MCP | ❌ non prévu (contrainte de temps) |
| Script d'évaluation automatique | ❌ non prévu (contrainte de temps) |
| Protection anti prompt-injection | ⏳ mesure basique prévue si le temps le permet |

## Limites connues

- Corpus volontairement réduit à un seul document pour la démo.
- Pas de gestion multi-utilisateur / multi-session persistante.
- (section à compléter en fin de projet)

## Pistes d'amélioration

- (section à compléter en fin de projet)
