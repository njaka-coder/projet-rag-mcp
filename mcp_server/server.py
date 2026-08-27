"""
Serveur MCP — outils pour l'assistant notice de médicament.

Expose deux outils de domaines clairement distincts :
1. web_search        : recherche web (DuckDuckGo, gratuit, sans clé API)
2. calculate_paracetamol_dose : calcul local, sans API, dérivé des règles de
   posologie publiques (ANSM) — démonstration d'un outil de traitement de données.

"""

from mcp.server.fastmcp import FastMCP
from ddgs import DDGS

mcp = FastMCP("notice-medicament-tools")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Recherche des informations sur le web via DuckDuckGo.

    À utiliser pour toute question dont la réponse ne peut PAS se trouver dans la
    notice du médicament : actualités, rappels de sécurité récents, alternatives
    génériques, avis, disponibilité en pharmacie, etc.

    Args:
        query: la requête de recherche.
        max_results: nombre maximum de résultats à retourner (par défaut 5).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Erreur lors de la recherche web : {exc}"

    if not results:
        return "Aucun résultat trouvé pour cette recherche."

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Sans titre")
        body = r.get("body", "")
        url = r.get("href", "")
        formatted.append(f"{i}. {title}\n{body}\nSource : {url}")
    return "\n\n".join(formatted)


if __name__ == "__main__":
    mcp.run(transport="stdio")