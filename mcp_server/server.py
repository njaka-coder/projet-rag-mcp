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

@mcp.tool()
def calculate_paracetamol_dose(poids_kg: float) -> str:
    """Calcule la dose de paracétamol recommandée selon le poids du patient.

    Basé sur les règles de posologie publiques du Doliprane 500mg (source ANSM) :
    dose maximale par prise, intervalle minimum entre deux prises, dose maximale
    par jour. Réservé aux patients à partir de 27 kg (environ 8 ans).

    Args:
        poids_kg: poids du patient en kilogrammes.
    """
    disclaimer = (
        "\n\n(Information calculée à titre indicatif à partir de la notice. "
        "Ne remplace pas l'avis d'un médecin ou d'un pharmacien.)"
    )

    if poids_kg < 27:
        return (
            "Ce médicament est réservé à l'adulte et à l'enfant à partir de 27 kg "
            "(environ 8 ans). Pour un poids inférieur, demandez conseil à votre "
            "médecin ou pharmacien pour une présentation adaptée." + disclaimer
        )
    if poids_kg <= 40:
        return (
            "Poids 27-40 kg : dose maximale par prise 500 mg (1 comprimé), "
            "intervalle minimum 6 heures, dose maximale par jour 2000 mg "
            "(4 comprimés)." + disclaimer
        )
    if poids_kg <= 49:
        return (
            "Poids 41-49 kg : dose maximale par prise 500 mg (1 comprimé), "
            "intervalle minimum 4 heures, dose maximale par jour 3000 mg "
            "(6 comprimés)." + disclaimer
        )
    return (
        "Poids 50 kg et plus : dose maximale par prise 500 mg à 1000 mg "
        "(1 à 2 comprimés), intervalle minimum 4 heures, dose maximale par jour "
        "3000 mg (6 comprimés)." + disclaimer
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")