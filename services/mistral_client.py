"""
Mistral LLM client — thin wrapper for generating narrative commentary.
"""

from __future__ import annotations

import os
import pandas as pd
import json

from mistralai import Mistral, UserMessage, SystemMessage



class LLMHandler:

    def __init__(self) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing MISTRAL_API_KEY in env")
        
        self.client = Mistral(api_key=api_key)

    def query(
            self,
            prompt: str, 
            model: str = "mistral-small-latest"
        ) -> str:
        """Call Mistral API and return a narrative commentary string.

        Uses a lazy import so the app degrades gracefully if mistralai is not installed.
        Returns None on any error (missing key, network failure, rate-limit, etc.).
        """
        response = self.client.chat.complete(
            model=model,
            messages=[UserMessage(content=prompt)],
        )
        return str(response.choices[0].message.content)
    
    def comment_cluster(
            self, 
            cluster_statistics: pd.DataFrame, 
            corr_plot: pd.DataFrame,
            model: str = "mistral-small-latest"
        ) -> dict:
        """
        Call mistral to comment clusters based on their statistics

        Returns:
            str: Mistral response
        """
        prompt = f"""
        Voici les statistics descriptives des clusters:
        {cluster_statistics.to_csv(index=False)}
        """
        system_prompt = """
        Tu travail sur des données de cyber sécurité d'un firewall d'entreprise. Tu trouves un nom et écrit une courte description en francais sur des clusters de requêtes en suivant ce format:
        {
            "cluster_id": {
                "name": ""
                "description"
            }
        }
        """

        response = self.client.chat.complete(
            model=model,
            response_format = {'type': 'json_object'},
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=prompt)
                ],
        )
        resp = str(response.choices[0].message.content)
        return json.loads(resp)
    
    def comment_projection(
            self,
            corr_plot: pd.DataFrame,
            model: str = "mistral-small-latest"
        ) -> dict:
        """
        Call mistral to comment projection axis based on their loadings

        Returns:
            str: Mistral response
        """
        prompt = """
        Nous avons projeté les données dans un espace en 3 dimension grace aux 3 premieres composantes latente d'une ACP.
        Tu dois trouver un nom a chacun des axes du graphique pour mieux comprendre le sens de chacun des axes.

        Voici les variables avec l'impact le plus négatif / positif sur chacun des axes (ainsi que leur coeficient):
        """
        for comp in corr_plot.columns:                # e.g. PC1, PC2, PC3
            sorted_comp = corr_plot[comp].sort_values()
            negatif = [f"{var} ({coef})" for var, coef in sorted_comp.head(3).items()]
            positif = [f"{var} ({coef})" for var, coef in sorted_comp.tail(3).items()]
            prompt += f"""
            {comp}:
            - positif: {", ".join(positif)}
            - negatif {", ".join(negatif)}
            """

        system_prompt = """
        Tu travail sur des données de cyber sécurité d'un firewall d'entreprise. Tu commentes en francais les analyse de cette donnée:
        {
            "PC#": {
                "name": un nom court pour l'axe,
                "description": description de l'axe,
            }
        }
        """

        response = self.client.chat.complete(
            model=model,
            response_format = {'type': 'json_object'},
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=prompt)
                ],
        )
        resp = str(response.choices[0].message.content)
        return json.loads(resp)

llm_handler = LLMHandler()