# llm_midi

## Résultats observés
Pour l'instant au bout de 28 lignes, on ne génère plus le texte correctement.
En donnant le nom du musicien comme contexte de départ, on génère des styles différents.
Le gpt arrive à générer des accords mineurs et majeurs qu'il a sûrement du voir dans son entraînement.

Il y a un problème pour la position de départ, la génération en donnant une position de départ n'est pas cohérente avec la note suivante qui est bien plus tard.