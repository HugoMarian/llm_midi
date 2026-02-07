# llm_midi

**Debeve Geoffrey** et **Marian Hugo**

## Compostion des notes

Dans nos fichiers tokenisés, nos notes sont composés de 4 paramètres qui doivent être dans cette ordre:

1. POSITION_\<NB>; La postion de la note dans le temps en ticks.
2. NOTE_ON_\<NB>; La note à jouer.
3. VELOCITY_\<NB>; La puissance avec laquelle la note doit être jouée.
4. DURATION_\<NB>; Le temps ou la note doit être maintenu.

## Méthode d'entraînement

L'entraînement du modèle se déroule selon ces étapes:  

* Tirage d'un nombre choisi de musiques aléatoirement dans le répertoire
* Tokenisation des musiques en ajoutant l'intitulé à chaque fois
* Entraînement de 10 époques du gpt sur les textes concaténés

Une fois le premier entraînement réalisé, on peut reprendre les paramètres du modèle stockés dans le fichier `modelGPTmidiAll.pth`.

## Résultats observés

Pour l'instant au bout de 28 lignes (7 notes), on ne génère plus le texte correctement.
En donnant le nom du musicien comme contexte de départ, on génère des styles différents.
Le gpt arrive à générer des accords mineurs et majeurs qu'il a sûrement du voir dans son entraînement.

Il y a un problème pour la position de départ, la génération en donnant une position de départ n'est pas cohérente avec la note suivante qui est bien plus tard.
