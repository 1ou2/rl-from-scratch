# Tutoriel sur l'utilisation d'une value function en Dynamic Programming

## L'environnement
Grille 4x4
En haut à gauche, et en bas à droite sont des états terminaux

  T  |  1  |  2  |  3 
----------------------
  4  |  5  |  6  |  7
----------------------
  8  |  9  |  10 |  11
----------------------
  12 | 13  |  14 |  T

Actions possibles : haut, bas, gauche, droite
Récompense : -1 par mouvement

## Politique
Politique aléatoire : chaque action a une probabilité de 0.25

## Value function
Initialisée à 0 pour chaque état
γ = 1

## Objectif
Calculer la value function pour chaque état en utilisant l'algorithme de policy evaluation

## Algorithme de policy evaluation
1. Initialiser la value function V(s) pour chaque état s à 0
2. Répéter jusqu'à convergence :
    - Pour chaque état s non terminal :
      - Calculer la nouvelle valeur V(s) en utilisant la formule :
         V(s) = Σ [π(a|s) * (R(s,a) + γ * V(s'))]
         où π(a|s) est la probabilité de prendre l'action a dans l'état s,
         R(s,a) est la récompense reçue après avoir pris l'action a dans l'état s,
         γ est le facteur de discount (0 < γ < 1),
         s' est l'état résultant après avoir pris l'action a dans l'état s.
    - Mettre à jour V(s) avec la nouvelle valeur calculée.



