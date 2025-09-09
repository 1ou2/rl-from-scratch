# Structure du labyrinthe
Le labyrinthe a un début et une fin.
Le labyrinthe a des murs

# L’agent
L’agent peut se déplacer suivant haut, bas, gauche droite
S’il va dans un mur il reste sur place
S’il va hors des limites du labyrinthe il reste sur place

# Objectif
Trouver la sortie le plus rapidement possible

# Fin
Le jeu s’arrête si l’agent a trouvé la sortie ou s’il a fait plus de cols*rows mouvements

# Récompense
-1 : à chaque déplacement pour valoriser les sorties rapides
+10 : s’il trouve la sortie

# questions 
Différences entre un épisode et une trajectoire ?