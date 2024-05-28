# Waste Collection Optimization

## Description
Ce projet utilise des algorithmes d'optimisation pour résoudre le problème de la collecte dynamique des déchets. Il implémente principalement l'algorithme de recuit simulé pour résoudre le problème du VRP (Vehicle Routing Problem) avec des contraintes de capacité.

## Fonctionnalités
- Chargement des données de localisation et de poids des conteneurs depuis un fichier Excel.
- Application de l'algorithme de recuit simulé pour optimiser les routes de collecte.
- Visualisation des résultats sur une interface graphique développée avec Tkinter.
- Visualisation des conteneurs et des routes optimisées avec Matplotlib intégré dans Tkinter.

## Prérequis
- Python 3.12
- Tkinter
- Matplotlib
- xlrd
- numpy
- sklearn

## Installation
1. Clonez le dépôt :
    ```sh
    git clone https://github.com/votre-utilisateur/votre-depot.git
    cd votre-depot
    ```
2. Créez et activez un environnement virtuel :
    ```sh
    python -m venv venv
    source venv/bin/activate # Sur Windows, utilisez `venv\Scripts\activate`
    ```
3. Installez les dépendances :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation
1. Placez votre fichier Excel avec les données des conteneurs à un emplacement accessible.
2. Exécutez le script principal :
    ```sh
    python votre_script.py
    ```
3. Entrez le chemin vers le fichier Excel dans l'interface et sélectionnez l'algorithme à utiliser (Stratégie 1 ou Stratégie 2).

## Exemples
### Exemple de structure de fichier Excel
Votre fichier Excel doit contenir les colonnes suivantes :
- `Coordonnées X` : Les coordonnées X des conteneurs.
- `Coordonnées Y` : Les coordonnées Y des conteneurs.
- `Poids` : Le poids ou la capacité de chaque conteneur.
- `Capacité` : La capacité des véhicules.
- `Var` : Une variable indicative de l'état des conteneurs.

### Exemple de visualisation
Les résultats incluent des visualisations telles que :
- Illustration des conteneurs sur la carte.
- Illustration des états des conteneurs (par couleur).
- Illustration des tournées optimisées pour chaque véhicule.

## Contribuer
Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request.


## Auteurs
- [Imad BOUCHAREB]
