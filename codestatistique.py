import random
import pandas as pd
from scipy import stats
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import statistics
import statsmodels.api as smi
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
import random
from datetime import datetime
from statsmodels.stats.multicomp import pairwise_tukeyhsd


print() #visibilité
# Lecture du fichier CSV
df = pd.read_csv('data_2.csv', sep = ';')
# Affichage des premières lignes du DataFrame
print(df.head())
print() #visibilité
# Afficher les noms des colonnes
print("Noms des colonnes :", df.columns.tolist())
print() #la meme

# #df = df.drop(columns=['Start Time'])

# Traitement de la colonne 'Start Time'
if 'Start Time' in df.columns:
    # Conversion de la colonne 'Start Time' en format datetime
    try:
        df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S')  # Ajustez le format si nécessaire
        print("Start Time: converti en type DateTime")
    except ValueError as e:
        print("Erreur lors de la conversion de 'Start Time':", e)

    # Conversion en timestamp (ou autre traitement pour analyse statistique)
    colonne_DateTime = df['Start Time'].astype('int64')
else:
    print("Start Time n'existe pas dans le DataFrame.")
    colonne_DateTime = None  # Affecter None si la colonne n'existe pas



def is_column_numeric(column):
     """ Vérifie si une colonne est entièrement composée de données numériques. """
     for item in column:
         # Convertit les chaînes avec des virgules en nombres
         if isinstance(item, str):
             try:
                 float(item.replace(',', '.'))
             except ValueError:
                 return False
         elif not isinstance(item, (int, float)):
             return False
     return True



# Parcourir chaque colonne et déterminer son type
for col in df.columns:
    column_type = 'variable_quantitative' if is_column_numeric(df[col]) else 'variable_qualitative'
    print(f"{col}: {column_type}")

def convertir_en_nombre(chaine):
    #""" Convertit une chaîne en nombre en remplaçant la virgule par un point. """
    try:
        return float(chaine.replace(',', '.'))
    except ValueError:
        return None

def est_colonne_numerique(colonne):
    """ Vérifie si une colonne est entièrement composée de données numériques. """
    for item in colonne:
        if isinstance(item, str):
            if convertir_en_nombre(item) is None:
                return False
        elif not isinstance(item, (int, float)):
            return False
    return True

# Convertir les colonnes numériques avec des virgules en points
for col in df.columns:
    if est_colonne_numerique(df[col]):
        df[col] = df[col].apply(lambda x: convertir_en_nombre(x) if isinstance(x, str) else x)
# Liste pour stocker les noms des colonnes quantitatives et qualitatives
colonnes_quantitatives = []
colonnes_qualitatives = []

# Parcourir chaque colonne et déterminer son type
for col in df.columns:
    if is_column_numeric(df[col]):
        colonnes_quantitatives.append(col)
    else:
        colonnes_qualitatives.append(col)

# Afficher les résultats

print()  # Ajout d'une ligne vide pour la lisibilité

print("Colonnes Quantitatives:", colonnes_quantitatives)
print("Colonnes Qualitatives:", colonnes_qualitatives)
print("Colonnes Type DateTime:", colonne_DateTime)

print()  # Ajout d'une ligne vide pour la lisibilité


#affcihe le nom des varibale qauli et le nb de modalité pour moi plus clair
# Créer un dictionnaire pour stocker le nombre de modalités de chaque variable qualitative
modalites_par_variable = {}

# Parcourir chaque colonne et compter les modalités pour les variables qualitatives
for col in df.columns:
    if not est_colonne_numerique(df[col]):
        nombre_modalites = len(df[col].unique())
        modalites_par_variable[col] = nombre_modalites

# Trier le dictionnaire par le nombre de modalités (ordre croissant)
modalites_par_variable = dict(sorted(modalites_par_variable.items(), key=lambda item: item[1]))

# Afficher le nombre de modalités pour chaque variable qualitative
for var, modalites in modalites_par_variable.items():
    print(f"Variable qualitative '{var}': {modalites} modalités")
    
print()  # Ajout d'une ligne vide pour la lisibilité
variables_2_modalites = {}  # Variables avec exactement 2 modalités
autres_variables = {}       # Variables avec un nombre différent de modalités

for variable, nombre_modalites in modalites_par_variable.items():
    if nombre_modalites == 2:
        variables_2_modalites[variable] = nombre_modalites
    else:
        autres_variables[variable] = nombre_modalites

print("Variables avec 2 modalités:", variables_2_modalites)
print("Autres variables:", autres_variables)
print()
# Vous pouvez ensuite procéder à vos calculs statistiques comme avant

def calculer_stats_quantitatives(colonne):
    """ Calcule les statistiques pour une variable quantitative. """
    return {
        'Moyenne': colonne.mean(),
        'Médiane': colonne.median(),
        'Écart Type': colonne.std(),
        'Variance': colonne.var(),
        'Min': colonne.min(),
        'Max': colonne.max()
    }

def calculer_stats_qualitatives(colonne):
    """ Calcule les statistiques pour une variable qualitative. """
    mode = colonne.mode()
    modalites = colonne.unique()
    return {
        'Fréquence': colonne.value_counts(),
        'Mode': mode[0] if not mode.empty else None,
        'Modalités': modalites,
        'Nombre de Modalités': len(modalites)
    }

def est_colonne_numerique(colonne):
    """ Vérifie si une colonne est entièrement composée de données numériques. """
    for item in colonne:
        if isinstance(item, str):
            try:
                float(item.replace(',', '.'))
            except ValueError:
                return False
        elif not isinstance(item, (int, float)):
            return False
    return True

# Analyse de chaque colonne
for col in df.columns:
    if est_colonne_numerique(df[col]):
        print(f"Statistiques pour la variable quantitative '{col}':")
        stats = calculer_stats_quantitatives(df[col])
    else:
        print(f"Statistiques pour la variable qualitative '{col}':")
        stats = calculer_stats_qualitatives(df[col])
    
    for cle, valeur in stats.items():
        print(f"  {cle}: {valeur}")
    print()  # Ajout d'une ligne vide pour la lisibilité

def est_colonne_numerique(colonne):
    """ Vérifie si une colonne est entièrement composée de données numériques. """
    return colonne.dtype.kind in 'bifc'  # b: bool, i: int, f: float, c: complex

import pandas as pd
from scipy import stats

# Itérer sur chaque colonne et effectuer les tests de normalité si la colonne est numérique
for col in df.columns:
    if est_colonne_numerique(df[col]):
        print(f"Test de normalité pour la colonne: {col}")

        # Test de Shapiro-Wilk
        shapiro_test = stats.shapiro(df[col].dropna())  # Supprimer les valeurs NaN
        print("  Shapiro-Wilk Test")
        print("  Statistique de test :", shapiro_test[0])
        print("  P-value :", shapiro_test[1])

        # Interprétation
        if shapiro_test[1] < 0.05:
            print("  La distribution ne suit pas une loi normale.")
            print("  Choisir un test non paramétrique.")
        else:
            print("  La distribution suit une loi normale.")
            print("  Choisir un test paramétrique.")
        print()

    
        

class TestStatistiques:

    
    @staticmethod
    def anova(df: pd.DataFrame, variable_quantitative: str, variable_qualitative: str) -> Tuple[float, float]:
        # Préparer les groupes pour l'ANOVA
        groupes = [df[df[variable_qualitative] == val][variable_quantitative] for val in df[variable_qualitative].unique()]

        # Effectuer le test ANOVA
        stat_value, p_value = stats.f_oneway(*groupes)
        return stat_value, p_value
      
    @staticmethod
    #à utiliser avec un test d'anova si on s'est que le test est significatif et que l'on veut conclure et savoir où est la diff
    def post_hoc_tukey(*sample):
        tukey_instance = stats.tukey_hsd(*sample)
        return tukey_instance
    

    
    @staticmethod
    #test d'independance : test une variable quantitative par rapport a une variable qualitative avec 2 modalités
    def t_test_independant(variable_1: np.ndarray, variable_2: np.ndarray) -> Tuple[float, float]:
        stat_value, p_value = stats.ttest_ind(variable_1, variable_2)
        return stat_value, p_value
    

 
    @staticmethod
    def mann_and_whitney_test(
             variable_1: np.ndarray,
             variable_2: np.ndarray
         ) -> Tuple[float, float]:
         stat_value, p_value = stats.mannwhitneyu(
             variable_1,
             variable_2
         )
         return stat_value, p_value
         pass


    @staticmethod
        #test de normalité: pour definir si le test est paramétrique ou non paramétrique
    def normality_test(variable) -> Tuple[float, float]:
            stat_value, p_value = stats.shapiro(variable)
            return stat_value, p_value



    @staticmethod
    # Test Kruskal-Wallis : tire au hasard une variable quantitative et une variable catégorielle avec 3 modalités
    def kruskal_wallis_test_random(df: pd.DataFrame) -> Tuple[float, float]:
        # Sélectionner toutes les colonnes quantitatives
        colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()

        # Sélectionner toutes les colonnes catégorielles avec 3 modalités
        colonnes_catégorielles = [col for col in df.columns if df[col].nunique() == 3 and df[col].dtype == 'object']

        if not colonnes_catégorielles:
            raise ValueError("Aucune colonne catégorielle avec exactement trois modalités n'a été trouvée.")

        # Choisir au hasard une colonne quantitative et une colonne catégorielle
        colonne_quantitative_choisie = random.choice(colonnes_quantitatives)
        colonne_catégorielle_choisie = random.choice(colonnes_catégorielles)

        # Extraire les valeurs pour le test
        variable_quantitative = df[colonne_quantitative_choisie].values
        variable_catégorielle = df[colonne_catégorielle_choisie].values

        # Effectuer le test de Kruskal-Wallis
        stat_value, p_value = stats.kruskal(variable_quantitative, variable_catégorielle)

        return stat_value, p_value

    @staticmethod
       
    def t_test_apparie_random(df: pd.DataFrame) -> Tuple[float, float]:
        # Sélectionner toutes les colonnes quantitatives
        colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()

        # Sélectionner toutes les colonnes qualitatives de temps (format de temps)
        colonnes_qualitatives_temps = [col for col in df.columns if df[col].dtype == 'object' and ':' in df[col].iloc[0]]

        if not colonnes_qualitatives_temps:
            raise ValueError("Aucune colonne qualitative de temps n'a été trouvée.")

        # Choisir au hasard une colonne quantitative et une colonne qualitative de temps
        colonne_quantitative_choisie = random.choice(colonnes_quantitatives)
        colonne_qualitative_temps_choisie = random.choice(colonnes_qualitatives_temps)

        # Extraire les valeurs pour le test
        variable_quantitative = df[colonne_quantitative_choisie].values
        variable_qualitative_temps = df[colonne_qualitative_temps_choisie].values

        # Effectuer le test t apparié
        stat_value, p_value = stats.ttest_rel(variable_quantitative, variable_qualitative_temps)

        return stat_value, p_value

    @staticmethod
       
    def wilcoxon_apparai_rang_signe_random(df: pd.DataFrame) -> Tuple[float, float]:
        # Sélectionner toutes les colonnes quantitatives
        colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()

        # Sélectionner toutes les colonnes qualitatives avec exactement deux modalités
        colonnes_qualitatives_deux_modalites = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() == 2]

        if not colonnes_qualitatives_deux_modalites:
            raise ValueError("Aucune colonne qualitative avec exactement deux modalités n'a été trouvée.")

        # Choisir au hasard une colonne quantitative et une colonne qualitative avec deux modalités
        colonne_quantitative_choisie = random.choice(colonnes_quantitatives)
        colonne_qualitative_choisie = random.choice(colonnes_qualitatives_deux_modalites)

        # Extraire les valeurs pour le test
        variable_quantitative = df[colonne_quantitative_choisie].values
        variable_qualitative = df[colonne_qualitative_choisie].values

        # Effectuer le test de Wilcoxon
        stat_value, p_value = stats.wilcoxon(variable_quantitative, variable_qualitative)

        return stat_value, p_value
    

        # Organiser les tests en paramétriques et non paramétriques
    @classmethod
    def classer_tests(cls):
            tests_parametriques = {
                't_test_independant': cls.t_test_independant,
                'anova': cls.anova,
                't_test_apparie': cls.t_test_apparie
            }

            tests_non_parametriques = {
                'mann_and_whitney_test': cls.mann_and_whitney_test,
                'kruskal_wallis': cls.kruskal_wallis,
                'wilcoxon_apparai_rang_signe': cls.wilcoxon_apparai_rang_signe,
                #'normality_test': cls.normality_test
            }

            return {
             'Tests Paramétriques': tests_parametriques, 
             'Tests Non Paramétriques': tests_non_parametriques
             }
    

    # def choisir_et_effectuer_test(df: pd.DataFrame):
    #     colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()
    #     colonnes_qualitatives = [col for col in df.columns if df[col].nunique() >= 3 and df[col].dtype == 'object']


def effectuer_test_t(df, colonnes_quantitatives, colonnes_qualitatives):
    if colonnes_qualitatives:
        colonne_quantitative = random.choice(colonnes_quantitatives)
        colonnes_qual_2_modalites = [col for col in colonnes_qualitatives if len(df[col].unique()) == 2]

        if colonnes_qual_2_modalites:
            colonne_qualitative = random.choice(colonnes_qual_2_modalites)
            modalites = df[colonne_qualitative].unique()
            groupe1 = df[df[colonne_qualitative] == modalites[0]][colonne_quantitative]
            groupe2 = df[df[colonne_qualitative] == modalites[1]][colonne_quantitative]
            resultat_stat, resultat_p = stats.ttest_ind(groupe1, groupe2, equal_var=False)
            print(f"Résultat du test t pour '{colonne_quantitative}' et '{colonne_qualitative}':")
            print("Valeur statistique du test t:", resultat_stat)
            print("P-value du test t:", resultat_p)
        else:
            print("Aucune colonne qualitative avec exactement 2 modalités.")
        if resultat_p <= 0.05:
                print("Je rejète HO et il existe une différence significative par rapport à l'hypothèse de départ.")
        else:
                print("Je rejete H1 et il n'existe pas de différence significative.")
    else:
        print("Aucune colonne qualitative appropriée pour le test t.")

# Choix aléatoire d'un test :"t_test", "anova", "mann_andwhitney_test", "wilcoxon_apparai_rang_signe_random", "kruskal_wallis_test_random", "t_test_apparie_random"
test_choisi = random.choice(["t_test", "anova", "mann_andwhitney_test", "wilcoxon_apparai_rang_signe_random", "kruskal_wallis_test_random", "t_test_apparie_random"])

# Exécuter le test correspondant au choix
if test_choisi == "t_test":
    # df = pd.DataFrame(...)  # Votre DataFrame
    effectuer_test_t(df, colonnes_quantitatives, colonnes_qualitatives)


elif test_choisi == "anova":
    if colonnes_quantitatives and autres_variables:
        colonne_quantitative = random.choice(colonnes_quantitatives)
        liste_autres_variables = list(autres_variables.keys())
        colonne_qualitative = random.choice(liste_autres_variables)

        print(f"Résultat de l'ANOVA pour '{colonne_quantitative}' et '{colonne_qualitative}':")

        # Appel correct à la méthode 'anova' avec les noms des colonnes
        resultat_stat, resultat_p = TestStatistiques.anova(df, colonne_quantitative, colonne_qualitative)

        print("Valeur statistique de l'ANOVA:", resultat_stat)
        print("P-value de l'ANOVA:", resultat_p)
        if resultat_p <= 0.05:
            print("Je rejète HO et il existe une différence significative par rapport à l'hypothèse de départ.")
            print('je vais faire un post hoc test pour savoir qu''elle est cette différence significative.')
            print()
        else:
            print("Je rejete H1 et il n'existe pas de différence significative.")
    else:
        print("Vérifiez que les listes 'colonnes_quantitatives' et 'autres_variables' sont bien définies et non vides.")
 

     
   
       
elif test_choisi == "mann_andwhitney_test":
    # Sélectionner aléatoirement une variable quantitative et une qualitative pour le test de Mann-Whitney
    if colonnes_qualitatives and colonnes_quantitatives:
        colonne_quantitative = random.choice(colonnes_quantitatives)
        colonne_qualitative = random.choice(colonnes_qualitatives)
        
        modalites = list(df[colonne_qualitative].unique())
        if len(modalites) >= 2:
            groupe1 = df[df[colonne_qualitative] == modalites[0]][colonne_quantitative]
            groupe2 = df[df[colonne_qualitative] == modalites[1]][colonne_quantitative]

            # Effectuer le test de Mann-Whitney
            resultat_stat, resultat_p = stats.mannwhitneyu(groupe1, groupe2)

            print(f"Résultat du test de Mann-Whitney pour '{colonne_quantitative}' et '{colonne_qualitative}':")
            print("Valeur statistique du test de Mann-Whitney:", resultat_stat)
            print("P-value du test de Mann-Whitney:", resultat_p)
            if resultat_p <= 0.05:
                print("Je rejète HO et il existe une différence significative par rapport à l'hypothèse de départ.")
            else:
                print("Je rejete H1 et il n'existe pas de différence significative")
        else:
            print("La variable qualitative sélectionnée n'a pas suffisamment de modalités pour le test de Mann-Whitney.")
    else:
        print("Aucune colonne qualitative appropriée pour le test de Mann-Whitney.")


elif test_choisi == "wilcoxon_apparai_rang_signe_random":


    def wilcoxon_apparie_rang_signe_random(df: pd.DataFrame) -> Tuple[float, float]:
        # Sélectionner toutes les colonnes quantitatives
        colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()

        # Sélectionner toutes les colonnes qualitatives avec exactement deux modalités
        colonnes_qualitatives_deux_modalites = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() == 2]

        if not colonnes_qualitatives_deux_modalites:
            raise ValueError("Pour le test de Wilcoxon : aucune colonne qualitative avec exactement deux modalités n'a été trouvée.")

        # Choisir au hasard une colonne quantitative et une colonne qualitative avec deux modalités
        colonne_quantitative_choisie = random.choice(colonnes_quantitatives)
        colonne_qualitative_choisie = random.choice(colonnes_qualitatives_deux_modalites)

        # Séparer la variable quantitative en fonction des modalités de la variable qualitative
        modalites = df[colonne_qualitative_choisie].unique()
        groupe1 = df[df[colonne_qualitative_choisie] == modalites[0]][colonne_quantitative_choisie]
        groupe2 = df[df[colonne_qualitative_choisie] == modalites[1]][colonne_quantitative_choisie]

        # Vérifier si les tailles des groupes sont égales
        if len(groupe1) != len(groupe2):
            raise ValueError("Pour le test de Wilcoxon : les groupes n'ont pas la même taille.")

        # Effectuer le test de Wilcoxon
        stat_value, p_value = stats.wilcoxon(groupe1, groupe2)

        return stat_value, p_value
    
    try:
        stat_value, p_value = wilcoxon_apparie_rang_signe_random(df)
        print(f"Résultat du test de Wilcoxon pour échantillons appariés (choix aléatoire) :")
        print("Valeur statistique :", stat_value)
        print("P-value :", p_value)
        if p_value <= 0.05:
            print("Je rejète HO et il existe une différence significative par rapport à l'hypothèse de départ.")
        else:
            print("Je rejete H1 et il n'existe pas de différence significative.")
    except ValueError as e:
        print(str(e))

elif test_choisi == "kruskal_wallis_test_random":
    

    #     # Effectuer le test de Kruskal-Wallis
    #     return stat_value, p_value
    def kruskal_wallis_test_random(df: pd.DataFrame) -> Tuple[float, float]:
    # Sélectionner toutes les colonnes quantitatives
        colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()

    # Sélectionner toutes les colonnes catégorielles avec 3 modalités
        colonnes_catégorielles = [col for col in df.columns if df[col].nunique() == 3 and df[col].dtype == 'object']

        if not colonnes_catégorielles:
             raise ValueError("Aucune colonne catégorielle avec exactement trois modalités n'a été trouvée.")

    # Choisir au hasard une colonne quantitative et une colonne catégorielle
        colonne_quantitative_choisie = random.choice(colonnes_quantitatives)
        colonne_catégorielle_choisie = random.choice(colonnes_catégorielles)

    # Séparer les données en fonction des catégories
        groupes = [df[df[colonne_catégorielle_choisie] == cat][colonne_quantitative_choisie] for cat in df[colonne_catégorielle_choisie].unique()]

    # Effectuer le test de Kruskal-Wallis
        stat_value, p_value = stats.kruskal(*groupes)

        return stat_value, p_value
    
    try:
        stat_value, p_value = kruskal_wallis_test_random(df)
        print(f"Résultat du test de Kruskal-Wallis (choix aléatoire) :")
        print("Valeur statistique :", stat_value)
        print("P-value :", p_value)
        if p_value <= 0.05:
            print("Je rejète HO et il existe une différence significative par rapport à l'hypothèse de départ.")
        else:
            print("Je rejete H1 et il n'existe pas de différence significative.")
    except ValueError as e:
        print(str(e))

elif test_choisi == "t_test_apparie_random":
    
    def t_test_apparie_random(df: pd.DataFrame, colonne_DateTime) -> Tuple[float, float]:
        # Sélectionner toutes les colonnes quantitatives
        colonnes_quantitatives = df.select_dtypes(include=[np.number]).columns.tolist()

        if not colonnes_quantitatives:
            raise ValueError("Aucune colonne quantitative n'a été trouvée.")

        # Choisir au hasard une colonne quantitative
        colonne_quantitative_choisie = random.choice(colonnes_quantitatives)

        # Extraire les valeurs pour le test
        variable_quantitative = df[colonne_quantitative_choisie].values

        # Effectuer le test t apparié
        stat_value, p_value = stats.ttest_rel(variable_quantitative, colonne_DateTime)

        return stat_value, p_value

    if colonne_DateTime is not None:
        try:
            stat_value, p_value = t_test_apparie_random(df, colonne_DateTime)
            print(f"Résultat du test t apparié (choix aléatoire) :")
            print(f"Valeur statistique : {stat_value}")
            print(f"P-value : {p_value}")
            if p_value <= 0.05:
                print("Rejet de l'hypothèse nulle et il existe une différence significative.")
            else:
                print("Pas de rejet de l'hypothèse nulle et il n'existe pas de différence significative.")
        except ValueError as e:
            print(str(e))

print()

