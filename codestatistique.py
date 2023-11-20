import random
import pandas as pd
from scipy import stats
import numpy
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




# Lecture du fichier CSV
df = pd.read_csv('data_2.csv', sep = ';')
# Affichage des premières lignes du DataFrame
print(df.head())

# Afficher les noms des colonnes
print("Noms des colonnes :", df.columns.tolist())


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
    #test d'anova : on teste une variable quantitative par rapport a une variable categorielle avec 3 modalités
    def anova(*sample) -> Tuple[float, float]:
        stat_value, p_value = stats.f_oneway(*sample)

    # Utilisation d'une structure if pour vérifier la significativité
        if p_value < 0.05:
            print("La différence est statistiquement significative.")
            print("Je fais un post hoc test.")
        else:
            print("La différence n'est pas statistiquement significative.")

        resultat_de_lanova = (stat_value, p_value)
        return resultat_de_lanova

    @staticmethod
    #à utiliser avec un test d'anova si on s'est que le test est significatif et que l'on veut conclure et savoir où est la diff
    def post_hoc_tukey(*sample):
        tukey_instance = stats.tukey_hsd(*sample)
        return tukey_instance

    @staticmethod
    #test d'independance : test une variable quantitative par rapport a une variable qualitative avec 2 modalités
    def t_test_independant(
            variable_1: numpy.ndarray,
            variable_2: numpy.ndarray
        ) -> Tuple[float, float]:
        stat_value, p_value = stats.ttest_ind(
            variable_1,
            variable_2
        )
        return stat_value, p_value
        pass

    @staticmethod
    #test d'independance=mann et whitney : test une variable quantitative par rapport a une variable qualitative avec 2 modalités 
    def mann_and_whitney_test(
            variable_1: numpy.ndarray,
            variable_2: numpy.ndarray
        ) -> Tuple[float, float]:
        stat_value, p_value = stats.mannwhitneyu(
            variable_1,
            variable_2
        )
        return stat_value, p_value
        pass

    # ... autres méthodes statiques ...
    @staticmethod
    #test de normalité: pour definir si le test est paramétrique ou non paramétrique
    def normality_test(variable) -> Tuple[float, float]:
        stat_value, p_value = stats.shapiro(variable)
        return stat_value, p_value
    
      
    @staticmethod
    #test d'anova=kruskal wallis: on teste une variable quantitative par rapport a une variable categorielle avec 3 modalités
    def kruskal_wallis(*sample) -> Tuple[float, float]:
        stat_value, p_value = stats.kruskal(*sample)
        return stat_value, p_value
     

    @staticmethod
    #test appareillé: une varibale quantitaive etudié en fonction d'une variable qualitative
    #ex= même mesure à des temps différents
    def t_test_apparie(variable_1, variable_2, raise_error: bool = False) -> Tuple[float, float]:
        if raise_error:
            if len(variable_1) != len(variable_2):
                raise ValueError("Les longeurs des echantillons doivent être identiques")
        else:
            stat_value, p_value = stats.ttest_rel(variable_1, variable_2)
            return stat_value, p_value
        pass

    @staticmethod
    #test appareillé = wilcoxon signed=one sample
    def wilcoxon_apparai_rang_signe(variable_1, variable_2, raise_error: bool = False) -> Tuple[float, float]:
        if raise_error:
            if len(variable_1) != len(variable_2):
                raise ValueError("Les longeurs des echantillons doivent être identiques")
        else:
            stat_value, p_value = stats.wilcoxon(variable_1, variable_2)
            return stat_value, p_value
        pass

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
            'normality_test': cls.normality_test
        }

        return {
         'Tests Paramétriques': tests_parametriques, 
         'Tests Non Paramétriques': tests_non_parametriques
         }

resultats_tests = TestStatistiques.classer_tests()
print(resultats_tests)
#
#
#je choisis un test au hasard



# STATISTIQUES
