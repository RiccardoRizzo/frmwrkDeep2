"""
Libreria per il rendering dei risultati
"""

import yaml
import argparse
import os

import pickle
import numpy as np

import h5py

from sklearn.metrics import classification_report, accuracy_score, \
                            f1_score, precision_score, recall_score

import matplotlib.pyplot as plt


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

def printname(name):
    print(name)


def esplora(inFile):
    """
    Fornisce informazioni sul file HDF5 in ingresso

    Stampa il contenuto di inFile.attrs["info"]
    quindi naviga nella struttura del file e
    stampa i nomi delle strutture
    """
    with h5py.File(inFile, 'r') as f:
        # stampa il manuale del file
        print(f.attrs["info"])
        # attraversa e stampa
        f.visit(printname)


def vis_boxplot(dati, nomi, titolo, cfg):
    """
    Calcola il boxplot relativo al vettore dati
    passato.

    In nomi va il nome di ogni categoria sull'asse x

    """

    fig7, ax7 = plt.subplots()

    ax7.set_title(titolo)

    ax7.boxplot(dati)
    ax7.set_xticklabels(nomi)  # i nomi degli assi vanno qui

    # salva la figura
    nomeFileFig = cfg["file"]["dirOut"] + "/" + cfg["file"]["output"] + titolo + ".png"
    plt.savefig(nomeFileFig, dpi=300)

    # plt.show() # per la visualizzazione


def dimensiona_matrici(cfg):
    """
    Ricava dalla struttura cfg la dimensione delle matrici

    :param cfg: struttura con tutti i parametri degli esperimenti da cui
                ricavare la dimensione delle matrici
    :return:    matrice numpy delle dimensioni corrette
    """
    # suppongo R=X C=Y
    dim_R = len(cfg["parametri_rete"]["kernel_length"])  # sulle R i valori di dim kernel
    dim_C = len(cfg["training"]["epochs"])

    return np.ndarray((dim_R, dim_C))


def leggeRisultatiClassificazione(fileHDF5, cfg, nomeKK, nomeEpoch):
    """
    Ricava i parametri della classificazione usando il file HDF5

    Si suppone che siano stati effettuati degli esperimenti in un k-fold, quindi
    si legge la struttura fileHDH5 fino ad arrivare al fondo, dove l'ultimo
    livello rappresenta il fold. Quindi si crea una lista dei
    parametri di valutazione; ogni valore delle lista corrisponde al risultato in un fold

    :param fileHDF5: puntatore al file HDF5
    :param cfg:      struttura di configurazione dell'esperimento
    :param nomeKK:  uno dei nomi della gerarchia del file HDF5
    :param nomeEpoch:   altro nome della gerarchia HDF5
    :return:    lista di parametri di misura della classificazione, al momento
                tt, acc, f1, pre, rec    cioe'
                tempo, accuracy, F1, precision, recall
    """

    tt = []
    acc = []  # vettore accuracy
    pre = []
    rec = []
    f1 = []

    for i in range(cfg["training"]["k_fold"]):
        nome = "fold_" + str(i)
        x = fileHDF5[nomeKK][nomeEpoch][nome]["tempo_training"].value
        tt.append(x)

        pred = fileHDF5[nomeKK][nomeEpoch][nome]["y_pred"].value
        true = fileHDF5[nomeKK][nomeEpoch][nome]["y_true"].value

        # print(classification_report(true, pred, target_names=nomiClassi))

        acc.append(accuracy_score(true, pred))
        f1.append(f1_score(true, pred))
        pre.append(precision_score(true, pred))
        rec.append(recall_score(true, pred))

    tt = np.asarray(tt)
    acc = np.asarray(acc)
    f1 = np.asarray(f1)
    pre = np.asarray(pre)
    rec = np.asarray(rec)

    return tt, acc, f1, pre, rec


def visSuperficie(cfg, X, Y, matrice, nomeX, nomeY, titolo):
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, matrice, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel(nomeX)
    ax.set_ylabel(nomeY)
    ax.set_title(titolo)
    # salva la figura
    nomeFileFig = cfg["file"]["dirOut"] + "/" + cfg["file"]["output"] + titolo + ".jpg"
    plt.savefig(nomeFileFig, dpi=300)
    #plt.show()
    plt.close()


def visGrafici(cfg, X, Y, matrice, nomeX, nomeY, titolo):

    fig = plt.figure()
    x = X
    tratto = ["k", "k--", "k:"]
    for i in range(len(matrice)):
        y = matrice[i]
        y_lab = str(Y[i]) + " epochs"
        plt.plot(x, y, tratto[i], label=y_lab)

    plt.xlabel(nomeX)    
    plt.title(titolo)

    plt.legend()

    # salva la figura
    nomeFileFig = cfg["file"]["dirOut"] + "/" + cfg["file"]["output"] + titolo + "_grafico.jpg"
    plt.savefig(nomeFileFig, dpi=300)
    plt.close()

    #plt.show()
    