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

import libRend as lr

"""
Script di elaborazione dei dati finali dell'esperimento

Matplotlib: https://matplotlib.org/
Bokeh : https://bokeh.pydata.org/en/latest/index.html#
holoview : http://holoviews.org/getting_started/index.html
"""



#===========================================================================
def main(fileParam):
    with open(fileParam, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    # legge il file di output dei programmi
    inFile=fout=cfg["file"]["dirOut"] + "/" + cfg["file"]["output"]+".hdf5" 
    #lr.esplora(inFile)
    
    with h5py.File(inFile, 'r') as f:
       
        
        # vettore con nomi classi
        nomiClassi=f["classes"]
        
        avg_tt=lr.dimensiona_matrici(cfg)    #
        avg_acc=lr.dimensiona_matrici(cfg)   # matrice con i valori di accuracy medi
        avg_pre=lr.dimensiona_matrici(cfg)
        avg_rec=lr.dimensiona_matrici(cfg)
        avg_f1=lr.dimensiona_matrici(cfg)



        nomi=[]
        # per ogni grandezza di kernel
        for ikk in range(len(cfg["parametri_rete"]["kernel_length"])):

            kk=cfg["parametri_rete"]["kernel_length"][ikk]
            nomeKK="k="+str(kk)



            # per ogni valore di epoca
            for iee in range(len(cfg["training"]["epochs"])):

                epoch=cfg["training"]["epochs"][iee]
                nomeEpoch = "epochs=" + str(epoch)



                # ricavo le serie di valori relative al fold
                tt, acc, f1, pre, rec = lr.leggeRisultatiClassificazione(f, cfg, nomeKK, nomeEpoch)

                # faccio le medie dei valori delle serie e memorizzo nella matrice
                avg_tt[ikk][iee] = np.average(tt)
                avg_acc[ikk][iee] = np.average(acc)
                avg_f1[ikk][iee] = np.average(f1)
                avg_pre[ikk][iee] = np.average(pre)
                avg_rec[ikk][iee] = np.average(rec)



    # disegno le superfici
    kernelDim=cfg["parametri_rete"]["kernel_length"]
    numEpochs=cfg["training"]["epochs"]

    titolo="Accuracy"
    lr.visSuperficie(cfg, kernelDim, numEpochs, avg_acc, "kernel dimension", "epochs", titolo)
    lr.visGrafici(cfg, kernelDim, numEpochs, avg_acc, "kernel dimension", "epochs", titolo)

    titolo = "Precision"
    lr.visSuperficie(cfg, kernelDim, numEpochs, avg_pre, "kernel dimension", "epochs", titolo)
    lr.visGrafici(cfg, kernelDim, numEpochs, avg_pre, "kernel dimension", "epochs", titolo)

    titolo = "Recall"
    lr.visSuperficie(cfg, kernelDim, numEpochs, avg_rec, "kernel dimension", "epochs", titolo)
    lr.visGrafici(cfg, kernelDim, numEpochs, avg_rec, "kernel dimension", "epochs", titolo)

    titolo = "F1"
    lr.visSuperficie(cfg, kernelDim, numEpochs, avg_f1, "kernel dimension", "epochs", titolo)
    lr.visGrafici(cfg, kernelDim, numEpochs, avg_f1, "kernel dimension", "epochs", titolo)
    


#=======================================================================
if __name__ == '__main__':
    """
    Esegue gli esperimenti in serie 
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-par', required=True, type=str,
    # richiede il nome del file di parametri
            help='nomefile parametri')
    
    opt = parser.parse_args()
    
    main(opt.par)
