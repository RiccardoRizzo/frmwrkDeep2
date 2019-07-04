import yaml
import argparse
import os
import time

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold

# memorizzazione delle uscite in hdf5
import h5py


import conv_model as cm

#-----------------------------------------------------------------------------
def dividiDataset(dataframe, inCol, outCol, k):
    """
    Divide il file di dati in input per fare il k-fold 
    Restituisce :
    
    fold :  una struttura dati che contiene la lista degli indici 
            per ogni fold
            Per il fold i si avra':
            
            fold[i]["train"] = <lista di indici relativa al 
                                train del fold i>
            fold[i]["test"] = <lista di indici relativa al test
                                del fold i>
                                
    
    l'uso e'con 
        
        inCol="4x"
        outCol="label"
        
        Per il traning del fold i 
        X_train=inData.loc[fold[i]["train"]][inCol].values
        Y_train=inData.loc[fold[i]["train"]][outCol].values
        doce X e' l'input e Y e' l'output
        
    """
    
    # selezioni i dati X: ingresso, Y: uscita
    X=dataframe[inCol].values
    Y=dataframe[outCol].values
    
    # esegue la divisione per il kfold con mescolamento
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    
    fold=[]
    for train, test in skf.split(X, Y):
        v={"train": train, "test": test}
        fold.append(v)
    
    return fold
    
#-----------------------------------------------------------------------------    
def estraiDati(listaIndici, dataframe, etichettaInput, etichettaOutput):
    """
    Estrae i dati dal dataframe e prepara i vettori
    di input e di output per l'apprendimento. 
    """
     
    # seleziona le righe e colonne corrette
    X=dataframe.ix[listaIndici, etichettaInput].values
    Y=dataframe.ix[listaIndici, etichettaOutput].values
    
    X, Y = reshape(X, Y)
    
    return X, Y

#-----------------------------------------------------------------------------
def createModel(cfg, k, X_train):
    """
    crea il modello da addestrare
    
    """
    inputShape=np.shape(X_train[0])
    
    # dimensiona il modello 
    modello = cm.build_conv_2conv(inputShape,
                                cfg["parametri_rete"]["filters1"], 
                                k, #grandezza del kernel  
                                cfg["parametri_rete"]["filters2"]
                                )
    return modello


#-----------------------------------------------------------------------------
def trainModel(modello, cfg, X_train, Y_train, numEpoche):
    """
    Esegue il training del modello usando i parametri letti in cfg
    I due argomenti X_train e Y_train devono gia' essere nel 
    formato giusto. Il formato dell'input e' letto direttamente da X_train.
    
    In output si ottengono:
        il modello addestrato
        il tempo necessario all'addestramento
    """
    
    
    start_time = time.time()
    # esegue il training
    modello.fit(X_train, Y_train,
                epochs = numEpoche,
                batch_size=cfg["training"]["batch_size"], 
                validation_split=cfg["training"]["validation_split"], 
                verbose=cfg["training"]["verbose"])
    end_time=time.time()
    
    # passa il modello
    return modello, end_time-start_time
        
    
        
#-----------------------------------------------------------------------------        
def reshape(XI, YI):
    """
    Esegue il reshape delle matrici di input
    """
    # stabilisco le dimensioni dei vari elementi
    batchLen=len(XI)
    shapeXI=np.shape(XI[0])
    shapeYI=np.shape(YI)
    
    # il modello accetta gli input trasposti, quindi devo trasporre 
    # ogni singolo elemento del batch
    in_rig=shapeXI[1]
    in_col=shapeXI[0]
    
    out_rig=1

    
    X=np.empty(shape=(XI.shape[0], in_rig, in_col))
    for i in range(XI.shape[0]):
        X[i,:,:] = XI[i].transpose()
 
    Y=np.empty(shape=(YI.shape[0], out_rig))
    for i in range(YI.shape[0]):
        Y[i,:] = YI[i].transpose()
    
    return X, Y
    


#-----------------------------------------------------------------------------
def calcConfMatr(y_pred, y_true, nClassi):
    """
    A partire dai vettori delle classi test (reali) e predette
    calcola la matrice di confusione
    
    La disposizione della matrice:
        righe : classe reale
        colonne: predizione della rete

    :param y_pred: output predetto dalla rete
    :param y_true: output reale
    :param nClassi: numero delle classi previste
    :return matrice di confusione
    """
    #confMat=np.zeros( ( len(y_pred), len(y_true) ) )
    confMat=np.zeros( ( nClassi, nClassi ) )
    for i in range(len(y_pred)):
        r=int(y_true[i])
        c=int(y_pred[i])
        confMat[r][c] = confMat[r][c] + 1
        
    return confMat
    


#-----------------------------------------------------------------------------
def main(fileParam):
    """
    Legge il file di parametri, confgura i programmi ed esegue 
    l'esperimento. 

    In questa versione il numero di epoche aumenta con l'aumentare
    della dimensione dei kernel.

    Gli elementi di nome "foldX" contengono la matrice di confusione 
    memorizzata come :
        classe reale sulle righe
        classe predeta sulle colonne
        
    time:
        tempo di training in secondi
        
    data:
        data dell'esperimento
    """
    # la doc string sara' inserita nel file dei risultati

    with open(fileParam, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    
    # apro il dataframe e carico i dati
    inData = pd.read_pickle(cfg["file"]["input"])

    # se la directory di output non esiste la crea
    if not os.path.exists(cfg["file"]["dirOut"]):
        os.makedirs(cfg["file"]["dirOut"])

    # definisco le colonne di input e di output
    inCol="4x"
    outCol="label"      
    
    k=10
    # crea la lista di indici per il k fold
    fold=dividiDataset(inData, inCol, outCol, k)
    
    # history = TrainingHistory()
    
    classi=["linker", "nucleosome"]
    
    # al momento per l'output uso un dict
    # scrive nella variabile di output 
    # la descrizione del programma
    # e della struttura dell'output
    out={"struttura": main.__doc__}   

    print("poco prima di aprire il file di out")
    fout=cfg["file"]["dirOut"] + "/" + cfg["file"]["output"]+".hdf5"

    
    # apre il file 
    with h5py.File(fout) as f:
        # inserisce le informazioni nel file dei risultati
        infor = main.__doc__
        f.attrs["info"] = infor
        
        classes = np.string_("classes")
        nucleosome = np.string_("nucleosome")
        linker = np.string_("linker")
        
        f.create_dataset(classes, data=[nucleosome, linker])
        
        for kk in cfg["parametri_rete"]["kernel_length"]:

            print("crea il gruppo relativo alla dimensione kk del kernel")
            grpK = f.create_group("k=" + str(kk))

            for epochs in cfg["training"]["epochs"]:

                print("crea il gruppo relativo al numero di epoche")
                grpE = grpK.create_group("epochs=" + str(epochs))

                # chiamata fittizia per ottenere un vettore di ingresso
                # X_train da cui prelevare la dimensione degli ingressi
                X_train, Y_train = estraiDati(fold[0]["train"], inData, inCol, outCol)

                # inizia il ciclo del k-fold
                for i in range(len(fold)):
                    # crea il sottogruppo del fold
                    grp = grpE.create_group("fold_" + str(i))

                    X_train, Y_train = estraiDati(fold[i]["train"], inData, inCol, outCol)
                    X_test, Y_test = estraiDati(fold[i]["test"], inData, inCol, outCol)

                    # dimensiono il modello
                    modello = createModel(cfg, kk, X_train)

                    # training del modello
                    modello, tempo_training=trainModel(modello, cfg, X_train, Y_train, epochs)

                    print(tempo_training)

                    # validazione
                    y_true = np.asarray([x for x in Y_test])
                    y_pred = cm.predict_classes(modello, X_test)

                    # calcola la matrice di confusione
                    numClassi=2
                    m=calcConfMatr(y_pred, y_true, len(classi))

                    grp.create_dataset("confMat", data=m)
                    grp.create_dataset("tempo_training", data = tempo_training)

                    grp.create_dataset("y_true", data=y_true)
                    grp.create_dataset("y_pred", data=y_pred)


                # memorizzazione di out come pickle
                timestr = time.strftime("%Y%m%d-%H%M%S")
                grpK.attrs["data"]=timestr



if __name__ == '__main__':
    """
    Prende il input il nome del file di parametri, scritto secondo 
    YAML e lo passa al main che contiene la logica dell'esperimento
    
    Il programma e' richiamato da expScript.py
    
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-par', required=True, type=str,
    # richiede il nome del file di parametri
            help='nomefile parametri')
    
    opt = parser.parse_args()
    
    
    main(opt.par)
    
    
