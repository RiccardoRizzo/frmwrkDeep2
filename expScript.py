import subprocess
import smtplib
import time

import argparse
import os

#import sys
#sys.path.append("./")
import mail
import datiEmail as dm

"""
Qui non devono andarci funzioni, si scrive sono la sequenza 
degli esperimenti insieme con i file parametri.
"""



#=======================================================================
if __name__ == '__main__':
    """
    Esegue gli esperimenti in serie 
    """
    
    parser = argparse.ArgumentParser(description=" Lancia la serie di esperimenti." )

    parser.add_argument('-lpar', required=True, type=str,
            help="nome del file con la lista degli esperimenti; \n \
                Il file .csv deve contenere una lista di nomi file yaml, \
                nella sola prima riga, \
                formattati secondo l'esempio in esempio.yaml")

    parser.add_argument('-outdir', required=True, type=str,
            help='nome della directory dove mettere i file errore e log')
    
    opt = parser.parse_args()

    dirTemp=opt.outdir

    fin = open(opt.lpar, "r")
    listaFile=fin.readline()
    fin.close()    

    # crea la lista dei nomi file da chiamare; il nome del file deve essere
    # piu' lungo di un carattere
    lfile = [ x.strip() for x in  listaFile.split(",") if len(x) > 2]


    if not os.path.exists(dirTemp):
        os.makedirs(dirTemp)

    dirTemp= dirTemp+"/"
    
    print("""\nINIZIO ESPERIMENTI -------------------------------------""")


    for filePar in lfile:


        #-------------------------------------------------------------------

        nomeEsperimento = "esperimento :" + filePar
        
        print("""\n::::::""" + nomeEsperimento + """ ::::::""")    
        exp= "python expLogic.py -par " + filePar + \
                                " > " + dirTemp + filePar + "_rapp.out" + \
                                " 2> "+ dirTemp + filePar+ "_rapp.err"
        ### tempo di run
        start_time = time.time()
        subprocess.call(exp, shell=True)
        runtime=(time.time() - start_time)
        
        print("--- %s seconds ---" % runtime)


        print(""":::::: postProcessing ::::::""")
        rendering= "python expRend.py -par " + filePar + \
                                " > " + dirTemp + filePar + "_rend_rapp.out" + \
                                " 2> "+ dirTemp + filePar+ "_rend_rapp.err"
        subprocess.call(rendering, shell=True)


        # email risultati
        subject = "Esperimenti " + filePar
        body = "finiti tutti gli esperimenti"
        mail.send_email(dm.user, dm.pwd, dm.recipient, subject, body)

        #-------------------------------------------------------------------
    
    

    
    
    
    print("""\nFINE ESPERIMENTI ---------------------------------------""")
    
    
