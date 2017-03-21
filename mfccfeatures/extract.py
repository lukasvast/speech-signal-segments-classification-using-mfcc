from features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os, sys, shutil, fnmatch

def extract_mfcc():
    
    #nadi sve .wav fileove u audio direktoriju
    audioDir = "audio/"
    if not os.path.exists(audioDir):
        os.makedirs(audioDir)
        print "Ne postoji audio direktorij!"
    audioFiles =[]
    for file in os.listdir(audioDir):
        if fnmatch.fnmatch(file, '*.wav'):
            audioFiles.append(file)
    
    #ispis broja pronadenih .wav datoteka u audio direktoriju 
    print ""      
    print "Pronasao sam %d audio zapisa u %s direktoriju!" % (len(audioFiles), audioDir )
    print ""
    
    #petlja koja prolazi kroz sve .wav datoteke unutar audio direktorija
    for x in range(0, len(audioFiles)):
        
        #ime .wav audio zapisa, ime .png grafa, ime .txt formata audio zapisa, ime direktorija za pohranu svega
        filename = audioFiles[x]
        floatFile = filename.split(".")[0] + ".txt"
        directory = audioDir + filename.split(".")[0]
        
        #provjere dali postoji datoteka ili direktoriji
        if not os.path.isfile(audioDir+filename):
            sys.exit("File does not exist!")
        if not os.path.exists(directory):
            os.makedirs(directory)        
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        
        #citanje wav datoteke i slanje za izracun mfcc 
        (rate,sig) = wav.read(audioDir+filename)
        mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,preemph=0.99)
        
        #ispis imena izracunatih .wav datoteka
        print "MFCC karakteristike izracunate za %s!" %filename
        
        #spremanje signala u float formatu i mfcc znacajki u .txt datoteke
        np.savetxt(directory+"/"+floatFile, sig, fmt="%.4f")
        np.savetxt(directory+"/mfcc_features.txt", mfcc_feat, fmt="%.16f", delimiter=",")
    
    #ispis broja .wav datoteka iz kojih smo izvukli MFCC
    print "Ukupno %d MFCC karakteristika izracunato!" %len(audioFiles)
    
    return 1    
 
def extract_vox(): 
    
    #izlistaj sadrzaj audio direktorija u audioDirFeatures listu
    audioDirFeatures =[]
    for file in os.listdir("audio"):
        if not fnmatch.fnmatch(file, '*.wav'):
            audioDirFeatures.append(file)
    
    #kreiraj direktorij glasovi ako ne postoji
    if not os.path.exists("glasovi"):
        os.makedirs("glasovi")
    
    #ispis broja mfcc matrica koji odgovara broju direktorija u audio direktoriju
    print ""
    print "Broj mfcc matrica za obradu: %d" %len(audioDirFeatures)
    print ""
    
    #definiraj imena potrebnih direktorija za input
    labDir="lab/"
    audioDir="audio/"
    
    #petlja koja prolazi kroz sve matrice mfcc koeficijenata u direktoriju glasovi
    for x in range(0,len(audioDirFeatures)): 
        
        #definiramo ime lab datoteke u kojoj se nalaze izgovoreni glasovi i njihov vremenski raspon u audio datoteci
        #definiramo odgovarajucu mfcc datoteku koja pripada lab fileu
        labFile = labDir+audioDirFeatures[x]+".lab"
        mfccFile = audioDir+audioDirFeatures[x]+"/mfcc_features.txt"
        brojStvorenihFileova=0
        
        #loadamo taj lab file
        labData=np.genfromtxt(labFile, dtype="str")
        
        #loadamo mfcc file
        mfccData=np.loadtxt(mfccFile, delimiter=",")
        
        #prebacujemo zapis iz str u float i konvertiramo vremensku jedinicu
        #zapis prilagodujemo extract_mfcc modulu i matrici mfcc tako da vremenski raspon odgovoara redcima matrice mfcc
        for x in range(0, labData.size/3):    
            labData[x][0]=int(round(float(labData[x][0])/100000))
            labData[x][1]=int(round(float(labData[x][1])/100000))
               
        #kreiramo fileove glasova ako vec ne postoje u stavljamo ih u folder glasovi
        for x in range(0, labData.size/3):
            if not os.path.exists("glasovi/"+labData[x][2]+".txt"):
                open("glasovi/"+labData[x][2]+".txt", 'w').close()
                brojStvorenihFileova=brojStvorenihFileova+1                
        
        #otvori file za append s imenom glasa koji se nalazi u trecem stupcu matrice labData
        for x in range(0, labData.size/3):
            pocetak=int(labData[x][0])
            kraj=int(labData[x][1])
            fHandle=open("glasovi/"+labData[x][2]+".txt","a")
            np.savetxt(fHandle, mfccData[(pocetak):(kraj)], fmt="%.16f", delimiter=",")
            fHandle.close
        
        #ispis karakteristika koje smo izvukli za svaki pojedini audio zapis
        print "Uspjeh! Broj novih stvorenih glasova iz %s: %d" % (labFile, brojStvorenihFileova)
        print "Sveukupan broj glasova u fileu: %d" % float(labData.size/3)
    
    return 1
        
def mean_vectors():
    
    #ispis sta radi program
    print ""
    print "Racunanje vektora srednjih vrijednosti!"
    
    #kreiranje direktorija za vektore srednjih vrijednosti glasova
    if os.path.exists("glasoviMean"):
        shutil.rmtree("glasoviMean")
    os.makedirs("glasoviMean")
    
    #izlistava direktorij glasovi i sprema nazive u listu
    glasoviFeatures =[]
    for file in os.listdir("glasovi/"):
        glasoviFeatures.append(file)
        glasoviFeatures.sort()
    
    #ispis broja detektiranih glasova u glasovi direktoriju
    print "Detektirano %d glasova/zvukova!" %len(glasoviFeatures)
    print ""    
    
    #racunanje vektora srednje vrijednosti za svaku matricu mfcc glasa 
    for x in range(0, len(glasoviFeatures)):
        #ispis koji se trenutno glas obraduje    
        print "Izracun vektora srednje vrijednosti za glas %s" %glasoviFeatures[x]
        
        #racunanje vektora srednje vrijednosti mfcc matrice glasova
        mfccData=np.loadtxt("glasovi/"+glasoviFeatures[x], delimiter=",")
        meanVector=np.mean(mfccData, axis=0)
        
        #spremanje vektora srednje vrijednosti u direktorij glasoviMean
        np.savetxt("glasoviMean/"+glasoviFeatures[x], meanVector, fmt="%.16f")
        
    return 1
    
def cov_matrix():
    
    #ispis sta radi program
    print ""
    print "Racunanje matrica kovarijacije!"    
    
    #kreiranje direktorija za matrice kovarijacije glasova
    if os.path.exists("glasoviCov"):
        shutil.rmtree("glasoviCov")
    os.makedirs("glasoviCov")    
    
    #izlistava direktorij glasovi i sprema nazive u listu
    glasoviFeatures =[]
    for file in os.listdir("glasovi/"):
        glasoviFeatures.append(file)
        glasoviFeatures.sort()
    
    #ispis broja detektiranih glasova u glasovi direktoriju
    print "Detektirano %d glasova/zvukova!" %len(glasoviFeatures)
    print "" 
    
    #racunanje matrice kovarijacije 
    for x in range(0, len(glasoviFeatures)):
        #ispis koji se trenutno glas obraduje    
        print "Izracun matrice kovarijacije za glas %s" %glasoviFeatures[x]
        
        #racunanje matrice kovarijacije glasa
        mfccData=np.loadtxt("glasovi/"+glasoviFeatures[x], delimiter=",")
        covMatrix=np.cov(mfccData,rowvar=False)
        
        #spremanje matrica kovarijacije glasova u glasoviCov
        np.savetxt("glasoviCov/"+glasoviFeatures[x], covMatrix, fmt="%.16f", delimiter=",")
    
    return 1

#pokretanje svih skripti/modula sekvencijalno    
flag = 1    
if flag==1: 
    flag=extract_mfcc()
else:
    print "Greska u inicijalizaciji varijabli!"
 
if flag==1:
    flag=extract_vox()
else:
    print "Greska u extract_mfcc funkciji!"

if flag==1:
    flag=mean_vectors()
else:
    print "Greska u extract_vox funkciji!"

if flag==1:
    flag=cov_matrix()
else:
    print "Greska u mean_vectors funkciji!"
    
if flag==1:
    print ""
    print "SUCCESS!"
    print ""
else:
    print "Greska u cov_matrix funkciji!"            