from features import mfcc
from numpy.linalg import inv
from Levenshtein import distance as lev
import os, fnmatch, sys, json
import scipy.io.wavfile as wav
import numpy as np
import scipy.spatial.distance as dist

#disablea sve warninge
import warnings
warnings.filterwarnings("ignore")

#alternativni i prvi izbor prepoznatog glasa te string prepoznatih glasova
poljeAlt=[]
poljeOrig=[]
prepoznatiGlasovi=""

#deklariranje direkotorija mean vektora, matrica kovarijacije i glasova
dirGlasoviMean="../mfccfeatures/glasoviMean/"
dirGlasoviCov="../mfccfeatures/glasoviCov/"
dirGlasovi="../mfccfeatures/glasovi/"

#naziv audio zapisa i distance iz argumenta
if len(sys.argv)==3 and (str(sys.argv[2])=="M" or str(sys.argv[2])=="E"):
    audioSignal=str(sys.argv[1])
    labDatoteka=audioSignal.split(".")[0]+".lab"
    odabirDistance=str(sys.argv[2])
    print "Obradujem audio zapis: "+str(sys.argv[1])
else:
    print "Krivi poziv funkcije!\n"
    print "Primjer poziva: python speech_recognition.py sm04010105107.wav M\n"
    sys.exit()

#inicijalizacija direktorija za output
if not os.path.exists("out"):
    os.makedirs("out")

#nadi sve glasove
glasovi =[]
for file in os.listdir(dirGlasovi):
    if fnmatch.fnmatch(file, '*.txt'):
        glasovi.append(file)
glasovi.sort()
print "Nadeno %d glasova!"%len(glasovi)
print glasovi
print ""

#izvlacanje mfcc koeficijenata signala govora i spremanje u mfcc_feat
(rate,sig) = wav.read(audioSignal)
mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,preemph=0.99)

#stvaranje datoteke za zapis prepoznatih glasova
fHandle = open("out/"+audioSignal.split(".")[0]+"GlasoviRecog.txt", "w")
fHandle.write("Prepoznati glasovi za audio datoteku: "+audioSignal+"\n")
fHandle.write("Trajanje datoteke: %dms\n" %(len(mfcc_feat)*10))
fHandle.write("Svaki redak predstavlja vremenski okvir od 10ms\n")
fHandle.write("Prvi izbor, Drugi izbor, Treci izbor / Distanca za prvi izbor, Distanca za drugi izbor, Distanca za treci izbor\n")
fHandle.close()

#prepoznavanje mahalanobisovom ili euklidovom distancom
print "Prepoznavanje glasova!"

#petlja koja prolazi kroz sve vektore karakteristika u mfcc_feat matrici
for y in range(0, len(mfcc_feat)):
    
    #dictonary sa vrijednostima distanci pojedinih glasova
    distances={}
    
    #petlja koja prolazi koroz sve glasove
    for x in range(0, len(glasovi)):
        
        #ucitaj inverz kovarijacijske matrice i mean vektor glasa
        invcov=inv(np.loadtxt(dirGlasoviCov+glasovi[x], delimiter=","))
        mean=np.loadtxt(dirGlasoviMean+glasovi[x], delimiter=",")
        
        #odabir distance za izracune
        if odabirDistance=="M":
            #mahalanobisova distanca
            distance=dist.mahalanobis(mean, mfcc_feat[y:y+1], invcov)   
        elif odabirDistance=="E":
            #euklidova distanca
            distance=dist.euclidean(mean, mfcc_feat[y:y+1])
        
        #spremi distance i glasove u dictionary 
        distances[glasovi[x]]=distance
        
    #nadi najbolje predpostavke za prepoznati glas
    recog = sorted(distances, key=distances.get)[0:5]
    
    #ispis najboljih predpostavki i distance za najbolju predpostavku
    print recog
    print distances.get(recog[0])
    
    #spremanje alternativnog i prvog izbora glasa u polja
    poljeAlt.append(recog[1][:-4])
    poljeOrig.append(recog[0][:-4])
    
    #formatiranje ispisa u tekstualnu datoteku sa tri najbolja izbora i njihovim udaljenostima
    fHandle = open("out/"+audioSignal.split(".")[0]+"GlasoviRecog.txt", "a")
    fHandle.write(str(y)+": ")
    fHandle.write(recog[0].split(".")[0]+", "+recog[1].split(".")[0]+", "+recog[2].split(".")[0])
    fHandle.write(" / "+str(distances.get(recog[0]))+", "+str(distances.get(recog[1]))+", "+str(distances.get(recog[2])))
    fHandle.write("\n")
    fHandle.close()    
print "SUCCESS!"

#izvlacenje glasova iz .lab datoteke za usporedbu preciznosti
labData=np.genfromtxt(labDatoteka, dtype="str")

#stavljanje u string
stringLab=""
for x in range(len(labData)):
    slovo=labData[x][2]
    if slovo[-1]==":":
        stringLab=stringLab+str(slovo[:-1])
    elif slovo=="sil":
        stringLab=stringLab+""
    elif slovo=="uzdah":
        stringLab=stringLab+""
    else:       
        stringLab=stringLab+str(labData[x][2])

#filtriranje poljaOrig
for x in range(len(poljeOrig)):
    slovo=str(poljeOrig[x])
    if slovo[-1]== ":":
        poljeOrig[x]=poljeOrig[x][:-1]
    else:   
        poljeOrig[x]=poljeOrig[x]
        
#filtriranje poljaAlt
for x in range(len(poljeAlt)):
    slovo=str(poljeAlt[x])
    if slovo[-1]== ":":
        poljeAlt[x]=poljeAlt[x][:-1]
    else:   
        poljeAlt[x]=poljeAlt[x]

#ciscenje poljaOrig od ponavljajucih glasova i krivi prepoznatih glasova zbog suma 
clean=[]
for x in range(2,len(poljeOrig)-1):
    if (poljeOrig[x]!=poljeOrig[x-1] and poljeOrig[x-1]==poljeOrig[x-2] and poljeOrig[x-1]==poljeOrig[x-3]) or (poljeOrig[x]!=poljeOrig[x-1] and poljeOrig[x-1]==poljeOrig[x-2] and poljeOrig[x-1]==poljeOrig[x+1]):
        if len(clean)>2:
            if clean[len(clean)-1]!=poljeOrig[x-1]:
                clean.append(poljeOrig[x-1])
        else:
            clean.append(poljeOrig[x-1])

#stavljanje u string i ciscenje sil, uzdah, buka i greska segmenata
stringGlasova=""
for x in range(len(clean)):
    if str(clean[x])!="sil" and str(clean[x])!="uzdah" and str(clean[x])!="buka" and str(clean[x])!="greska":
        stringGlasova=stringGlasova+clean[x]
 
#ispis prepoznatog i originalnog stringa glasova       
print "\nISPIS ORIGINAL I PREPOZNATO"
print stringLab
print stringGlasova

#racunanje preciznosti s levenshtein distancom
edit_dist=lev(stringLab, stringGlasova)
print "\nPRECIZNOST"
if odabirDistance=="M":
    print "KORISTECI MAHALANOBISOVU UDALJENOST"
elif odabirDistance=="E":
    print "KORISTECI EUKLIDOVU UDALJENOST"
print str(100-float(edit_dist)/len(labData)*100)+"%"

#formatiranje i zapisivanje u out.lab
start=0
distribution = len(mfcc_feat)*100000/len(clean)
text_file = open("out/"+audioSignal.split(".")[0]+"Out.lab", "w")
text_file.write("")
text_file.close()
for x in range(len(clean)):
    text_file = open("out/"+audioSignal.split(".")[0]+"Out.lab", "a")
    text_file.write(str(start)+" "+str(start+distribution)+" "+str(clean[x])+"\n")
    text_file.close()
    start=start+distribution    

def klasifikacija():
    
    #prikazi broj pojedinih slova u original i prepoznatom stringu
    brojSlovaLab = {}
    brojSlovaGlasova={}
    for slovo in stringLab:
        brojSlovaLab[slovo] = stringLab.count(slovo)    
    for slovo in stringGlasova:
        brojSlovaGlasova[slovo] = stringGlasova.count(slovo)    
    print brojSlovaLab
    print brojSlovaGlasova
    
    #klasifikacija pojedinih glasova
    klasifikacijaPrepoznatihGlasova={}
    tempKlasifikacija={}
    for key in brojSlovaLab:
        if key in brojSlovaGlasova:
            if brojSlovaLab[key]<brojSlovaGlasova[key]:
                tempKlasifikacija[key]=round(float(brojSlovaLab[key])/brojSlovaGlasova[key]*100)
            else:
                tempKlasifikacija[key]=round(float(brojSlovaGlasova[key])/brojSlovaLab[key]*100)
        else:
            tempKlasifikacija[key]=float(0)
            
    if not os.path.exists("out/klasifikacija.txt"):
        klasifikacijaPrepoznatihGlasova=tempKlasifikacija
        json.dump(klasifikacijaPrepoznatihGlasova,open("out/klasifikacija.txt","w"), sort_keys=True)
    else:
        klasifikacijaPrepoznatihGlasova=json.load(open("out/klasifikacija.txt"))
        for key in tempKlasifikacija:
            if key in klasifikacijaPrepoznatihGlasova:
                klasifikacijaPrepoznatihGlasova[key]=float(klasifikacijaPrepoznatihGlasova[key]+tempKlasifikacija[key])/2
            else:
                klasifikacijaPrepoznatihGlasova[key]=tempKlasifikacija[key]
        json.dump(klasifikacijaPrepoznatihGlasova,open("out/klasifikacija.txt","w"), sort_keys=True)
    return 1
        
klasifikacija()