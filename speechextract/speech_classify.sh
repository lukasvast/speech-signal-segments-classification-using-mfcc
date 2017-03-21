#!/bin/bash
#Pokrece speech_recognition.py modul za klasifikaciju svih datoteka u trenutnom direktoriju

for i in `ls *.wav`; do python speech_recognition.py $i $1; done;
echo "Klasifikacija dovršena nad slijedećim testnim uzorcima:"
for i in `ls *.wav`; do echo $i; done;
