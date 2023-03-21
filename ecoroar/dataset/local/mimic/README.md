This dataset contains a diabetes and and anemia classifcation task, using the MIMIC-III
dataset.

See https://physionet.org/content/mimiciii/1.4/ regarding MIMIC-III.
See https://mimic.physionet.org/gettingstarted/access/ for how to access MIMIC.

Once you have access, download the files with:
wget -N -c -np --user $USER --ask-password https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz -O $PERSISTENT_DIR/mimic/noteevents.csv.gz
wget -N -c -np --user $USER --ask-password https://physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz -O $PERSISTENT_DIR/mimic/diagnoses_icd.csv.gz
