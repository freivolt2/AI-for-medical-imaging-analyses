from PatientsParser import PatientsParser

NC = 'NC'
AD = 'AD'
S_MCI = 'sMCI'
P_MCI = 'pMCI'

PATIENTS = 'parsed_patients.txt'
PATIENT_LIST = '_patient_list.txt'
LISTS_PATH = 'patient_lists/'

PatientsParser.clearFile(PATIENTS)

pMCI = PatientsParser.readPatients(LISTS_PATH + P_MCI + PATIENT_LIST)
PatientsParser.writePatients(PATIENTS, pMCI, P_MCI)

sMCI = PatientsParser.readPatients(LISTS_PATH + S_MCI + PATIENT_LIST)
PatientsParser.writePatients(PATIENTS, sMCI, S_MCI)

pMCI = PatientsParser.readPatients(LISTS_PATH + AD + PATIENT_LIST)
PatientsParser.writePatients(PATIENTS, pMCI, AD)

pMCI = PatientsParser.readPatients(LISTS_PATH + NC + PATIENT_LIST)
PatientsParser.writePatients(PATIENTS, pMCI, NC)
