class PatientsParser:
    """Class which handles parsing of patients files (/patient_lists) to the format, that can be later passed to ADNI's
    search form. Files in /patient_lists contain ADNI PIDs (IDs of the patients)"""

    @staticmethod
    def clearFile(fileName):
        """Delete formatted file's content"""
        open(fileName, 'w').close()

    @staticmethod
    def readPatients(fileName):
        """Read from patients file"""
        file = open(fileName, 'r')
        return file.read().replace('\n', ',')[:-1]

    @staticmethod
    def writePatients(fileName, patients, patientClass):
        """Write to formatted file"""
        file = open(fileName, 'a')
        file.write(patientClass + ': ' + patients + '\n')
