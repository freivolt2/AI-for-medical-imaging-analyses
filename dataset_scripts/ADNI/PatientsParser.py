class PatientsParser:

    @staticmethod
    def clearFile(fileName):
        open(fileName, 'w').close()

    @staticmethod
    def readPatients(fileName):
        file = open(fileName, 'r')
        return file.read().replace('\n', ',')[:-1]

    @staticmethod
    def writePatients(fileName, patients, patientClass):
        file = open(fileName, 'a')
        file.write(patientClass + ': ' + patients + '\n')