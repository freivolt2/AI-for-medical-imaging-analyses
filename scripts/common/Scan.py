"""This class main purpose is to limit risk of switching labels during preprocessing."""
class Scan:

    def __init__(self, scan, label):
        self._scan = scan
        self._label = label

    def getScan(self):
        return self._scan

    def getLabel(self):
        return self._label

    def setScan(self, scan):
        self._scan = scan

    def setLabel(self, label):
        self._label = label