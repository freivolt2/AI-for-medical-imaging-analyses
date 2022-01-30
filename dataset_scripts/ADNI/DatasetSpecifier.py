from os import walk
from shutil import copyfile


class DatasetSpecifier:
    """Prior to this work we downloaded, not only raw MRI scans, but also all kinds of postprocessed scans. This
    class serves to select specifc types of postprocessed data from dataset, thus creates a new dataset of such
    selected data."""

    def __init__(self, postprocessingType, directions):
        self._postprocessingType = postprocessingType
        self._directions = directions
        self.cleanLogs()

    def getAllSubjectIds(self, rootdir):
        """Gets PIDs of all subjects from the subdirectories in 'rootdir' directory"""
        for _, directories, _ in walk(rootdir):
            return directories

    def getAllDirectories(self, directory):
        """Gets all directories in the directory"""
        return [x[0].replace('\\', '/') for x in walk(directory)]

    def getFullDirectory(self, subdirectories, postprocessingType):
        """Gets full directory to the files based on postprocessingType"""
        sub = [x for x in subdirectories if '/' + postprocessingType + '/' in x]
        if not sub:
            return None
        return sub[-1] + '/'

    def getFiles(self, directory):
        """Gets file paths form the directory"""
        return [x[2] for x in walk(directory)][0]

    def getNiiFiles(self, files):
        """Gets all NIfTI files from the list of files"""
        return [x for x in files if '.nii' in x]

    def cleanLogs(self):
        """Cleans the log file"""
        file_object = open('dataset_specifier_log.txt', 'w')
        file_object.write('Excluded subjects:\n')
        file_object.close()

    def log(self, subjectId):
        """Logs the excluded files"""
        file_object = open('dataset_specifier_log.txt', 'a')
        file_object.write(subjectId + '\n')
        file_object.close()

    def createSpecializedDataset(self, rootDirectory, copyDirectory):
        """Creates specialized dataset by copying chosen files from 'rootDirectory' to 'copyDirtectory'"""
        subjectIds = self.getAllSubjectIds(rootDirectory)
        for subjectId in subjectIds:
            subdirectories = self.getAllDirectories(rootDirectory + subjectId)
            fullDirectory = self.getFullDirectory(subdirectories, 'MPR__' + self._postprocessingType)
            if fullDirectory is None:
                fullDirectory = self.getFullDirectory(subdirectories, 'MPR-R__' + self._postprocessingType)
            if fullDirectory is None:
                self.log(subjectId)
                continue
            files = self.getFiles(fullDirectory)
            file = self.getNiiFiles(files)[0]
            copyfile(fullDirectory + file, copyDirectory + file)

    def run(self):
        """Runs the process of creating specialized dataset"""
        for direction in self._directions:
            print('Processing ' + direction[0] + '...')
            self.createSpecializedDataset(direction[0], direction[1])
            print('Processing ' + direction[0] + ' finished.')
        print('All processes finished.')
