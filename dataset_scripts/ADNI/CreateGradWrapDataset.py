from DatasetSpecifier import *

DatasetSpecifier('GradWarp', [('D:/ADNI/AD/MRI/', 'D:/ADNI/GradWrap/AD/'),
                              ('D:/ADNI/NC/MRI/', 'D:/ADNI/GradWrap/NC/'),
                              ('D:/ADNI/pMCI/MRI/', 'D:/ADNI/GradWrap/pMCI/'),
                              ('D:/ADNI/sMCI/MRI/', 'D:/ADNI/GradWrap/sMCI/')]).run()
