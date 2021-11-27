from itertools import islice

if __name__ == '__main__':
    pruneScale = 1000
    csiPrunedFile = open('csi_pruned.csv', 'w')
    with open('csi.csv', 'r') as csiCsvFile:
        for csvItem in islice(csiCsvFile, pruneScale - 1, None, pruneScale):
            csiPrunedFile.write(csvItem)
    csiCsvFile.close()
    csiPrunedFile.close()