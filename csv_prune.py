from itertools import islice

PRUNE_SCALE = 1000

if __name__ == '__main__':
    csiPrunedFile = open('csi_pruned.csv', 'w')
    with open('csi.csv', 'r') as csiCsvFile:
        for csvItem in islice(csiCsvFile, PRUNE_SCALE - 1, None, PRUNE_SCALE):
            csiPrunedFile.write(csvItem)
    csiCsvFile.close()
    csiPrunedFile.close()
