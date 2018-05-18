import csv

#class UnicodeDictReader(csv.DictReader, object):
#
#    def next(self):
#        row = super(UnicodeDictReader, self).next()
#        return {unicode(key, 'utf-8'): unicode(value, 'utf-8') for key, value in row.iteritems()}

class Dataset(object):

    def __init__(self, language, amountdata):
        self.language = language

        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        devset_path = "datasets/{}/{}_Dev.tsv".format(language, language.capitalize())
        testset_path = "datasets/{}/{}_Test.tsv".format(language, language.capitalize())

        self.trainset = self.read_dataset(trainset_path, amountdata)
        self.devset = self.read_dataset(devset_path)
        self.testset = self.read_dataset(testset_path)

    def read_dataset(self, file_path, amountdata = 100):
        with open(file_path, encoding='utf-8') as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader]

        return dataset[:int((len(dataset)) * amountdata / 100)]
    
       