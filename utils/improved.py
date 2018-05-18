from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
import re
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

class Feature(object):
    def __init__(self):
        self.dic_feats = {}
        #Training features
    def train(self, features, label):
        if label == '0':
            label = '-1'
        for feat in features:
            if feat not in self.dic_feats:
                self.dic_feats[feat] = 1 * int(label)
            else:
                self.dic_feats[feat] += 1  * int(label)
        #transforminf features
    def transform(self, features):
        value = 0
        for feat in features:
            if feat in self.dic_feats:
                value += self.dic_feats[feat]
        return value

class Improved(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.Suff=['able', 'ible','acy','age','al', 'ial','ance','ence', 'dom','ical',  'ship','ate',
                       'ant', 'ent', 'ary', 'en', 'ful', 'fy', 'ic', 'ac', 'il', 'ile', 'ily', 'cracy',
                       'ile', 'ish','ism', 'ist', 'ity', 'ise', 'ive', 'ize', 'less','logy', 'hood','arian',
                       'ly', 'ment', 'ness', 'ous', 'tion', 'sion', 'ion','esque','pathy','trophy',
                       'ization','log','or','onym','ular','some','ware','wise','iatric','ette', 'est','emia', 
                       'eer','ative', 'acity']
            self.Preff=['in','ir', 'im', 'dis', 'mis', 'de', 'for', 'non', 'pre', 're', 'un',' non', 'anti', 'mal',
                        'dys', 'hyper', 'inter', 'intra','poly', 'oligo','  sub', 'ab', 'ad', 'ecto', 'post',
                        'endo', 'super', 'epi', 'hemi', 'ultra', 'uni', 'pro', 'multi', 'macro', 'micro', 'meta']
            self.vowelsTwo = re.compile(r'(?=([aeiouyAEIOUY]{2}))')
            with open("DaleChallEasyWordList.txt", "r", encoding = 'utf-8') as f:
                self.SimpleWord = f.readlines()
            self.lan = 'en'
            
        else:  # spanish
            self.avg_word_length = 6.2
            self.Preff=['an', 'ana', 'pari', 'peri', 'pro','eu','exo','epi','dis','anti','contra','auto','ente',
                          'des','entre','extra','extro', 'multi', 'macro', 'micro', 'meta', 'sobre', 'super','semi',
                          'sin','intro','inter','ambi','ex','re','sub','epi','co']
            self.Suff=['aco','aca','ana','ano','ario','aria','able','ción','dor','dora','ada','ado','era', 'ero','ismo', 
                         'mento','menta','triz','oso','osa','uro','ura','ficar','tud','able','dad','al','ar','logia',
                         'logo', 'tecnia', 'ismo', 'encia', 'dad', 'eza', 'or','anza', 'ista', 'il', 'udo','uda', 
                         'ear', 'ecer', 'ificar', 'izar','orrear', 'otear', 'uquear', 'iente', 'idor', 'ón', 'adizo',
                         'isma', 'és', 'ense', 'eño','ía']
            self.accents = re.compile('[áéíúóÁÉÍÓÚ]')
            self.vowelsTwo = re.compile(r'(?=([aeiouAEIOUáéíúóÁÉÍÓÚ]{2}))')
            self.SimpleWord = []
            self.lan = 'es'
        self.preprocess = re.compile(r'[\W\d_]', re.U)
        self.char_ngram = Feature()
        self.suff = Feature()
        self.preff = Feature()

###Models
#        self.modelLog = LogisticRegression(verbose = 0.01)
        if language == 'spanish':
            self.modelLog = tree.DecisionTreeClassifier(max_depth=5, criterion="entropy")
        else:
            self.modelLog= RandomForestClassifier(max_depth=13, criterion="entropy")
#        self.modelLog = AdaBoostClassifier()
#        self.modelLog = KNeighborsClassifier(15, algorithm = 'brute', p = 2)
#        self.modelLog = MLPClassifier(hidden_layer_sizes = (100, 150, 50), activation = 'tanh', learning_rate = 'adaptive')

    def extract_features(self, word, sent, start_offset):
#        word = re.sub(self.preprocess, '', word)
        len_chars = len(word) / self.avg_word_length #Avereged length of word
        len_tokens = len(word.split(' '))#length of word
        position_norm = len(sent[:int(start_offset)].split())/len_tokens
        
        
        tokens = re.sub(self.preprocess, '', word).lower().split()
        trigs = []
        suffix_list = []
        preffix_list = []
        InWikiSimple = 0
        freq = 0
        for w in tokens:
            if len(w) > 2:
                trigs.extend([w[i : i + 3]for i in range(len(w)-2)])
            suffix = ''
            preffix = ''
            for i in range(len(w), 0, -1):
                if w[-i:] in self.Suff:
                    suffix = w[-i:]
                    break
            suffix_list.append(suffix)
            for i in range(len(w), 0, -1):
                if w[:i] in self.Preff:
                    preffix = w[:i]
                    break
            preffix_list.append(preffix)
            if w in self.SimpleWord:
                InWikiSimple += 1
            freq += zipf_frequency(w,self.lan) / len_tokens
        char_ngram = self.char_ngram.transform(trigs) / (len_chars - 2)
        suff_val = self.suff.transform(suffix) / len_tokens
        preff_val = self.preff.transform(preffix) / len_tokens
        InWikiSimple /= len_tokens
        
        
        vowelCount = len(re.findall(self.vowelsTwo, word))/len_chars
        NumSenses = 0
        PossiblePOS=0
        SynonMean=0
        
        for w in tokens:
            if self.language=='english':
                syns = wn.synsets(w)
            else:
                syns = wn.synsets(w, lang = 'spa')
            NumSenses += len(syns)/len_tokens
            listPOS=[]
            listsynonym=[]
            for syn in syns:
                listPOS.append(syn.pos())
                listsynonym.extend(syn.lemmas())
            PossiblePOS += len(set(listPOS))/len_tokens
            SynonMean += len(set(listsynonym))/len_tokens
            
        
        ##FEATURES
        features = [len_chars, len_tokens, position_norm, char_ngram, suff_val,
                    preff_val, vowelCount, NumSenses, PossiblePOS, SynonMean,
                    InWikiSimple, freq]
        if self.language == 'spanish':
            accents = len(re.findall(self.accents, word)) / len_tokens
            features.append(accents)
        
        return features

    def train(self, trainset):
        for sent in trainset:
            tokens = sent['target_word'].lower().split()
            # Preprocess chars
            trigs = []
            suffix_list = []
            preffix_list = []
            for w in tokens:
                suffix = ''
                preffix = ''
                #Trains suffix
                for i in range(len(w), 0, -1):
                    if w[-i:] in self.Suff:
                        suffix = w[-i:]
                        break
                suffix_list.append(suffix)
                #Trains preffix
                for i in range(len(w), 0, -1):
                    if w[:i] in self.Preff:
                        preffix = w[:i]
                        break
                preffix_list.append(preffix)
                #Trains trigram of chars
                if len(w) > 2:
                    trigs.extend([w[i : i + 3]for i in range(len(w)-2)])
            self.suff.train(suffix, sent['gold_label'])
            self.preff.train(preffix, sent['gold_label'])
            self.char_ngram.train(trigs, sent['gold_label'])

        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'], sent['sentence'], sent['start_offset']))
            y.append(sent['gold_label'])
        self.modelLog.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],sent['sentence'], sent['start_offset']))

        return self.modelLog.predict(X)
 