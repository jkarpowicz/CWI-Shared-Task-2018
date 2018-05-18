from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.improved import Improved
from utils.scorer import report_score
import random


def execute_demo(language, amountdata = 100):
    data = Dataset(language, amountdata)
    

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    print('\nInitialising')
    baseline = Baseline(language)
    improved = Improved(language)    

    print('Training')
    baseline.train(data.trainset)
    improved.train(data.trainset)

    print('Predicting')
    predictions = baseline.test(data.devset)
    predictionImp=improved.test(data.devset)
    gold_labels = [sent['gold_label'] for sent in data.devset]
    target = [sent['target_word'] for sent in data.devset]

    print("\nScore for baseline:")
    report_score(gold_labels, predictions)
    print("Score for improved model:")
    report_score(gold_labels, predictionImp)
    
    print('Predicting on testset')
    predictions2 = baseline.test(data.testset)
    predictionImp2 = improved.test(data.testset)
    gold_labels2 = [sent['gold_label'] for sent in data.testset]
    target2 = [sent['target_word'] for sent in data.testset]

    print("\nScore for baseline:")
    report_score(gold_labels2, predictions2)
    print("Score for improved model:")
    report_score(gold_labels2, predictionImp2)
    
    results = [(predictions[i], predictionImp[i], gold_labels[i], target[i]) for i in range(len(target))]
    ####to show wrong predictions
    results = [tup for tup in results if tup[0] != tup[2] and tup[1] != tup[2]]
    
    results2 = [(predictions2[i], predictionImp2[i], gold_labels2[i], target2[i]) for i in range(len(target2))]
    return results, results2



if __name__ == '__main__':
    print('\n----------- ENGLISH 100% -----------\n')
    eng, eng2 = execute_demo('english')
#    print('\n----------- ENGLISH 90% -----------\n')
#    execute_demo('english', 90)
#    print('\n----------- ENGLISH 80% -----------\n')
#    execute_demo('english', 80)
#    print('\n----------- ENGLISH 70% -----------\n')
#    execute_demo('english', 70)
#    print('\n----------- ENGLISH 60% -----------\n')
#    execute_demo('english', 60)
#    print('\n----------- ENGLISH 50% -----------\n')
#    execute_demo('english', 50)
#    print('\n----------- ENGLISH 40% -----------\n')
#    execute_demo('english', 40)
#    print('\n----------- ENGLISH 30% -----------\n')
#    execute_demo('english', 30)
#    print('\n----------- ENGLISH 20% -----------\n')
#    execute_demo('english', 20)
#    print('\n----------- ENGLISH 10% -----------\n')
#    execute_demo('english', 10)
#    
#    
#    
    
    print('\n----------- SPANISH 100% -----------\n')
    spa, spa2 = execute_demo('spanish')
#    print('\n----------- SPANISH 90% -----------\n')
#    execute_demo('spanish', 90)
#    print('\n----------- SPANISH 80% -----------\n')
#    execute_demo('spanish', 80)
#    print('\n----------- SPANISH 70% -----------\n')
#    execute_demo('spanish', 70)
#    print('\n----------- SPANISH 60% -----------\n')
#    execute_demo('spanish', 60)
#    print('\n----------- SPANISH 50% -----------\n')
#    execute_demo('spanish', 50)
#    print('\n----------- SPANISH 40% -----------\n')
#    execute_demo('spanish', 40)
#    print('\n----------- SPANISH 30% -----------\n')
#    execute_demo('spanish', 30)
#    print('\n----------- SPANISH 20% -----------\n')
#    execute_demo('spanish', 20)
#    print('\n----------- SPANISH 10% -----------\n')
#    execute_demo('spanish', 10)
##

##########EXAMPLES #############
#print(eng[:10])
#print(spa[:10])


print(random.sample(eng,20))
print(random.sample(spa,20))

