# Gerasimos Lampouras, 2017:
import nltk
import itertools
import os.path
import _pickle as pickle
from structuredPredictionNLG.Action import Action
from collections import Counter
from structuredPredictionNLG.DatasetInstance import cleanAndGetAttr, cleanAndGetValue

class SimpleContentPredictor(object):

    def __init__(self, datasetID, attributes, trainingInstances, N=5):
        self.dataset = datasetID
        self.trainingLen = {}
        self.N_size = N
        self.trainingOrders = []
        for predicate in attributes:
            self.trainingLen[predicate] = len(trainingInstances[predicate])

            permuts = itertools.permutations(attributes[predicate], 3)
            for permut in permuts:
                order = []
                for content in permut:
                    order.append(cleanAndGetAttr(content))
                self.trainingOrders.append(">".join(order))
        self.gram_counts = {}
        maxLen = 0
        if True or not self.loadContentPredictor():
            grams = {}
            for i in range(1, self.N_size + 1):
                grams[i] = []
                self.gram_counts[i] = {}
            for predicate in attributes:
                self.gram_counts[1][predicate] = Counter(attributes[predicate])
                for di in trainingInstances[predicate]:
                    # seq = [o for o in di.directAttrValueSequence]
                    seq = [o for o in di.directAttrSequence]
                    if len(seq) > maxLen:
                        maxLen = len(seq)
                    for i in range(2, self.N_size + 1):
                        grams[i].extend(nltk.ngrams(seq, i, pad_left=True, pad_right=True))

                for i in range(2, self.N_size + 1):
                    self.gram_counts[i][predicate] = Counter(grams[i])
            print('Max length of training content sequence:', maxLen)
            # self.writeContentPredictor()

    def getLMProbability(self, predicate, sentence_x, smoothing=0.0):
        unique_words = len(self.gram_counts[1][predicate].keys()) + 2 # For the None paddings
        prob_x = 1.0
        for i in range(2, self.N_size + 1):
            grams = nltk.ngrams(sentence_x, i, pad_left=True, pad_right=True)
            for gram in grams:
                if i == 2 and gram[-1] == None:
                    prob_gram = (self.gram_counts[i][predicate][gram] + smoothing) / (self.trainingLen[predicate] + smoothing * unique_words)
                else:
                    prob_gram = (self.gram_counts[i][predicate][gram] + smoothing) / (self.gram_counts[i - 1][predicate][gram[:-1]] + smoothing * unique_words)
                prob_x = prob_x * prob_gram
        return prob_x

    def rollContentSequence_withLearnedPolicy(self, datasetInstance, contentSequence=False):
        if not contentSequence:
            contentSequence = []

        attrs = set(datasetInstance.input.attributeValues.keys())
        for action in contentSequence:
            attrs.remove(action.attribute)
        # permutations = itertools.permutations(['{}={}'.format(attr, datasetInstance.input.attributeValues[attr]) for attr in attrs])
        permutations = itertools.permutations(attrs)

        bestPermut = False
        max = -1
        for permut in permutations:
            prob = self.getLMProbability(datasetInstance.input.predicate, list(permut), 1.0)
            if prob > max:
                max = prob
                bestPermut = permut

        seq = contentSequence[:]
        # seq.extend([Action(Action.TOKEN_SHIFT, attrValue, 'content') for attrValue in bestPermut])
        seq.extend([Action(Action.TOKEN_SHIFT, '{}={}'.format(attr, datasetInstance.input.attributeValues[attr]), 'content') for attr in bestPermut])
        if seq[-1].attribute != Action.TOKEN_SHIFT:
            seq.append(Action(Action.TOKEN_SHIFT, Action.TOKEN_SHIFT, 'content'))
        return seq

    '''
    # TODO: Implement/fix expert policy for content sequence (if we actually need it)
    def rollContentSequence_withExpertPolicy(self, datasetInstance, rollInSequence):
        minCost = 1.0
        for refSeq in datasetInstance.getEvaluationReferenceAttrValueSequences():
            currentAttr = rollInSequence.sequence[- 1].attribute

            rollOutList = rollInSequence.sequence[:]
            refList = refSeq.sequence[:]

            if len(rollOutList) < len(refList):
                if currentAttr == Action.TOKEN_EOS:
                    while len(rollOutList) == len(refList):
                        rollOutList.append(Action("££", "££"))
                else:
                    rollOutList.extend(refList.subList[len(rollInSequence.sequence()):])
            else:
                while len(rollOutList) != len(refList):
                    refList.append(Action("££", "££"))

            rollOut = ActionSequence(rollOutList).getAttrSequenceToString().lower().strip()
            newRefSeq = ActionSequence(refList)
            refWindows = []
            refWindows.append(newRefSeq.getAttrSequenceToString().lower().strip())

            totalAttrValuesInRef = 0;
            attrValuesInRefAndNotInRollIn = 0;
            for attrValueAct in refList:
                if attrValueAct.attribute != Action.TOKEN_EOS:
                    totalAttrValuesInRef += 1

                    containsAttrValue = False
                    for a in rollOutList:
                        if a.attribute == attrValueAct.attribute:
                            containsAttrValue = True;
                            break;
                    if not containsAttrValue:
                        attrValuesInRefAndNotInRollIn != 1
            coverage = attrValuesInRefAndNotInRollIn / totalAttrValuesInRef
            #System.out.println("ROLLOUT " + rollOut);
            #System.out.println("REFS " + refWindows);
            refCost = LossFunction.getCostMetric(rollOut, refWindows, coverage);
            if refCost < minCost:
                minCost = refCost;
        return minCost
    '''

    def writeContentPredictor(self):
        print("Writing content predictor...")

        with open('../cache/contentPredictor_gram_counts_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.gram_counts, handle)

    def loadContentPredictor(self):
        print("Attempting to load content predictor...")

        self.gram_counts = False

        if os.path.isfile('../cache/contentPredictor_gram_counts_' + self.dataset + '.pickle'):
            with open('../cache/contentPredictor_gram_counts_' + self.dataset + '.pickle', 'rb') as handle:
                self.gram_counts = pickle.load(handle)

        if self.gram_counts:
            return True
        return False