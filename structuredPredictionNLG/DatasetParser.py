# Gerasimos Lampouras, 2017:
from structuredPredictionNLG.Action import Action
from structuredPredictionNLG.MeaningRepresentation import MeaningRepresentation
from structuredPredictionNLG.DatasetInstance import DatasetInstance, cleanAndGetAttr, lexicalize_word_sequence
from structuredPredictionNLG.FullDelexicalizator import full_delexicalize_E2E
from structuredPredictionNLG.SimpleContentPredictor import SimpleContentPredictor
from collections import Counter
import os.path
import re
import Levenshtein
import _pickle as pickle
import json
import xml.etree.ElementTree as ET
import string
from dateutil import parser as dateparser
from nltk.util import ngrams


'''
 This is a general specification of a DatasetParser.
 A descendant of this class will need to be creater for every specific dataset
 (to deal with dataset-specific formats)
'''
class DatasetParser:

    def __init__(self, trainingFile, developmentFile, testingFile, dataset, opt):
        # self.base_dir = '../'
        self.base_dir = ''

        self.dataset = dataset
        self.dataset_name = opt.name

        self.trainingInstances = {}
        self.developmentInstances = {}
        self.testingInstances = {}

        self.trainingInstances = False
        self.developmentInstances = False
        self.testingInstances = False

        self.train_src_to_di = {}
        self.dev_src_to_di = {}
        self.test_src_to_di = {}

        self.available_values = set()
        self.available_subjects = set()

        self.availableContentActions = {}
        self.availableWordActions = {}
        self.availableWordCounts = {}

        self.trim = opt.trim

        self.ngram_lists_per_word_sequence = {}
        self.ngram_lists_per_relexed_word_sequence = {}
        self.total_relexed_ngram_lists = set()

        self.check_cache_path()

        if (opt.reset or not self.loadTrainingLists(opt.trim, opt.full_delex, opt.infer_MRs)) and trainingFile:
            self.predicates = []
            self.attributes = {}
            self.valueAlignments = {}
            self.vocabulary = set()

            self.maxWordSequenceLength = 0

            self.trainingInstances = self.createLists(self.base_dir + trainingFile, forTrain=True, full_delex=opt.full_delex, infer_MRs=opt.infer_MRs)

            # Post-processing of training data begins
            for predicate in self.trainingInstances:
                for di in self.trainingInstances[predicate]:
                    for attr in di.input.attributeValues:
                        self.available_values.add(di.input.attributeValues[attr])
                    if di.input.attributeSubjects:
                        for attr in di.input.attributeSubjects:
                            self.available_subjects.add(di.input.attributeSubjects[attr])
            # Create the evaluation refs for train data
            for predicate in self.trainingInstances:
                for di in self.trainingInstances[predicate]:
                    refs = set()
                    refs.add(di.directReference)
                    refSeqs = [[o.label.lower() for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    refActionSeqs = [[o for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    for di2 in self.trainingInstances[predicate]:
                        if di != di2 and di2.input.getAbstractMR() == di.input.getAbstractMR():
                            refs.add(" ".join(lexicalize_word_sequence(di2.directReferenceSequence, di.input.delexicalizationMap, complex_relex=True)).strip())
                            if di2.directReferenceSequence not in refSeqs:
                                refSeqs.append(list(o.label.lower() for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT))
                                refActionSeqs.append(list([o for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT]))
                    di.output.evaluationReferences = refs
                    di.output.evaluationReferenceSequences = refSeqs
                    di.output.evaluationReferenceActionSequences = refActionSeqs
                    di.output.calcEvaluationReferenceAttrValueSequences()

                    for refSeq in refSeqs:
                        refSeqTxt = ' '.join(refSeq)
                        if refSeqTxt not in self.ngram_lists_per_word_sequence:
                            self.ngram_lists_per_word_sequence[refSeqTxt] = self.get_ngram_list(refSeq)

            self.initializeActionSpace()
            if opt.trim:
                self.trimTrainingSpace()
                # Initializing the action space again after trimming results in less actions
                self.initializeActionSpace()

            self.vocabulary_per_attr = {}
            for predicate in self.attributes:
                self.vocabulary_per_attr[predicate] = {}
                for attr in self.attributes[predicate]:
                    self.vocabulary_per_attr[predicate][attr] = set()
            for predicate in self.trainingInstances:
                for di in self.trainingInstances[predicate]:
                    for a in di.directReferenceSequence:
                        if a.label != Action.TOKEN_SHIFT:
                            attr = cleanAndGetAttr(a.attribute)
                            self.vocabulary_per_attr[di.input.predicate][attr].add(a.label)

            for predicate in self.trainingInstances:
                if predicate not in self.train_src_to_di:
                    self.train_src_to_di[predicate] = {}
                for di in self.trainingInstances[predicate]:
                    di.input.nn_src = " ".join(["{} {}".format(attr, di.input.attributeValues[attr]) if attr in di.input.attributeValues else "{}@none@ {}_value@none@".format(attr, attr) for attr in [i for i in di.directAttrSequence if i != Action.TOKEN_SHIFT]])
                    self.train_src_to_di[predicate][di.input.nn_src] = di
                    for ref in di.output.evaluationReferences:
                        refSeq = ref.split(" ")
                        if ref not in self.ngram_lists_per_relexed_word_sequence:
                            self.ngram_lists_per_relexed_word_sequence[ref] = self.get_ngram_list(refSeq)

            self.writeTrainingLists(opt.trim, opt.full_delex, opt.infer_MRs)
        else:
            self.initializeActionSpace()

        self.most_common_words = set()
        for predicate in self.trainingInstances:
            for word, count in self.availableWordCounts[predicate].most_common():
                inAll = True
                for attr in self.attributes[predicate]:
                    if word not in self.availableWordActions[predicate][attr]:
                        inAll = False
                        break
                if inAll:
                    self.most_common_words.add(word)
                    if len(self.most_common_words) > 30:
                        break
        # Silly way of filtering at least one occurence
        #total_relexed_ngram_lists_tmp = set()
        for predicate in self.trainingInstances:
            for di in self.trainingInstances[predicate]:
                for ref in di.output.evaluationReferences:
                    refSeq = ref.split(" ")
                    for n_gram in self.get_ngram_list(refSeq, min=3):
                        #if n_gram in total_relexed_ngram_lists_tmp:
                        self.total_relexed_ngram_lists.add(n_gram)
                        #else:
                        #    total_relexed_ngram_lists_tmp.add(n_gram)

        scp = SimpleContentPredictor(self.dataset, self.attributes, self.trainingInstances)
        if (opt.reset or not self.loadDevelopmentLists(opt.full_delex)) and developmentFile:
            devs = self.createLists(self.base_dir + developmentFile, full_delex=opt.full_delex)
            # Create the evaluation refs for development data, as described in https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs
            self.developmentInstances = {}
            for predicate in devs:
                self.developmentInstances[predicate] = []
                for di in devs[predicate]:
                    di.init_alt_outputs()

                    refs = set()
                    refs.add(di.directReference)
                    refSeqs = [[o.label.lower() for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    refActionSeqs = [[o for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    for di2 in devs[predicate]:
                        if di != di2 and di2.input.getAbstractMR() == di.input.getAbstractMR():
                            refs.add(" ".join(lexicalize_word_sequence(di2.directReferenceSequence, di.input.delexicalizationMap, complex_relex=True)).strip())
                            if di2.directReferenceSequence not in refSeqs:
                                refSeqs.append(list(o.label.lower() for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT))
                                refActionSeqs.append(list([o for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT]))
                    di.output.evaluationReferences = set(refs)
                    di.output.evaluationReferenceSequences = refSeqs[:]
                    di.output.evaluationReferenceActionSequences = refActionSeqs[:]
                    di.output.calcEvaluationReferenceAttrValueSequences()

                    self.developmentInstances[predicate].append(di)

            for predicate in self.developmentInstances:
                if predicate not in self.dev_src_to_di:
                    self.dev_src_to_di[predicate] = {}
                for di in self.developmentInstances[predicate]:
                    content_sequence = [cleanAndGetAttr(a.attribute) for a in
                                        scp.rollContentSequence_withLearnedPolicy(di) if
                                        cleanAndGetAttr(a.attribute) != Action.TOKEN_SHIFT]
                    di.input.nn_src = " ".join(["{} {}".format(attr, di.input.attributeValues[
                        attr]) if attr in di.input.attributeValues else "{}@none@ {}_value@none@".format(attr,
                                                                                                           attr) for
                                                    attr in content_sequence])

                    self.dev_src_to_di[predicate][di.input.nn_src] = di

            self.writeDevelopmentLists(opt.full_delex)

            for predicate in self.developmentInstances:
                for di in self.developmentInstances[predicate]:
                    for ref in di.output.evaluationReferences:
                        refSeq = ref.split(" ")
                        if ref not in self.ngram_lists_per_relexed_word_sequence:
                            self.ngram_lists_per_relexed_word_sequence[ref] = self.get_ngram_list(refSeq)

            for predicate in self.developmentInstances:
                for di in self.developmentInstances[predicate]:
                    for attr in di.input.attributeValues:
                        self.available_values.add(di.input.attributeValues[attr])
                    if di.input.attributeSubjects:
                        for attr in di.input.attributeSubjects:
                            self.available_subjects.add(di.input.attributeSubjects[attr])
            self.writeTrainingLists(opt.trim, opt.full_delex, opt.infer_MRs)
        if (opt.reset or not self.loadTestingLists(opt.full_delex)) and testingFile:
            tests = self.createLists(self.base_dir + testingFile, full_delex=opt.full_delex)

            self.testingInstances = {}
            for predicate in tests:
                self.testingInstances[predicate] = []
                for di in tests[predicate]:
                    di.init_alt_outputs()

                    refs = set()
                    refs.add(di.directReference)
                    refSeqs = [
                        [o.label.lower() for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    refActionSeqs = [[o for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    for di2 in tests[predicate]:
                        if di != di2 and di2.input.MRstr == di.input.MRstr:
                            refs.add(di2.directReference)
                            if di2.directReferenceSequence not in refSeqs:
                                refSeqs.append(list(o.label.lower() for o in di2.directReferenceSequence if
                                                    o.label != Action.TOKEN_SHIFT))
                                refActionSeqs.append(
                                    list([o for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT]))
                    di.output.evaluationReferences = set(refs)
                    di.output.evaluationReferenceSequences = refSeqs[:]
                    di.output.evaluationReferenceActionSequences = refActionSeqs[:]
                    di.output.calcEvaluationReferenceAttrValueSequences()

                    self.testingInstances[predicate].append(di)

            for predicate in self.testingInstances:
                if predicate not in self.test_src_to_di:
                    self.test_src_to_di[predicate] = {}
                for di in self.testingInstances[predicate]:
                    content_sequence = [cleanAndGetAttr(a.attribute) for a in
                                        scp.rollContentSequence_withLearnedPolicy(di) if
                                        cleanAndGetAttr(a.attribute) != Action.TOKEN_SHIFT]
                    di.input.nn_src = " ".join(["{} {}".format(attr, di.input.attributeValues[
                        attr]) if attr in di.input.attributeValues else "{}@none@ {}_value@none@".format(attr,
                                                                                                         attr)
                                                for
                                                attr in content_sequence])

                    self.test_src_to_di[predicate][di.input.nn_src] = di
            self.writeTestingLists(opt.full_delex)

        self.write_onmt_data(opt)

    def get_ngram_list(self, word_sequence, min=1):
        ngram_list = []
        seq = word_sequence[:]
        if min <= 4:
            ngram_list.extend(ngrams(seq, 4, pad_left=True, pad_right=True))
        if min <= 3:
            ngram_list.extend(ngrams(seq, 3, pad_left=True, pad_right=True))
        if min <= 2:
            ngram_list.extend(ngrams(seq, 2, pad_left=True, pad_right=True))
        if min <= 1:
            ngram_list.extend(seq)
        return ngram_list

    def createLists(self, dataFile, forTrain=False, full_delex=False, infer_MRs=False):
        if self.dataset.lower() == 'e2e':
            return self.createLists_E2E(dataFile, forTrain, full_delex, infer_MRs)
        elif self.dataset.lower() == 'webnlg':
            return self.createLists_webnlg(dataFile, forTrain)
        elif self.dataset.lower() == 'sfhotel':
            return self.createLists_SFX(dataFile, forTrain)

    def createLists_E2E(self, dataFile, forTrain=False, full_delex=False, infer_MRs=False):
        print("Create lists from ", dataFile, "...")
        singlePredicate = 'inform'

        instances = dict()
        instances[singlePredicate] = []
        dataPart = []

        # We read the data from the data files.
        with open(dataFile, encoding="utf8") as f:
            lines = f.readlines()
            for s in lines:
                s = str(s)
                if s.startswith("\""):
                    dataPart.append(s)

        # This dataset has no predicates, so we assume a default predicate
        self.predicates.append(singlePredicate)
        num = 0
        err = 0
        # Each line corresponds to a MR
        for line in dataPart:
            #if num == 0:
            #    num += 1
            #    continue
            num += 1

            if "\"," in line:
                MRPart = line.split("\",")[0].strip()
                refPart = line.split("\",")[1].lower().strip()
            else:
                MRPart = line.strip()
                refPart = ""
            if refPart.startswith("\"") and refPart.endswith("\""):
                refPart = refPart[1:-1]
            if MRPart.startswith("\""):
                MRPart = MRPart[1:]
            if refPart.startswith("\""):
                refPart = refPart[1:]
            if refPart.endswith("\""):
                refPart = refPart[:-1]
            refPart = re.sub("([.,?:;!'-])", " \g<1> ", refPart)
            refPart = refPart.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace("  ", " ").strip()
            refPart = " ".join(refPart.split())
            MRAttrValues = MRPart.split(",")

            if full_delex:
                while ' moderately ' in " " + refPart + " ":
                    refPart = (" " + refPart + " ").replace(" moderately ", " moderate -ly ").strip()
                while ' averagely ' in " " + refPart + " ":
                    refPart = (" " + refPart + " ").replace(" averagely ", " average -ly ").strip()
                while ' highly ' in " " + refPart + " ":
                    refPart = (" " + refPart + " ").replace(" highly ", " high -ly ").strip()
                while ' 5the ' in " " + refPart + " ":
                    refPart = (" " + refPart + " ").replace(" 5the ", " 5 the ").strip()

            # Map from original values to delexicalized values
            delexicalizedMap = {}
            # Map attributes to their values
            attributeValues = {}

            for attrValue in MRAttrValues:
                value = attrValue[attrValue.find("[") + 1:attrValue.find("]")].strip().lower()
                attribute = attrValue[0:attrValue.find("[")].strip().lower().replace(" ", "_").lower()

                if attribute == 'familyfriendly' and full_delex:
                    if value == 'yes':
                        value = 'family friendly'
                    else:
                        value = 'not family friendly'
                elif value == "yes" or value == "no":
                    value = attribute + "_" + value

                if forTrain and singlePredicate not in self.attributes:
                    self.attributes[singlePredicate] = set()
                if attribute:
                    if forTrain:
                        self.attributes[singlePredicate].add(attribute)
                    attributeValues[attribute] = value
            for attribute in ['area', 'food', 'name', 'near', 'eattype', 'pricerange', 'customer_rating', 'pricerange', 'customer_rating', 'familyfriendly']:
                delexValue = Action.TOKEN_X + attribute + "_0"
                if attribute in attributeValues and delexValue not in delexicalizedMap:
                    value = attributeValues[attribute]
                    if attribute == 'name' or attribute == 'near' or full_delex:
                        v = re.sub("([.,?:;!'-])", " \g<1> ", value)
                        v = v.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace("  ", " ").strip().lower()
                        if " " + v + " " in " " + refPart + " ":
                            delexicalizedMap[delexValue] = v
                            value = delexValue
                        elif full_delex:
                            temp_value = full_delexicalize_E2E(delexicalizedMap, attribute, v, attributeValues, refPart)
                            if temp_value:
                                delexicalizedMap[delexValue] = temp_value
                                value = delexValue

                        if delexValue in delexicalizedMap:
                            if (" " + delexicalizedMap[delexValue].lower() + " ") in (" " + refPart + " "):
                                refPart = (" " + refPart + " ").replace((" " + delexicalizedMap[delexValue].lower() + " "),
                                                                        (" " + delexValue + " ")).strip()

                            attributeValues[attribute] = value

            '''
            for attribute in ['area', 'food', 'name', 'near', 'eattype', 'pricerange', 'customer_rating', 'familyfriendly']:
                delexValue = Action.TOKEN_X + attribute + "_0"
                if attribute in attributeValues and delexValue not in delexicalizedMap:
                    if attribute == 'familyfriendly':
                        err += 1
                        print(attribute)
                        print(attributeValues[attribute])
                        print(refPart)
                        print('-----------------------------------')
            '''

            observedWordSequence = []
            refPart = refPart.replace(", ,", " , ").replace(". .", " . ").replace('"', ' " ').replace("  ", " ").strip()
            refPart = " ".join(refPart.split())
            if refPart:
                words = refPart.split(" ")
                for word in words:
                    word = word.strip()
                    if word:
                        if "0f" in word:
                            word = word.replace("0f", "of")

                        m = re.search("^@x@([a-z]+)_([0-9]+)", word)
                        if m and m.group(0) != word:
                            var = m.group(0)
                            realValue = delexicalizedMap.get(var)
                            realValue = word.replace(var, realValue)
                            delexicalizedMap[var] = realValue
                            observedWordSequence.append(var.strip())
                        else:
                            m = re.match("([0-9]+)([a-z]+)", word)
                            if m and m.group(1).strip() == "o":
                                observedWordSequence.add(m.group(1).strip() + "0")
                            elif m:
                                observedWordSequence.append(m.group(1).strip())
                                observedWordSequence.append(m.group(2).strip())
                            else:
                                m = re.match("([a-z]+)([0-9]+)", word)
                                if m and (m.group(1).strip() == "l" or m.group(1).strip() == "e"):
                                    observedWordSequence.append("£" + m.group(2).strip())
                                elif m:
                                    observedWordSequence.append(m.group(1).strip())
                                    observedWordSequence.append(m.group(2).strip())
                                else:
                                    m = re.match("(£)([a-z]+)", word)
                                    if m:
                                        observedWordSequence.append(m.group(1).strip())
                                        observedWordSequence.append(m.group(2).strip())
                                    else:
                                        m = re.match("([a-z]+)(£[0-9]+)", word)
                                        if m:
                                            observedWordSequence.append(m.group(1).strip())
                                            observedWordSequence.append(m.group(2).strip())
                                        else:
                                            m = re.match("([0-9]+)([a-z]+)([0-9]+)", word)
                                            if m:
                                                observedWordSequence.append(m.group(1).strip())
                                                observedWordSequence.append(m.group(2).strip())
                                                observedWordSequence.append(m.group(3).strip())
                                            else:
                                                m = re.match("([0-9]+)(@x@[a-z]+_0)", word)
                                                if m:
                                                    observedWordSequence.append(m.group(1).strip())
                                                    observedWordSequence.append(m.group(2).strip())
                                                else:
                                                    m = re.match("(£[0-9]+)([a-z]+)", word)
                                                    if m and m.group(2).strip() == "o":
                                                        observedWordSequence.append(m.group(1).strip() + "0")
                                                    else:
                                                        observedWordSequence.append(word.strip())

            MR = MeaningRepresentation(singlePredicate, attributeValues, MRPart, delexicalizedMap)

            # We store the maximum observed word sequence length, to use as a limit during generation
            if forTrain and len(observedWordSequence) > self.maxWordSequenceLength:
                self.maxWordSequenceLength = len(observedWordSequence)

            # We initialize the alignments between words and attribute/value pairs
            wordToAttrValueAlignment = []
            for word in observedWordSequence:
                if re.match("[.,?:;!'\"]", word.strip()):
                    wordToAttrValueAlignment.append(Action.TOKEN_PUNCT)
                else:
                    wordToAttrValueAlignment.append("[]")
            directReferenceSequence = []
            for r, word in enumerate(observedWordSequence):
                directReferenceSequence.append(Action(word, wordToAttrValueAlignment[r], "word"))
                if forTrain:
                    self.vocabulary.add(word)

            alingedAttributes = []
            if directReferenceSequence:
                # Align subphrases of the sentence to attribute values
                observedValueAlignments = {}
                valueToAttr = {}
                for attr in MR.attributeValues.keys():
                    value = MR.attributeValues[attr]
                    if not value.startswith(Action.TOKEN_X) and not full_delex:
                        observedValueAlignments[value] = set()
                        valueToAttr[value] = attr
                        valuesToCompare = set()
                        valuesToCompare.update([value, attr])
                        valuesToCompare.update(value.split(" "))
                        valuesToCompare.update(attr.split(" "))
                        valuesToCompare.update(attr.split("_"))
                        for valueToCompare in valuesToCompare:
                            # obtain n-grams from the sentence
                            for n in range(1, 6):
                                grams = ngrams(directReferenceSequence, n)

                                # calculate the similarities between each gram and valueToCompare
                                for gram in grams:
                                    if Action.TOKEN_X not in [o.label for o in gram].__str__() and Action.TOKEN_PUNCT not in [o.attribute for o in gram]:
                                        compare = " ".join(o.label for o in gram)
                                        backwardCompare = " ".join(o.label for o in reversed(gram))

                                        if compare.strip():
                                            # Calculate the character-level distance between the value and the nGram (in its original and reversed order)
                                            distance = Levenshtein.ratio(valueToCompare.lower(), compare.lower())
                                            backwardDistance = Levenshtein.ratio(valueToCompare.lower(), backwardCompare.lower())

                                            # We keep the best distance score; note that the Levenshtein distance is normalized so that greater is better
                                            if backwardDistance > distance:
                                                distance = backwardDistance
                                            if (distance > 0.3):
                                                observedValueAlignments[value].add((gram, distance))
                while observedValueAlignments.keys():
                    # Find the best aligned nGram
                    max = -1000
                    bestGrams = {}

                    toRemove = set()
                    for value in observedValueAlignments.keys():
                        if observedValueAlignments[value]:
                            for gram, distance in observedValueAlignments[value]:
                                if distance > max:
                                    max = distance
                                    bestGrams = {}
                                if distance == max:
                                    bestGrams[gram] = value
                        else:
                            toRemove.add(value)
                    for value in toRemove:
                        del observedValueAlignments[value]
                    # Going with the latest occurance of a matched ngram works best when aligning with hard alignments
                    # Because all the other match ngrams that occur to the left of the latest, will probably be aligned as well
                    '''
                    maxOccurance = -1
                    bestGram = False
                    bestValue = False
                    for gram in bestGrams:
                        occur = self.find_subList_in_actionList(gram, directReferenceSequence)[0] + len(gram)
                        if occur > maxOccurance:
                            maxOccurance = occur
                            bestGram = gram
                            bestValue = bestGrams[gram]
                    '''
                    # Otherwise might be better to go for the longest ngram

                    maxLen = 0
                    bestGram = False
                    bestValue = False
                    for gram in sorted(bestGrams):
                        if len(gram) > maxLen:
                            maxLen = len(gram)
                            bestGram = gram
                            bestValue = bestGrams[gram]

                    if bestGram:
                        # Find the subphrase that corresponds to the best aligned nGram
                        bestGramPos = self.find_subList_in_actionList(bestGram, directReferenceSequence)
                        if bestGramPos:
                            # Only apply the gram if the position is not already aligned
                            unalignedRange = True
                            for i in range(bestGramPos[0], bestGramPos[1] + 1):
                                if directReferenceSequence[i].attribute != '[]':
                                    unalignedRange = False
                            if unalignedRange:
                                for i in range(bestGramPos[0], bestGramPos[1] + 1):
                                    directReferenceSequence[i].attribute = valueToAttr[bestValue]
                                    alingedAttributes.append(directReferenceSequence[i].attribute)
                                if forTrain:
                                    # Store the best aligned nGram
                                    if bestValue not in self.valueAlignments.keys():
                                        self.valueAlignments[bestValue] = {}
                                    self.valueAlignments[bestValue][bestGram] = max
                                # And remove it from the observed ones for this instance
                                del observedValueAlignments[bestValue]
                            else:
                                observedValueAlignments[bestValue].remove((bestGram, max))
                        else:
                            observedValueAlignments[bestValue].remove((bestGram, max))
                for action in directReferenceSequence:
                    if action.label.startswith(Action.TOKEN_X):
                        attr = action.label[3:action.label.rfind('_')]
                        if attr not in alingedAttributes:
                            action.attribute = attr
                            alingedAttributes.append(action.attribute)
            if full_delex:
                alingedAttributes = []
                for action in directReferenceSequence:
                    if action.label.startswith(Action.TOKEN_X):
                        attr = action.label[3:action.label.rfind('_')]
                        action.attribute = attr
                        alingedAttributes.append(action.attribute)

            # If not all attributes are aligned, ignore the instance from training?
            # Alternatively, we could align them randomly; certainly not ideal, but usually it concerns edge cases
            if not forTrain and full_delex:
                for attr in MR.attributeValues:
                    MR.attributeValues[attr] = Action.TOKEN_X + attr + "_0"
            if full_delex and infer_MRs and forTrain and MR.attributeValues.keys() != set(alingedAttributes) and len(alingedAttributes) > 0:
                for attr in [o for o in MR.attributeValues.keys()]:
                    if attr not in alingedAttributes:
                        del MR.attributeValues[attr]
                MR.getAbstractMR(True)
            if (MR.attributeValues.keys() == set(alingedAttributes) and (not full_delex or len(MR.attributeValues.keys()) == len(alingedAttributes))) or not forTrain:
                if forTrain:
                    directReferenceSequence = inferNaiveAlignments(MR, directReferenceSequence)
                DI = DatasetInstance(MR, directReferenceSequence, self.postProcessRef(MR, directReferenceSequence))
                instances[singlePredicate].append(DI)
        return instances

    def createLists_SFX(self, dataFile, forTrain=False):
        print("Create lists from ", dataFile, "...")

        instances = dict()
        dataPart = []

        # We read the data from the data files.
        with open(dataFile, encoding="utf8") as f:
            lines = f.readlines()
            for s in lines:
                s = str(s)
                if s.startswith("\""):
                    dataPart.append(s)

        num = 0
        # Each line corresponds to a MR
        with open(dataFile) as f:
            dataPart = json.load(f)
            for line in dataPart:
                num += 1

                MRPart = line[0].lower().strip()
                refPart = line[1].lower().strip()

                if refPart.startswith("\"") and refPart.endswith("\""):
                    refPart = refPart[1:-1]
                if MRPart.startswith("\""):
                    MRPart = MRPart[1:]
                if refPart.startswith("\""):
                    refPart = refPart[1:]
                if refPart.endswith("\""):
                    refPart = refPart[:-1]
                refPart = re.sub("([.,?:;!'-])", " \g<1> ", refPart)
                refPart = refPart.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace("  ", " ").strip()
                refPart = refPart.replace(" hotels ", " hotel -s ")
                refPart = " ".join(refPart.split())

                predicate = MRPart[:MRPart.find('(')]
                if predicate not in self.predicates:
                    self.predicates.append(predicate)
                if predicate not in instances:
                    instances[predicate] = []
                MRAttrValues = MRPart[MRPart.find('(') + 1:MRPart.rfind(')')].split(";")

                # Map from original values to delexicalized values
                delexicalizedMap = {}
                # Map attributes to their values
                attributeValues = {}

                for attrValue in MRAttrValues:
                    if not attrValue:
                        attribute = 'none'
                        value = attribute + '_none'
                    elif '=' in attrValue:
                        attrValue = attrValue.split('=')
                        attribute = attrValue[0].strip().lower()
                        value = attrValue[1].strip().lower()
                        if value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                    else:
                        attribute = attrValue
                        value = attribute + '_none'

                    if forTrain and predicate not in self.attributes:
                        self.attributes[predicate] = set()
                    if attribute:
                        if forTrain:
                            self.attributes[predicate].add(attribute)
                        if attribute in attributeValues:
                            if (value == 'yes' and attributeValues[attribute] == attribute + '_no') or (value == 'no' and attributeValues[attribute] == attribute + '_yes') or (value == 'yes' and attributeValues[attribute] == attribute + '_dont_care') or (value == 'no' and attributeValues[attribute] == attribute + '_dont_care') or (value == 'dont_care' and attributeValues[attribute] == attribute + '_no') or (value == 'dont_care' and attributeValues[attribute] == attribute + '_yes'):
                                value = attribute + "_dont_care"
                            elif attributeValues[attribute] == attribute + '_' + value:
                                continue
                        if value == "yes" or value == "no" or value == "dont_care":
                            value = attribute + "_" + value
                        attributeValues[attribute] = value
                for attribute in attributeValues:
                    delexValue = Action.TOKEN_X + attribute + "_0"
                    if attribute in attributeValues and delexValue not in delexicalizedMap:
                        value = attributeValues[attribute]

                        v = re.sub("([.,?:;!'-])", " \g<1> ", value)
                        v = v.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace("  ", " ").strip().lower()
                        if " " + v + " " in " " + refPart + " ":
                            delexicalizedMap[delexValue] = v
                            value = delexValue
                        elif ' st ' in ' ' + v + ' ' and (' ' + v + ' ').replace(' st ', ' street ') in " " + refPart + " ":
                            delexicalizedMap[delexValue] = (' ' + v + ' ').replace(' st ', ' street ').strip()
                            value = delexValue
                        elif ' street ' in ' ' + v + ' ' and (' ' + v + ' ').replace(' street ', ' st ') in " " + refPart + " ":
                            delexicalizedMap[delexValue] = (' ' + v + ' ').replace(' street ', ' st ').strip()
                            value = delexValue
                        elif ' ave ' in ' ' + v + ' ' and (' ' + v + ' ').replace(' ave ', ' avenue ') in " " + refPart + " ":
                            delexicalizedMap[delexValue] = (' ' + v + ' ').replace(' ave ', ' avenue ').strip()
                            value = delexValue
                        elif ' avenue ' in ' ' + v + ' ' and (' ' + v + ' ').replace(' avenue ', ' ave ') in " " + refPart + " ":
                            delexicalizedMap[delexValue] = (' ' + v + ' ').replace(' avenue ', ' ave ').strip()
                            value = delexValue
                        elif ' or ' in ' ' + v + ' ' and (' ' + v.split(' or ')[1] + ' or ' + v.split(' or ')[0] + ' ') in " " + refPart + " ":
                            delexicalizedMap[delexValue] = (' ' + v.split(' or ')[1] + ' or ' + v.split(' or ')[0] + ' ').strip()
                            value = delexValue

                        if delexValue in delexicalizedMap:
                            if (" " + delexicalizedMap[delexValue].lower() + " ") in (" " + refPart + " "):
                                refPart = (" " + refPart + " ").replace((" " + delexicalizedMap[delexValue].lower() + " "),
                                                                        (" " + delexValue + " ")).strip()

                            attributeValues[attribute] = value
                '''
                for attribute in ['area', 'food', 'name', 'near', 'eattype', 'pricerange', 'customer_rating', 'familyfriendly']:
                    delexValue = Action.TOKEN_X + attribute + "_0"
                    if attribute in attributeValues and delexValue not in delexicalizedMap:
                        if attribute == 'familyfriendly':
                            err += 1
                            print(attribute)
                            print(attributeValues[attribute])
                            print(refPart)
                            print('-----------------------------------')
                '''

                observedWordSequence = []
                refPart = refPart.replace(", ,", " , ").replace(". .", " . ").replace('"', ' " ').replace("  ", " ").strip()
                refPart = " ".join(refPart.split())
                if refPart:
                    words = refPart.split(" ")
                    for word in words:
                        word = word.strip()
                        if word:
                            if "0f" in word:
                                word = word.replace("0f", "of")

                            m = re.search("^@x@([a-z]+)_([0-9]+)", word)
                            if m and m.group(0) != word:
                                var = m.group(0)
                                realValue = delexicalizedMap.get(var)
                                realValue = word.replace(var, realValue)
                                delexicalizedMap[var] = realValue
                                observedWordSequence.append(var.strip())
                            else:
                                m = re.match("([0-9]+)([a-z]+)", word)
                                if m and m.group(1).strip() == "o":
                                    observedWordSequence.add(m.group(1).strip() + "0")
                                elif m:
                                    observedWordSequence.append(m.group(1).strip())
                                    observedWordSequence.append(m.group(2).strip())
                                else:
                                    m = re.match("([a-z]+)([0-9]+)", word)
                                    if m and (m.group(1).strip() == "l" or m.group(1).strip() == "e"):
                                        observedWordSequence.append("£" + m.group(2).strip())
                                    elif m:
                                        observedWordSequence.append(m.group(1).strip())
                                        observedWordSequence.append(m.group(2).strip())
                                    else:
                                        m = re.match("(£)([a-z]+)", word)
                                        if m:
                                            observedWordSequence.append(m.group(1).strip())
                                            observedWordSequence.append(m.group(2).strip())
                                        else:
                                            m = re.match("([a-z]+)(£[0-9]+)", word)
                                            if m:
                                                observedWordSequence.append(m.group(1).strip())
                                                observedWordSequence.append(m.group(2).strip())
                                            else:
                                                m = re.match("([0-9]+)([a-z]+)([0-9]+)", word)
                                                if m:
                                                    observedWordSequence.append(m.group(1).strip())
                                                    observedWordSequence.append(m.group(2).strip())
                                                    observedWordSequence.append(m.group(3).strip())
                                                else:
                                                    m = re.match("([0-9]+)(@x@[a-z]+_0)", word)
                                                    if m:
                                                        observedWordSequence.append(m.group(1).strip())
                                                        observedWordSequence.append(m.group(2).strip())
                                                    else:
                                                        m = re.match("(£[0-9]+)([a-z]+)", word)
                                                        if m and m.group(2).strip() == "o":
                                                            observedWordSequence.append(m.group(1).strip() + "0")
                                                        else:
                                                            observedWordSequence.append(word.strip())

                MR = MeaningRepresentation(predicate, attributeValues, MRPart, delexicalizedMap)

                # We store the maximum observed word sequence length, to use as a limit during generation
                if forTrain and len(observedWordSequence) > self.maxWordSequenceLength:
                    self.maxWordSequenceLength = len(observedWordSequence)

                # We initialize the alignments between words and attribute/value pairs
                wordToAttrValueAlignment = []
                for word in observedWordSequence:
                    if re.match("[.,?:;!'\"]", word.strip()):
                        wordToAttrValueAlignment.append(Action.TOKEN_PUNCT)
                    else:
                        wordToAttrValueAlignment.append("[]")
                directReferenceSequence = []
                for r, word in enumerate(observedWordSequence):
                    directReferenceSequence.append(Action(word, wordToAttrValueAlignment[r], "word"))
                    if forTrain:
                        self.vocabulary.add(word)

                alingedAttributes = []
                if directReferenceSequence and forTrain:
                    # Align subphrases of the sentence to attribute values
                    observedValueAlignments = {}
                    valueToAttr = {}
                    for attr in MR.attributeValues.keys():
                        value = MR.attributeValues[attr]
                        if not value.startswith(Action.TOKEN_X):
                            observedValueAlignments[value] = set()
                            valueToAttr[value] = attr
                            valuesToCompare = set()
                            valuesToCompare.update([value, attr])
                            valuesToCompare.update(value.split(" "))
                            valuesToCompare.update(attr.split(" "))
                            valuesToCompare.update(attr.split("_"))
                            for valueToCompare in valuesToCompare:
                                # obtain n-grams from the sentence
                                for n in range(1, 6):
                                    grams = ngrams(directReferenceSequence, n)

                                    # calculate the similarities between each gram and valueToCompare
                                    for gram in grams:
                                        if Action.TOKEN_X not in [o.label for o in gram].__str__() and Action.TOKEN_PUNCT not in [o.attribute for o in gram]:
                                            compare = " ".join(o.label for o in gram)
                                            backwardCompare = " ".join(o.label for o in reversed(gram))

                                            if compare.strip():
                                                # Calculate the character-level distance between the value and the nGram (in its original and reversed order)
                                                distance = Levenshtein.ratio(valueToCompare.lower(), compare.lower())
                                                backwardDistance = Levenshtein.ratio(valueToCompare.lower(), backwardCompare.lower())

                                                # We keep the best distance score; note that the Levenshtein distance is normalized so that greater is better
                                                if backwardDistance > distance:
                                                    distance = backwardDistance
                                                if (distance > 0.3):
                                                    observedValueAlignments[value].add((gram, distance))
                    while observedValueAlignments.keys():
                        # Find the best aligned nGram
                        max = -1000
                        bestGrams = {}

                        toRemove = set()
                        for value in observedValueAlignments.keys():
                            if observedValueAlignments[value]:
                                for gram, distance in observedValueAlignments[value]:
                                    if distance > max:
                                        max = distance
                                        bestGrams = {}
                                    if distance == max:
                                        bestGrams[gram] = value
                            else:
                                toRemove.add(value)
                        for value in toRemove:
                            del observedValueAlignments[value]
                        # Going with the latest occurance of a matched ngram works best when aligning with hard alignments
                        # Because all the other match ngrams that occur to the left of the latest, will probably be aligned as well
                        '''
                        maxOccurance = -1
                        bestGram = False
                        bestValue = False
                        for gram in bestGrams:
                            occur = self.find_subList_in_actionList(gram, directReferenceSequence)[0] + len(gram)
                            if occur > maxOccurance:
                                maxOccurance = occur
                                bestGram = gram
                                bestValue = bestGrams[gram]
                        '''
                        # Otherwise might be better to go for the longest ngram

                        maxLen = 0
                        bestGram = False
                        bestValue = False
                        for gram in sorted(bestGrams):
                            if len(gram) > maxLen:
                                maxLen = len(gram)
                                bestGram = gram
                                bestValue = bestGrams[gram]

                        if bestGram:
                            # Find the subphrase that corresponds to the best aligned nGram
                            bestGramPos = self.find_subList_in_actionList(bestGram, directReferenceSequence)
                            if bestGramPos:
                                # Only apply the gram if the position is not already aligned
                                unalignedRange = True
                                for i in range(bestGramPos[0], bestGramPos[1] + 1):
                                    if directReferenceSequence[i].attribute != '[]':
                                        unalignedRange = False
                                if unalignedRange:
                                    for i in range(bestGramPos[0], bestGramPos[1] + 1):
                                        if valueToAttr[bestValue] != 'none':
                                            directReferenceSequence[i].attribute = valueToAttr[bestValue]
                                            alingedAttributes.append(directReferenceSequence[i].attribute)
                                    if forTrain:
                                        # Store the best aligned nGram
                                        if bestValue not in self.valueAlignments.keys():
                                            self.valueAlignments[bestValue] = {}
                                        self.valueAlignments[bestValue][bestGram] = max
                                    # And remove it from the observed ones for this instance
                                    del observedValueAlignments[bestValue]
                                else:
                                    observedValueAlignments[bestValue].remove((bestGram, max))
                            else:
                                observedValueAlignments[bestValue].remove((bestGram, max))
                    for action in directReferenceSequence:
                        if action.label.startswith(Action.TOKEN_X):
                            attr = action.label[3:action.label.rfind('_')]
                            action.attribute = attr
                            alingedAttributes.append(action.attribute)

                passport = True
                mr_attr_count = 0
                for attr in MR.attributeValues:
                    if attr != 'none':
                        mr_attr_count += 1
                if not forTrain or mr_attr_count == len(alingedAttributes):
                    for attr in MR.attributeValues:
                        if attr != 'none' and '_' not in MR.attributeValues[attr] and 'none' not in MR.attributeValues[attr]:
                            passport = False
                            if not forTrain:
                                delexSubject = Action.TOKEN_X + attr + "_0"
                                MR.delexicalizationMap[delexSubject] = MR.attributeValues[attr]
                                MR.attributeValues[attr] = delexSubject
                                MR.getAbstractMR(True)
                else:
                    passport = False
                # If not all attributes are aligned, ignore the instance from training?
                # Alternatively, we could align them randomly; certainly not ideal, but usually it concerns edge cases
                if forTrain and MR.attributeValues.keys() != set(alingedAttributes) and 'none' in MR.attributeValues:
                    directReferenceSequence[0].attribute = 'none'
                if passport or not forTrain:
                    if forTrain:
                        directReferenceSequence = inferNaiveAlignments(MR, directReferenceSequence)
                    DI = DatasetInstance(MR, directReferenceSequence, self.postProcessRef(MR, directReferenceSequence, False))
                    instances[predicate].append(DI)
        return instances

    def createLists_webnlg(self, dataFile, forTrain=False):
        print("Create lists from ", dataFile, "...")
        singlePredicate = 'inform'

        instances = dict()
        instances[singlePredicate] = []
        dataPart = []

        # This dataset has no predicates, so we assume a default predicate
        self.predicates.append(singlePredicate)

        norma_errors = 0
        norma_total = 0
        delex_errors = 0
        delex_total = 0
        for root, dirs, files in os.walk(dataFile):
            for file in files:
                # We read the data from the data files.
                tree = ET.parse(os.path.join(root, file))
                num = 0
                # Each line corresponds to a MR
                for entry in tree.getroot().iter('entry'):
                    id = entry.attrib["eid"]
                    raw_MRTriples = []
                    raw_refs = []
                    for modtriple in entry.iter('modifiedtripleset'):
                        for triple in modtriple.iter('mtriple'):
                            raw_MRTriples.append(triple.text)
                    for ref in entry.iter('lex'):
                        refPart = re.sub("([.,?:;!'-])", " \g<1> ", ref.text.lower())
                        refPart = refPart.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace('"', ' " ').replace(", ,", " , ").replace(". .", " . ").strip()
                        refPart = " ".join(refPart.split())
                        refPart = " ".join(refPart.split())
                        raw_refs.append(refPart)

                    for refPart in raw_refs:
                        # Map attributes to their values
                        attributeSubjects = {}
                        attributeValues = {}
                        # Map from original values to delexicalized values
                        delexicalizedMap = {}
                        normalizationMap = {}

                        has_delex_errors = False
                        MRPart = " , ".join(raw_MRTriples)
                        for triple in raw_MRTriples:
                            components = triple.split("|")
                            subject = components[0].strip().lower()
                            subject = re.sub("([.,?:;!'-])", " \g<1> ", subject)
                            subject = subject.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace('"', ' " ').strip()
                            subject = ' '.join(subject.split())
                            value = components[2].strip().lower()
                            value = re.sub("([.,?:;!'-])", " \g<1> ", value)
                            value = value.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace('"', ' " ').strip()
                            value = ' '.join(value.split())
                            attribute = components[1].strip().lower()
                            if " " in attribute:
                                attribute = attribute.replace(" ", "_")

                            if subject not in normalizationMap.values():
                                normalizedValues = []
                                normalizedValues.append(subject.lower())
                                normalizedValues.append(self.normalize_quotes(subject))
                                normalizedValues.append(self.normalize_underscores(subject))
                                normalizedValues.append(self.normalize_capitalization(subject))
                                normalizedValues.extend(self.normalize_remove_comma_subphrase(subject))
                                normalizedValues.extend(self.normalize_remove_parenthesis(subject))
                                normalizedValues.extend(self.normalize_date(subject))
                                normalizedValues.extend(self.normalize_punctuation(normalizedValues))
                                normalizedValues.extend(self.normalize_various_typos(normalizedValues))

                                normalized = False
                                for norm in normalizedValues:
                                    if (" " + norm.lower() + " ") in (" " + refPart + " "):
                                        normalized = norm
                                        break

                                norma_total += 1
                                if normalized:
                                    normalizationMap[normalized] = subject
                                    subject = normalized
                                else:
                                    '''
                                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                    print(file, id)
                                    print(subject)
                                    print(attribute)
                                    print(normalizationMap)
                                    print(normalizedValues)
                                    print(refPart)
                                    errors += 1
                                    '''
                            else:
                                subject = list(normalizationMap.keys())[list(normalizationMap.values()).index(subject)]

                            if subject not in delexicalizedMap.values():
                                delexSubject = Action.TOKEN_X + attribute + "_0"
                                delexicalizedMap[delexSubject] = subject
                                subject = delexSubject
                            else:
                                subject = list(delexicalizedMap.keys())[list(delexicalizedMap.values()).index(subject)]

                            if value not in normalizationMap.values():
                                normalizedValues = []
                                normalizedValues.append(value.lower().strip())
                                normalizedValues.append(self.normalize_quotes(value))
                                normalizedValues.append(self.normalize_underscores(value))
                                normalizedValues.append(self.normalize_capitalization(value))
                                normalizedValues.extend(self.normalize_remove_comma_subphrase(value))
                                normalizedValues.extend(self.normalize_remove_parenthesis(value))
                                normalizedValues.extend(self.normalize_remove_attribute(value, attribute))
                                normalizedValues.extend(self.normalize_date(value))
                                normalizedValues.extend(self.normalize_punctuation(normalizedValues))
                                normalizedValues.extend(self.normalize_various_typos(normalizedValues))

                                normalized = False
                                for norm in normalizedValues:
                                    if (" " + norm.lower() + " ") in (" " + refPart + " "):
                                        normalized = norm
                                        break

                                norma_total += 1
                                if normalized:
                                    normalizationMap[normalized] = value
                                    value = normalized
                                else:
                                    '''
                                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                    print(file, id)
                                    print(value)
                                    print(attribute)
                                    print(normalizationMap)
                                    print(normalizedValues)
                                    print(refPart)
                                    '''
                                    norma_errors += 1
                            else:
                                value = list(normalizationMap.keys())[list(normalizationMap.values()).index(value)]

                            if value not in delexicalizedMap.values():
                                delexValue = Action.TOKEN_X + attribute + "_0"
                                delexicalizedMap[delexValue] = value
                                value = delexValue
                            else:
                                value = list(delexicalizedMap.keys())[list(delexicalizedMap.values()).index(value)]

                            if forTrain and singlePredicate not in self.attributes:
                                self.attributes[singlePredicate] = set()
                            if attribute:
                                if forTrain:
                                    self.attributes[singlePredicate].add(attribute)
                                attributeSubjects[attribute] = subject
                                attributeValues[attribute] = value
                        for deValue in delexicalizedMap.keys():
                            delex_total += 1
                            value = delexicalizedMap[deValue]
                            if (" " + value.lower() + " ") in (" " + refPart + " "):
                                refPart = (" " + refPart + " ").replace((" " + value.lower() + " "), (" " + deValue + " ")).strip()
                            else:
                                delex_errors += 1
                                has_delex_errors = True

                        observedWordSequence = []
                        if refPart and not has_delex_errors:
                            words = refPart.split()
                            for word in words:
                                word = word.strip()
                                if word:
                                    if "0f" in word:
                                        word = word.replace("0f", "of")

                                    m = re.search("^@x@([a-z]+)_([0-9]+)", word)
                                    if m and m.group(0) != word:
                                        var = m.group(0)
                                        realValue = delexicalizedMap.get(var)
                                        realValue = word.replace(var, realValue)
                                        delexicalizedMap[var] = realValue
                                        observedWordSequence.append(var.strip())
                                    else:
                                        m = re.match("([0-9]+)([a-z]+)", word)
                                        if m and m.group(1).strip() == "o":
                                            observedWordSequence.add(m.group(1).strip() + "0")
                                        elif m:
                                            observedWordSequence.append(m.group(1).strip())
                                            observedWordSequence.append(m.group(2).strip())
                                        else:
                                            m = re.match("([a-z]+)([0-9]+)", word)
                                            if m and (m.group(1).strip() == "l" or m.group(1).strip() == "e"):
                                                observedWordSequence.append("£" + m.group(2).strip())
                                            elif m:
                                                observedWordSequence.append(m.group(1).strip())
                                                observedWordSequence.append(m.group(2).strip())
                                            else:
                                                m = re.match("(£)([a-z]+)", word)
                                                if m:
                                                    observedWordSequence.append(m.group(1).strip())
                                                    observedWordSequence.append(m.group(2).strip())
                                                else:
                                                    m = re.match("([a-z]+)(£[0-9]+)", word)
                                                    if m:
                                                        observedWordSequence.append(m.group(1).strip())
                                                        observedWordSequence.append(m.group(2).strip())
                                                    else:
                                                        m = re.match("([0-9]+)([a-z]+)([0-9]+)", word)
                                                        if m:
                                                            observedWordSequence.append(m.group(1).strip())
                                                            observedWordSequence.append(m.group(2).strip())
                                                            observedWordSequence.append(m.group(3).strip())
                                                        else:
                                                            m = re.match("([0-9]+)(@x@[a-z]+_0)", word)
                                                            if m:
                                                                observedWordSequence.append(m.group(1).strip())
                                                                observedWordSequence.append(m.group(2).strip())
                                                            else:
                                                                m = re.match("(£[0-9]+)([a-z]+)", word)
                                                                if m and m.group(2).strip() == "o":
                                                                    observedWordSequence.append(m.group(1).strip() + "0")
                                                                else:
                                                                    observedWordSequence.append(word.strip())

                            MR = MeaningRepresentation(singlePredicate, attributeValues, MRPart, delexicalizedMap)
                            MR.attributeSubjects = attributeSubjects

                            # We store the maximum observed word sequence length, to use as a limit during generation
                            if forTrain and len(observedWordSequence) > self.maxWordSequenceLength:
                                self.maxWordSequenceLength = len(observedWordSequence)

                            # We initialize the alignments between words and attribute/value pairs
                            wordToAttrValueAlignment = []
                            for word in observedWordSequence:
                                if re.match("[.,?:;!'\"]", word.strip()):
                                    wordToAttrValueAlignment.append(Action.TOKEN_PUNCT)
                                else:
                                    wordToAttrValueAlignment.append("[]")
                            directReferenceSequence = []
                            for r, word in enumerate(observedWordSequence):
                                directReferenceSequence.append(Action(word, wordToAttrValueAlignment[r], "word"))
                                if forTrain:
                                    self.vocabulary.add(word)

                            alingedAttributes = []
                            if directReferenceSequence:
                                for action in directReferenceSequence:
                                    if action.label.startswith(Action.TOKEN_X):
                                        action.attribute = action.label[3:action.label.rfind('_')]
                                        alingedAttributes.append(action.attribute)

                            # If not all attributes are aligned, ignore the instance from training?
                            # Alternatively, we could align them randomly; certainly not ideal, but usually it concerns edge cases
                            if forTrain:
                                directReferenceSequence = inferNaiveAlignments(MR, directReferenceSequence)
                            DI = DatasetInstance(MR, directReferenceSequence, self.postProcessRef(MR, directReferenceSequence))
                            instances[singlePredicate].append(DI)

        print("Normalization errors ( -> delexicalization misses): {} / {}".format(norma_errors, norma_total))
        print("Delexicalization errors: {} / {}".format(delex_errors, delex_total))
        return instances

    def normalize_quotes(self, t):
        s = str(t)
        if s.startswith("\""):
            s = s[1:]
        if s.endswith("\""):
            s = s[:-1]
        return s.lower().strip()

    def normalize_underscores(self, t):
        s = self.normalize_quotes(t)
        if '_' in s:
            r = " ".join(s.split('_')).strip().lower()
            return ' '.join(r.split()).strip()
        return s

    def normalize_capitalization(self, t):
        s = self.normalize_quotes(t)
        if '_' in s:
            s = " ".join(s.split('_')).strip()
            s = ' '.join(s.split())
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', s)
        return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower().strip()

    def normalize_remove_parenthesis(self, t):
        s = self.normalize_quotes(t)
        if '_' in s:
            s = " ".join(s.split('_')).strip()
            s = ' '.join(s.split())
        if "(" in s and ")" in s:
            start = s.find("(")
            end = s.find(")")
            if start < end:
                s1 = "{}{}".format(s[:start], s[end + 1:]).strip().lower()
                s1 = " ".join(s1.split()).strip()
                s2 = "{}{}{}".format(s[:start], s[start + 1:end], s[end + 1:]).strip().lower()
                s2 = " ".join(s2.split()).strip()
                return [s1, s2]
        return [s]

    def normalize_remove_attribute(self, t, a):
        s = self.normalize_quotes(t)
        if '_' in s:
            s = " ".join(s.split('_')).strip()
            s = ' '.join(s.split())
        if a in s:
            start = s.find(a)
            end = start + len(a)
            s1 = "{}".format(s[:start], s[end + 1:]).strip().lower()
            s1 = " ".join(s1.split()).strip()
            return [s1]
        return [s]

    def normalize_remove_comma_subphrase(self, t):
        s = self.normalize_quotes(t)
        if '_' in s:
            s = " ".join(s.split('_')).strip()
            s = ' '.join(s.split())
        if " , " in s:
            start = s.find(",")
            s1 = " ".join(s[:start].split()).lower().strip()
            s2 = "{}{}".format(s[:start], s[start + 1:]).strip().lower()
            s2 = " ".join(s2.split()).strip()
            s3 = "{} of {}".format(s[:start], s[start + 1:]).strip().lower()
            s3 = " ".join(s3.split()).strip()
            s4 = "{} in {}".format(s[:start], s[start + 1:]).strip().lower()
            s4 = " ".join(s4.split()).strip()
            return [s1, s2, s3, s4]
        return [s]

    def normalize_punctuation(self, list_t):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        corrected = []
        for t in list_t:
            s = regex.sub('', t).strip()
            s = " ".join(s.split())
            corrected.append(s.strip())
        return corrected

    def normalize_various_typos(self, list_t):
        corrected = []
        for t in list_t:
            t = " " + t + " "
            if " st " in t:
                corrected.append(t.replace(" st ", " st . ").strip())
            if " . 0 " in t:
                corrected.append(t.replace(" . 0 ", " ").strip())
            if " million " in t:
                corrected.append(t.replace(" million ", " m ").strip())
                corrected.append(t.replace(" million ", "m ").strip())
            if " m " in t:
                corrected.append(t.replace(" m ", " million ").strip())
                corrected.append(t.replace(" m ", " meter ").strip())
                corrected.append(t.replace(" m ", " meters ").strip())
            if " ' s " in t:
                corrected.append(t.replace(" ' s ", "s ").strip())
            if " united states " in t:
                corrected.append(t.replace(" united states ", " us ").strip())
                corrected.append(t.replace(" united states ", " usa ").strip())
                corrected.append(t.replace(" united states ", " u . s . ").strip())
                corrected.append(t.replace(" united states ", " u . s . a . ").strip())
        return corrected

    def normalize_date(self, t):
        try:
            norms = set()
            s = self.normalize_quotes(t)
            d = dateparser.parse(s)

            if d.day == 1:
                day_desc = "st"
            elif d.day == 2:
                day_desc = "nd"
            elif d.day == 3:
                day_desc = "rd"
            else:
                day_desc = "th"

            month_desc = False
            if d.month == 1:
                month_desc = "january"
            elif d.month == 2:
                month_desc = "february"
            elif d.month == 3:
                month_desc = "march"
            if d.month == 4:
                month_desc = "april"
            elif d.month == 5:
                month_desc = "may"
            elif d.month == 6:
                month_desc = "june"
            if d.month == 7:
                month_desc = "july"
            elif d.month == 8:
                month_desc = "august"
            elif d.month == 9:
                month_desc = "september"
            if d.month == 10:
                month_desc = "october"
            elif d.month == 11:
                month_desc = "november"
            elif d.month == 12:
                month_desc = "december"

            norms.add("{}{} {} {}".format(d.day, day_desc, month_desc, d.year))
            norms.add("{}{} {} , {}".format(d.day, day_desc, month_desc, d.year))
            norms.add("{}{} {} of {}".format(d.day, day_desc, month_desc, d.year))
            norms.add("{} {}{} {}".format(month_desc, d.day, day_desc, d.year))
            norms.add("{} {}{} , {}".format(month_desc, d.day, day_desc, d.year))
            norms.add("{} {}{} of {}".format(month_desc, d.day, day_desc, d.year))
            norms.add("{} the {}{} of {}".format(month_desc, d.day, day_desc, d.year))
            norms.add("{} the {}{} , {}".format(month_desc, d.day, day_desc, d.year))
            norms.add("{} the {}{} {}".format(month_desc, d.day, day_desc, d.year))

            norms.add("{} {}{}".format(month_desc, d.day, day_desc))
            norms.add("{} the {}{}".format(month_desc, d.day, day_desc))

            norms.add("{} {}".format(month_desc, d.year))
            norms.add("{} , {}".format(month_desc, d.year))
            norms.add("{} of {}".format(month_desc, d.year))
            return norms
        except ValueError:
            return set()

    def check_cache_path(self):
        save_cache_path = os.path.abspath('cache/')
        model_dirname = os.path.dirname(save_cache_path)
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)

    def loadTrainingLists(self, trim, full_delex, infer_MRs):
        print("Attempting to load training data...")
        self.vocabulary = False
        self.predicates = False
        self.attributes = False
        self.valueAlignments = False
        self.trainingInstances = {}
        self.maxWordSequenceLength = False

        fileNotFound = False

        if trim:
            trim = 'true'
        else:
            trim = 'false'
        if full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        if infer_MRs:
            fileSuffix = self.dataset_name + '_trim=' + str(trim) + '_full_delex=' + str(full_delex) + '_inferMRs=True.pickle'
        else:
            fileSuffix = self.dataset_name + '_trim=' + str(trim) + '_full_delex=' + str(full_delex) + '.pickle'

        if os.path.isfile(self.base_dir + 'cache/vocabulary_' + fileSuffix):
            with open(self.base_dir + 'cache/vocabulary_' + fileSuffix, 'rb') as handle:
                self.vocabulary = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/vocabulary_per_attr_' + fileSuffix):
            with open(self.base_dir + 'cache/vocabulary_per_attr_' + fileSuffix, 'rb') as handle:
                self.vocabulary_per_attr = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/predicates_' + fileSuffix):
            with open(self.base_dir + 'cache/predicates_' + fileSuffix, 'rb') as handle:
                self.predicates = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/attributes_' + fileSuffix):
            with open(self.base_dir + 'cache/attributes_' + fileSuffix, 'rb') as handle:
                self.attributes = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/available_values_' + fileSuffix):
            with open(self.base_dir + 'cache/available_values_' + fileSuffix, 'rb') as handle:
                self.available_values = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/available_subjects_' + fileSuffix):
            with open(self.base_dir + 'cache/available_subjects_' + fileSuffix, 'rb') as handle:
                self.available_subjects = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/valueAlignments_' + fileSuffix):
            with open(self.base_dir + 'cache/valueAlignments_' + fileSuffix, 'rb') as handle:
                self.valueAlignments = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/trainingInstances_' + fileSuffix):
            with open(self.base_dir + 'cache/trainingInstances_' + fileSuffix, 'rb') as handle:
                self.trainingInstances = pickle.load(handle)
        else:
            fileNotFound = True
        if os.path.isfile(self.base_dir + 'cache/maxWordSequenceLength_' + fileSuffix):
            with open(self.base_dir + 'cache/maxWordSequenceLength_' + fileSuffix, 'rb') as handle:
                self.maxWordSequenceLength = pickle.load(handle)
        else:
            fileNotFound = True

        if os.path.isfile(self.base_dir + 'cache/ngramListsPerWordSequence_' + fileSuffix):
            with open(self.base_dir + 'cache/ngramListsPerWordSequence_' + fileSuffix, 'rb') as handle:
                self.ngram_lists_per_word_sequence = pickle.load(handle)
        else:
            fileNotFound = True

        if os.path.isfile(self.base_dir + 'cache/ngramListsPerRelexedWordSequence_' + fileSuffix):
            with open(self.base_dir + 'cache/ngramListsPerRelexedWordSequence_' + fileSuffix, 'rb') as handle:
                self.ngram_lists_per_relexed_word_sequence = pickle.load(handle)
        else:
            fileNotFound = True

        if os.path.isfile(self.base_dir + 'cache/train_src_to_di_' + fileSuffix):
            with open(self.base_dir + 'cache/train_src_to_di_' + fileSuffix, 'rb') as handle:
                self.train_src_to_di = pickle.load(handle)
        else:
            fileNotFound = True

        if not fileNotFound:
            print("done!")
            return True
        print("failed!")
        return False

    def loadDevelopmentLists(self, full_delex):
        print("Attempting to load development data...")
        self.developmentInstances = {}

        if full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        if os.path.isfile(self.base_dir + 'cache/developmentInstances_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle'):
            with open(self.base_dir + 'cache/developmentInstances_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'rb') as handle:
                self.developmentInstances = pickle.load(handle)

        if os.path.isfile(self.base_dir + 'cache/dev_src_to_di_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle'):
            with open(self.base_dir + 'cache/dev_src_to_di_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'rb') as handle:
                self.dev_src_to_di = pickle.load(handle)

        if self.developmentInstances and self.dev_src_to_di:
            print("done!")
            return True
        print("failed!")
        return False

    def loadTestingLists(self, full_delex):
        print("Attempting to load testing data...")
        self.testingInstances = {}

        if full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        if os.path.isfile(self.base_dir + 'cache/testingInstances_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle'):
            with open(self.base_dir + 'cache/testingInstances_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'rb') as handle:
                self.testingInstances = pickle.load(handle)

        if os.path.isfile(self.base_dir + 'cache/test_src_to_di_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle'):
            with open(self.base_dir + 'cache/test_src_to_di_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'rb') as handle:
                self.test_src_to_di = pickle.load(handle)

        if self.testingInstances and self.test_src_to_di:
            print("done!")
            return True
        print("failed!")
        return False

    def loadLanguageModels(self):
        print("Attempting to load language models...")
        self.contentLMs_perPredicate = False
        self.wordLMs_perPredicate = False

        if os.path.isfile(self.base_dir + 'cache/contentLM_' + self.dataset_name + '.pickle'):
            with open(self.base_dir + 'cache/contentLM_' + self.dataset_name + '.pickle', 'rb') as handle:
                self.contentLMs_perPredicate = pickle.load(handle)
        if os.path.isfile(self.base_dir + 'cache/wordLM_' + self.dataset_name + '.pickle'):
            with open(self.base_dir + 'cache/wordLM_' + self.dataset_name + '.pickle', 'rb') as handle:
                self.wordLMs_perPredicate = pickle.load(handle)

        if self.contentLMs_perPredicate and self.wordLMs_perPredicate:
            print("done!")
            return True
        print("failed!")
        return False

    def writeTrainingLists(self, trim, full_delex, infer_MRs):
        print("Writing training data...")

        if trim:
            trim = 'true'
        else:
            trim = 'false'
        if full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        if infer_MRs:
            fileSuffix = self.dataset_name + '_trim=' + str(trim) + '_full_delex=' + str(full_delex) + '_inferMRs=True.pickle'
        else:
            fileSuffix = self.dataset_name + '_trim=' + str(trim) + '_full_delex=' + str(full_delex) + '.pickle'

        with open(self.base_dir + 'cache/vocabulary_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.vocabulary, handle)
        with open(self.base_dir + 'cache/vocabulary_per_attr_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.vocabulary_per_attr, handle)
        with open(self.base_dir + 'cache/predicates_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.predicates, handle)
        with open(self.base_dir + 'cache/attributes_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.attributes, handle)
        with open(self.base_dir + 'cache/available_values_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.available_values, handle)
        with open(self.base_dir + 'cache/available_subjects_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.available_subjects, handle)
        with open(self.base_dir + 'cache/valueAlignments_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.valueAlignments, handle)
        with open(self.base_dir + 'cache/trainingInstances_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.trainingInstances, handle)
        with open(self.base_dir + 'cache/maxWordSequenceLength_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.maxWordSequenceLength, handle)
        with open(self.base_dir + 'cache/ngramListsPerWordSequence_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.ngram_lists_per_word_sequence, handle)
        with open(self.base_dir + 'cache/ngramListsPerRelexedWordSequence_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.ngram_lists_per_relexed_word_sequence, handle)
        with open(self.base_dir + 'cache/train_src_to_di_' + fileSuffix, 'wb') as handle:
            pickle.dump(self.train_src_to_di, handle)

    def write_onmt_data(self, opt):
        print("Writing onmt training data...")
        gen_templ = self.get_onmt_file_templ(opt)
        train_src_templ, train_tgt_templ, train_eval_refs_templ, valid_src_templ, valid_tgt_templ, valid_eval_refs_templ, test_src_templ, test_tgt_templ, test_eval_refs_templ = self.get_onmt_file_templs(gen_templ)

        for predicate in self.predicates:
            if predicate in self.trainingInstances:
                with open(train_src_templ.format(predicate), 'w') as handle:
                    handle.writelines([item.input.nn_src + '\n' for item in self.trainingInstances[predicate]])
                with open(train_tgt_templ.format(predicate), 'w') as handle:
                    handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence if w.label != Action.TOKEN_SHIFT]) for item in self.trainingInstances[predicate]])
                    # handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence]) for item in self.trainingInstances[predicate]])
                with open(train_eval_refs_templ.format(predicate), 'w') as handle:
                    for di in self.trainingInstances[predicate]:
                        handle.writelines(["%s\n" % item for item
                                           in di.output.evaluationReferences])
                        handle.writelines(["\n"])

            if predicate in self.developmentInstances:
                with open(valid_src_templ.format(predicate), 'w') as handle:
                    handle.writelines([item.input.nn_src + '\n' for item in self.developmentInstances[predicate]])
                with open(valid_tgt_templ.format(predicate), 'w') as handle:
                    handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence if w.label != Action.TOKEN_SHIFT]) for item in self.developmentInstances[predicate]])
                    # handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence]) for item in self.developmentInstances[predicate]])
                with open(valid_eval_refs_templ.format(predicate), 'w') as handle:
                    for di in self.developmentInstances[predicate]:
                        handle.writelines(["%s\n" % item for item
                                           in di.output.evaluationReferences])
                        handle.writelines(["\n"])

            '''
            for predicate in self.predicates:
                if predicate in self.trainingInstances:
                    with open(train_src_templ.format(predicate), 'w') as handle:
                        handle.writelines(["%s\n" % " ".join(["{} {}".format(attr, item.input.attributeValues[attr]) if attr in item.input.attributeValues else "{}@none@ {}_value@none@".format(attr, attr) for attr in sorted(self.attributes[predicate])]) for item in self.trainingInstances[predicate][:1]])
                    with open(train_tgt_templ.format(predicate), 'w') as handle:
                        handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence if w.label != Action.TOKEN_SHIFT]) for item in self.trainingInstances[predicate][:1]])
                    with open(train_eval_refs_templ.format(predicate), 'w') as handle:
                        for di in self.trainingInstances[predicate][:1]:
                            handle.writelines(["%s\n" % item for item
                                               in di.output.evaluationReferences])
                            handle.writelines(["\n"])
    
                if predicate in self.developmentInstances:
                    with open(valid_src_templ.format(predicate), 'w') as handle:
                        handle.writelines(["%s\n" % " ".join(["{} {}".format(attr, item.input.attributeValues[attr]) if attr in item.input.attributeValues else "{}@none@ {}_value@none@".format(attr, attr) for attr in sorted(self.attributes[predicate])]) for item in self.trainingInstances[predicate][:1]])
                    with open(valid_tgt_templ.format(predicate), 'w') as handle:
                        handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence if w.label != Action.TOKEN_SHIFT]) for item in self.trainingInstances[predicate][:1]])
                    with open(valid_eval_refs_templ.format(predicate), 'w') as handle:
                        for di in self.trainingInstances[predicate][:1]:
                            handle.writelines(["%s\n" % item for item
                                               in di.output.evaluationReferences])
                            handle.writelines(["\n"])
            '''

            if predicate in self.testingInstances:
                with open(test_src_templ.format(predicate), 'w') as handle:
                    handle.writelines([item.input.nn_src + '\n' for item in self.testingInstances[predicate]])
                with open(test_tgt_templ.format(predicate), 'w') as handle:
                    handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence if w.label != Action.TOKEN_SHIFT]) for item in self.testingInstances[predicate]])
                    # handle.writelines(["%s\n" % " ".join([w.label for w in item.directReferenceSequence]) for item in self.testingInstances[predicate]])
                with open(test_eval_refs_templ.format(predicate), 'w') as handle:
                    for di in self.testingInstances[predicate]:
                        handle.writelines(["%s\n" % item for item
                                           in di.output.evaluationReferences])
                        handle.writelines(["\n"])

    def get_onmt_file_templ(self, opt):
        if opt.trim:
            trim = 'true'
        else:
            trim = 'false'
        if opt.full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        if opt.infer_MRs:
            fileSuffix = self.dataset_name + '_trim=' + str(trim) + '_full_delex=' + str(full_delex) + '_inferMRs=True'
        else:
            fileSuffix = self.dataset_name + '_trim=' + str(trim) + '_full_delex=' + str(full_delex)

        return '{:s}_{:s}'.format('{:s}', fileSuffix)

    def get_onmt_file_templs(self, gen_templ):
        train_src_templ = '{:s}cache/train_src_{:s}.txt'.format(self.base_dir, gen_templ)
        train_tgt_templ = '{:s}cache/train_tgt_{:s}.txt'.format(self.base_dir, gen_templ)
        train_eval_refs_templ = '{:s}cache/train_eval_refs_{:s}.txt'.format(self.base_dir, gen_templ)
        valid_src_templ = '{:s}cache/valid_src_{:s}.txt'.format(self.base_dir, gen_templ)
        valid_tgt_templ = '{:s}cache/valid_tgt_{:s}.txt'.format(self.base_dir, gen_templ)
        valid_eval_refs_templ = '{:s}cache/valid_eval_refs_{:s}.txt'.format(self.base_dir, gen_templ)
        test_src_templ = '{:s}cache/test_src_{:s}.txt'.format(self.base_dir, gen_templ)
        test_tgt_templ = '{:s}cache/test_tgt_{:s}.txt'.format(self.base_dir, gen_templ)
        test_eval_refs_templ = '{:s}cache/test_eval_refs_{:s}.txt'.format(self.base_dir, gen_templ)

        return train_src_templ, train_tgt_templ, train_eval_refs_templ, valid_src_templ, valid_tgt_templ, valid_eval_refs_templ, test_src_templ, test_tgt_templ, test_eval_refs_templ

    def writeDevelopmentLists(self, full_delex):
        print("Writing development data...")

        if full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        with open(self.base_dir + 'cache/developmentInstances_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'wb') as handle:
            pickle.dump(self.developmentInstances, handle)
        with open(self.base_dir + 'cache/dev_src_to_di_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'wb') as handle:
            pickle.dump(self.dev_src_to_di, handle)

    def writeTestingLists(self, full_delex):
        print("Writing testing data...")

        if full_delex:
            full_delex = 'true'
        else:
            full_delex = 'false'

        with open(self.base_dir + 'cache/testingInstances_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'wb') as handle:
            pickle.dump(self.testingInstances, handle)
        with open(self.base_dir + 'cache/test_src_to_di_' + self.dataset_name + '_full_delex=' + str(full_delex) + '.pickle', 'wb') as handle:
            pickle.dump(self.test_src_to_di, handle)


    @staticmethod
    def postProcessRef(mr, refSeq, punct=True):
        cleanedWords = ""
        for nlWord in refSeq:
            if nlWord.label != Action.TOKEN_SHIFT and nlWord.label != Action.TOKEN_PUNCT:
                if nlWord.label.startswith(Action.TOKEN_X):
                    cleanedWords += " " + mr.delexicalizationMap[nlWord.label]
                else:
                    cleanedWords += " " + nlWord.label
        cleanedWords = cleanedWords.strip()
        if not cleanedWords.endswith(".") and punct:
            cleanedWords += " ."
        return cleanedWords.strip()

    @staticmethod
    def find_subList_in_actionList(sl, l):
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e.label == sl[0].label):
            if [o.label for o in l[ind:ind + sll]] == [r.label for r in sl]:
                return ind, ind + sll - 1

    def trimTrainingSpace(self):
        # Keep only unique training data
        # Need to keep only one direct ref, we chose the one with 1) the most attrsvalues expressed, 2) most variables, and 3) the greatest avg. word freq
        for predicate in self.trainingInstances:
            uniqueTrainingMRs = {}
            for datasetInstance in self.trainingInstances[predicate]:
                if datasetInstance.input.getAbstractMR() not in uniqueTrainingMRs:
                    uniqueTrainingMRs[datasetInstance.input.getAbstractMR()] = []
                uniqueTrainingMRs[datasetInstance.input.getAbstractMR()].append(datasetInstance)
            uniqueTrainingInstances = []
            for uniqueMR in uniqueTrainingMRs:
                # Keep the dis that follow the most probable sequence of attrValues

                # keep the di whose direct reference's words have the greatest frequency
                bestDIs = set()
                mostVars = 0
                for di in uniqueTrainingMRs[uniqueMR]:
                    vars = 0
                    str = " ".join([o.label for o in di.directReferenceSequence])
                    str = " " + str + " "
                    for v in di.input.attributeValues.values():
                        if v in str:
                            vars += 1
                        if v.endswith('_no') and (' no ' in str or ' not ' in str or " n't " in str):
                            vars += 1
                    if vars > mostVars:
                        mostVars = vars
                        bestDIs = set()
                    if vars == mostVars:
                        bestDIs.add(di)

                if len(bestDIs) == 1:
                    for di in bestDIs:
                        uniqueTrainingInstances.append(di)
                else:
                    bestDI = False
                    maxAvgFreq = 0.0
                    for datasetInstance in bestDIs:
                        avgFreq = 0.0
                        total = 0.0
                        for a in datasetInstance.directReferenceSequence:
                            if a.label != Action.TOKEN_SHIFT:
                                avgFreq += self.availableWordCounts[predicate][a.label]
                                total += 1.0
                        if total != 0.0:
                            avgFreq /= total
                        if avgFreq >= maxAvgFreq:
                            maxAvgFreq = avgFreq
                            bestDI = datasetInstance
                    if bestDI:
                        uniqueTrainingInstances.append(bestDI)
                    #else:
                    #    print("Couldn't find appropriate DI")
                    #    print(uniqueMR)
                    #    print(uniqueTrainingMRs[uniqueMR])

            self.trainingInstances[predicate] = uniqueTrainingInstances[:]
            print("Trimmed training data size for {}: {}".format(predicate, len(self.trainingInstances[predicate])))
            print("-----------------------")

    def initializeActionSpace(self):
        for predicate in self.attributes:
            self.availableContentActions[predicate] = set(self.attributes[predicate])

        for predicate in self.trainingInstances:
            self.availableWordActions[predicate] = {}
            self.availableWordCounts[predicate] = Counter()
            for attr in self.attributes[predicate]:
                self.availableWordActions[predicate][attr] = set()
                self.availableWordActions[predicate][attr].add(Action.TOKEN_SHIFT)

            for datasetInstance in self.trainingInstances[predicate]:
                attrs = [cleanAndGetAttr(attr) for attr in datasetInstance.directAttrSequence if attr != Action.TOKEN_SHIFT]
                for a in datasetInstance.directReferenceSequence:
                    if a.label.strip() and a.label != Action.TOKEN_SHIFT:
                        self.availableWordCounts[predicate][a.label] += 1
                        if a.label.startswith(Action.TOKEN_X):
                            self.availableWordActions[predicate][cleanAndGetAttr(a.attribute)].add(a.label)
                        else:
                            for attr in attrs:
                                self.availableWordActions[predicate][attr].add(a.label)
            for attr in self.attributes[predicate]:
                print("{}:{} action count: {}".format(predicate, attr, len(self.availableWordActions[predicate][attr])))
            print("-----------------------")


def inferNaiveAlignments(meaning_representation, sequence):
    attrSeq = [o.attribute for o in sequence]
    while True:
        changes = {}
        for i, attr in enumerate(attrSeq):
            if attr != "[]" and attr != Action.TOKEN_PUNCT:
                if i - 1 >= 0 and attrSeq[i - 1] == "[]":
                    if attr not in changes:
                        changes[attr] = set()
                    changes[attr].add(i - 1)
                if i + 1 < len(attrSeq) and attrSeq[i + 1] == "[]":
                    if attr not in changes:
                        changes[attr] = set()
                    changes[attr].add(i + 1)
        for attr in changes:
            for index in changes[attr]:
                attrSeq[index] = attr
                sequence[index].attribute = attr
        if not changes:
            break
    attrSet = set([o for o in attrSeq if o != '@punct@' and o != '[]'])
    if not attrSet:
        print(attrSeq)
        print(meaning_representation.attributeValues)
        exit()
    while "[]" in attrSeq:
        index = attrSeq.index("[]")
        copyFrom = index - 1
        while copyFrom >= 0:
            if attrSeq[copyFrom] != "[]" and attrSeq[copyFrom] != Action.TOKEN_PUNCT:
                attrSeq[index] = attrSeq[copyFrom]
                sequence[index].attribute = attrSeq[copyFrom]
                copyFrom = -1
            elif attrSeq[copyFrom] == Action.TOKEN_PUNCT:
                copyFrom = -1
            else:
                copyFrom -= 1
        if attrSeq[index] == "[]":
            copyFrom = index + 1
            while copyFrom < len(attrSeq):
                if attrSeq[copyFrom] != "[]" and attrSeq[copyFrom] != Action.TOKEN_PUNCT:
                    attrSeq[index] = attrSeq[copyFrom]
                    sequence[index].attribute = attrSeq[copyFrom]
                    copyFrom = len(attrSeq)
                elif attrSeq[copyFrom] == Action.TOKEN_PUNCT:
                    copyFrom = len(attrSeq)
                else:
                    copyFrom += 1
        if attrSeq[index] == "[]":
            copyFrom = index - 1
            while copyFrom >= 0:
                if attrSeq[copyFrom] != "[]" and attrSeq[copyFrom] != Action.TOKEN_PUNCT:
                    attrSeq[index] = attrSeq[copyFrom]
                    sequence[index].attribute = attrSeq[copyFrom]
                    copyFrom = -1
                else:
                    copyFrom -= 1
            if attrSeq[index] == "[]":
                copyFrom = index + 1
                while copyFrom < len(attrSeq):
                    if attrSeq[copyFrom] != "[]" and attrSeq[copyFrom] != Action.TOKEN_PUNCT:
                        attrSeq[index] = attrSeq[copyFrom]
                        sequence[index].attribute = attrSeq[copyFrom]
                        copyFrom = len(attrSeq)
                    else:
                        copyFrom += 1

    # TODO decide on how we treat punctuations this time around
    '''
    while Action.TOKEN_PUNCT in attrSeq:
        index = attrSeq.index(Action.TOKEN_PUNCT)
        del attrSeq[index]
        del sequence[index]
    '''

    while Action.TOKEN_PUNCT in attrSeq:
        index = attrSeq.index(Action.TOKEN_PUNCT)
        if index > 0:
            i = index - 1
            while attrSeq[i] == Action.TOKEN_PUNCT and i >= 0:
                i -= 1
            attrSeq[index] = attrSeq[i]
            sequence[index].attribute = attrSeq[i]
        else:
            i = index + 1
            while attrSeq[i] == Action.TOKEN_PUNCT and i < len(attrSeq):
                i += 1
            attrSeq[index] = attrSeq[i]
            sequence[index].attribute = attrSeq[i]

    for act in sequence:
        act.attribute = "{}={}".format(act.attribute, meaning_representation.attributeValues[act.attribute])

    currentAttr = False
    for i, act in enumerate(sequence):
        if currentAttr and act.attribute != currentAttr:
            sequence.insert(i, Action(Action.TOKEN_SHIFT, currentAttr, "word"))
        currentAttr = act.attribute
    sequence.append(Action(Action.TOKEN_SHIFT, Action.TOKEN_SHIFT, "word"))
    return sequence