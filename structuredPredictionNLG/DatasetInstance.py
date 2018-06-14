# Gerasimos Lampouras, 2017:
from structuredPredictionNLG.Action import Action
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams
from structuredPredictionNLG.FullDelexicalizator import covers_attr_E2E
'''
 This represents a single instance in the dataset. 
'''
class DatasetInstance:

    def __init__(self, MR, directReferenceSequence, directReference):
        self.input = MR
        self.output = NLGOutput(self.input)

        self.output_abstract = False
        self.output_inc_training = False

        # A reference for the word actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset
        self.directReferenceSequence = directReferenceSequence
        # Realized string of the word actions in the direct reference
        self.directReference = directReference

        # A reference for the word actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset and not processed further
        # Primarily used as a cache for the reference, to be used in "cache baselines"
        self.originalDirectReferenceSequence = False
        # A reference for the content actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset
        self.directAttrSequence = []
        self.directAttrValueSequence = []
        previousAttr = ""
        for act in self.directReferenceSequence:
            if act.attribute != previousAttr:
                if act.attribute != Action.TOKEN_SHIFT and act.attribute != Action.TOKEN_PUNCT and act.attribute != '[]':
                    self.directAttrSequence.append(cleanAndGetAttr(act.attribute))
                elif act.attribute == '[]' and act.label.startswith(Action.TOKEN_X):
                    self.directAttrSequence.append(act.label[3:act.label.rfind('_')])
                elif act.attribute == Action.TOKEN_SHIFT:
                    self.directAttrSequence.append(act.attribute)

                if self.directAttrSequence and self.directAttrSequence[-1] in self.input.attributeValues:
                    self.directAttrValueSequence.append("{}={}".format(self.directAttrSequence[-1], self.input.attributeValues[self.directAttrSequence[-1]]))
                elif self.directAttrSequence:
                    self.directAttrValueSequence.append(self.directAttrSequence[-1])
            if act.attribute != Action.TOKEN_PUNCT:
                previousAttr = act.attribute
        # File from which DI was parsed; may be used for training/development/testing seperation
        self.originFile = False
        # References for the content actions of the DatasetInstance; this is constructed using the references
        # corresponding to any identical instance in the dataset

    def init_alt_outputs(self):
        self.output_abstract = NLGOutput(self.input)
        self.output_inc_training = NLGOutput(self.input)

    '''
     Sets the word action sequence (and also constructs the corresponding content action sequence) 
     to be used as direct reference sequence for the DatasetInstance.
     @param directReferenceSequence The word action sequence to be set.
    '''
    def setDirectReferenceSequence(self, directReferenceSequence):
        self.directReferenceSequence = directReferenceSequence
        self.directAttrSequence = []
        self.directAttrValueSequence = []
        previousAttr = ""
        for act in directReferenceSequence:
            if act.attribute != previousAttr:
                if act.attribute != Action.TOKEN_SHIFT:
                    self.directAttrSequence.append(act.attribute)
                else:
                    self.directAttrSequence.append(Action.TOKEN_SHIFT)

                if self.directAttrSequence[-1] in self.input.attributeValues:
                    self.directAttrValueSequence.append("{}={}".format(self.directAttrSequence[-1], self.input.attributeValues[self.directAttrSequence[-1]]))
                else:
                    self.directAttrValueSequence.append(self.directAttrSequence[-1])
            if act.attribute != Action.TOKEN_PUNCT:
                previousAttr = act.attribute


def cleanAndGetAttr(s):
    if "¬" in s:
        s = s[:s.find("¬")]
    if "=" in s:
        s = s[:s.find("=")]
    return s


def cleanAndGetValue(s):
    if "=" in s:
        return s[s.find("=") + 1:]
    return ""


def cleanAndGetAttrAndValue(s):
    if "¬" in s:
        s = s[:s.find("¬")]
    if "=" in s:
        s = s[:s.find("=")]
    return s


# TODO check that this works correctly
def endsWith(phrase, subPhrase):
    if len(subPhrase) > len(phrase):
        return False
    if phrase[-len(subPhrase):] == subPhrase:
        return True
    return False

def lexicalize_word_sequence(sequence, delex_map, complex_relex=False):
    real = []
    for act in sequence:
        if act.label != Action.TOKEN_GO and act.label != Action.TOKEN_SHIFT and not act.label.startswith('¬'):
            if act.label in delex_map:
                if real and complex_relex:
                    v = delex_map[act.label].split()
                    for a in v:
                        if real[-1] != a:
                            real.append(a)
                else:
                    real.append(delex_map[act.label])
            elif act.label == '-ly':
                real[-1] = real[-1] + act.label[1:]
            else:
                if real and complex_relex:
                    if real[-1] != act.label:
                        real.append(act.label)
                else:
                    real.append(act.label)
        elif act.label.startswith('¬'):
            real = apply_non_monotonic_action(real, act.label)
    return (" ".join([o for o in real])).strip()

def apply_non_monotonic_action(seq, non_monotonic_action):
    components = non_monotonic_action.split('|')
    if components[0] == '¬del':
        del seq[int(components[1])]
    elif components[1] == '¬ins':
        seq.insert(int(components[1]), components[2])
    elif components[1] == '¬rep':
        seq[int(components[1])] = components[2]
    return seq

class NLGOutput:
    def __init__(self, MR):
        # References to be used during evaluation of this DatasetInstance
        self.evaluationReferences = set()
        self.evaluationReferenceSequences = []
        self.evaluationReferenceActionSequences = []
        self.evaluationReferenceAttrValueSequences = []
        self.values = set()
        self.chencherry = SmoothingFunction()

        for attr, value in MR.attributeValues.items():
            if value in MR.delexicalizationMap:
                self.values.add(MR.delexicalizationMap[value].lower())
            else:
                self.values.add(value.lower())

    '''
     Infers sequences of content actions
     based on the references of this DatasetInstance.
    '''
    def calcEvaluationReferenceAttrValueSequences(self):
        self.evaluationReferenceAttrValueSequences = []
        for evaluationReferenceSequence in self.evaluationReferenceActionSequences:
            evaluation_reference_attr_value_sequence = []
            previousAttr = ""
            for act in evaluationReferenceSequence:
                if act.attribute != previousAttr:
                    if act.attribute != Action.TOKEN_SHIFT:
                        evaluation_reference_attr_value_sequence.append(act.attribute)
                if act.attribute != Action.TOKEN_PUNCT:
                    previousAttr = act.attribute
            self.evaluationReferenceAttrValueSequences.append(evaluation_reference_attr_value_sequence)

    # it must return an evalStats object with a loss
    def compareAgainstContent(self, predicted):
        evalStats = NLGEvalStats()

        maxBLEU = 0.0
        weights = (0.25, 0.25, 0.25, 0.25)
        if len(predicted) < 4:
            weights = (1 / len(predicted),) * len(predicted)
        for ref in self.evaluationReferenceAttrValueSequences:
            bleuOriginal = sentence_bleu([ref], predicted, weights, smoothing_function=self.chencherry.method2)
            if bleuOriginal > maxBLEU:
                maxBLEU = bleuOriginal

            # todo resolve issues with Rouge library, add it to cost metric
            '''
            maxROUGE = 0.0;
            for ref in refs:
                scores = rouge.get_scores(ref.lower(), gen.lower())
                print(scores)
                exit()
                if bleuOriginal > maxROUGE:
                    maxROUGE = bleuOriginal
            return (maxBLEU + maxROUGE) / 2.0
            '''
        evalStats.BLEU = maxBLEU
        evalStats.loss = 1.0 - maxBLEU
        return evalStats

    # it must return an evalStats object with a loss
    def compareAgainst(self, predicted, delexMap, against_refs = False):
        evalStats = NLGEvalStats()

        maxBLEU = 0.0
        maxROUGE = 0.0;
        maxCoverage = 0.0;
        weights = (0.25, 0.25, 0.25, 0.25)
        if predicted:
            against = self.evaluationReferenceSequences
            if against_refs:
                against = against_refs
            for ref in against:
                ref_mod = ref[:]
                predicted_mod = predicted[:]
                if len(ref_mod) > len(predicted_mod):
                    predicted_mod.extend(['@pred@' for i in range(len(ref_mod) - len(predicted_mod))])
                elif len(ref_mod) < len(predicted_mod):
                    ref_mod.extend(['@ref@' for i in range(len(predicted_mod) - len(ref_mod))])
                bleuOriginal = sentence_bleu([ref_mod], predicted_mod, weights, smoothing_function=self.chencherry.method2)
                if bleuOriginal > maxBLEU:
                    maxBLEU = bleuOriginal

                # if len(ref) != len(predicted):
                #    print(ref_mod)
                #    print(predicted_mod)
                rougeOriginal = self.rouge_n(ref_mod, predicted_mod, 4)
                if rougeOriginal > maxROUGE:
                    maxROUGE = rougeOriginal
                coverage = 0
                for value in self.values:
                    if value in delexMap.values():
                        value = list(delexMap.keys())[list(delexMap.values()).index(value)]
                    if value.endswith('_no'):
                        value = value[:-3]
                    if value.endswith('_yes'):
                        value = value[:-4]
                    if value == 'family friendly':
                        value = 'family'
                    if value in " ".join(predicted):
                        coverage += 1
                coverage /= len(self.values)
                if coverage > maxCoverage:
                    maxCoverage = coverage
        evalStats.BLEU = maxBLEU
        evalStats.ROUGE = maxROUGE
        evalStats.COVERAGE = maxCoverage
        evalStats.loss = 1.0 - ((maxBLEU + maxROUGE + maxCoverage) / 3.0)
        return evalStats

    def rouge_n(self, x, y, n):
        ngramsX = set()
        for N in range(1, n + 1):
            ngramsX.update(ngrams(x, N, pad_left=False, pad_right=False))

        ngramsY = set()
        for N in range(1, n + 1):
            ngramsY.update(ngrams(y, N, pad_left=False, pad_right=False))

        common = set(ngramsX & ngramsY)
        return len(common) / len(ngramsX)

    def get_word_sequence_optimal_score(self, seq, ref_ngram_list):
        orig_hits = 0
        cand_ngram_list = []
        cand_ngram_list.extend(ngrams(seq, 4, pad_left=True, pad_right=True))
        cand_ngram_list.extend(ngrams(seq, 3, pad_left=True, pad_right=True))
        cand_ngram_list.extend(ngrams(seq, 2, pad_left=True, pad_right=True))
        cand_ngram_list.extend(seq)

        totalCand = len(cand_ngram_list)
        totalRef = len(ref_ngram_list)
        if totalCand == 0 or totalRef == 0:
            return 0
        ref_ngram_list_copy = ref_ngram_list[:]
        for ngram in cand_ngram_list:
            if ngram in ref_ngram_list_copy:
                orig_hits += 1
                ref_ngram_list_copy.remove(ngram)
        precision = orig_hits / totalCand
        recall = orig_hits / totalRef
        return (precision + recall) / 2

    # it must return an evalStats object with a loss
    def evaluateAgainst(self, parser, predicted, full_eval=False):
        evalStats = NLGEvalStats()

        maxBLEU = 0.0
        maxBLEUSmooth = 0.0
        maxCoverage = 0.0
        maxPrecision = 0.0
        maxROUGE = 0.0
        weights = (0.25, 0.25, 0.25, 0.25)
        if predicted:
            ref_split = []
            for ref in self.evaluationReferences:
                ref_split.append(ref.split(" "))
                if full_eval:
                    if ref not in parser.ngram_lists_per_relexed_word_sequence:
                        parser.ngram_lists_per_relexed_word_sequence[ref] = parser.get_ngram_list(ref_split)

                    ref_ngram_list = parser.ngram_lists_per_relexed_word_sequence[ref]

                    precision = self.get_word_sequence_optimal_score(predicted, ref_ngram_list)

                    if precision > maxPrecision:
                        maxPrecision = precision
            for ref in ref_split:
                if len(predicted) >= 4:
                    bleuOriginal = sentence_bleu([ref], predicted)
                    if bleuOriginal > maxBLEU:
                        maxBLEU = bleuOriginal
                elif len(predicted) == 3:
                    bleuOriginal = sentence_bleu([ref], predicted, (0.33, 0.33, 0.33, 0.0))
                    if bleuOriginal > maxBLEU:
                        maxBLEU = bleuOriginal
                elif len(predicted) == 2:
                    bleuOriginal = sentence_bleu([ref], predicted, (0.5, 0.5, 0.0, 0.0))
                    if bleuOriginal > maxBLEU:
                        maxBLEU = bleuOriginal
                elif len(predicted) == 1:
                    bleuOriginal = sentence_bleu([ref], predicted, (1.0, 0.0, 0.0, 0.0))
                    if bleuOriginal > maxBLEU:
                        maxBLEU = bleuOriginal

                if full_eval:
                    bleuSmooth = sentence_bleu([ref], predicted, weights,
                                                 smoothing_function=self.chencherry.method2)
                    if bleuSmooth > maxBLEUSmooth:
                        maxBLEUSmooth = bleuSmooth
            if full_eval:
                evalStats.BLEU_allRefs = sentence_bleu(ref_split, predicted)
                evalStats.BLEUSmooth_allRefs = sentence_bleu(ref_split, predicted, weights, smoothing_function=self.chencherry.method2)
            for ref in ref_split:
                rougeOriginal = self.rouge_n(ref, predicted, 4)
                if rougeOriginal > maxROUGE:
                    maxROUGE = rougeOriginal

                coverage = 0
                predictedStr = " ".join(predicted)
                for value in self.values:
                    f = False
                    if value in predictedStr:
                        coverage += 1
                        f = True
                    if not f and value.endswith('_no'):
                        if value[:-3] in predictedStr:
                            coverage += 1
                            f = True
                    if not f and value.endswith('_yes'):
                        if value[:-4] in predictedStr:
                            coverage += 1
                            f = True
                    if not f and 'familyfriendly' in value:
                        if covers_attr_E2E('familyfriendly', predictedStr):
                            coverage += 1
                            f = True
                    if not f and 'customer_rating' in value:
                        if covers_attr_E2E('customer_rating', predictedStr):
                            coverage += 1
                            f = True
                    if not f and 'pricerange' in value:
                        if covers_attr_E2E('pricerange', predictedStr):
                            coverage += 1
                            f = True

                coverage /= len(self.values)

                if coverage > maxCoverage:
                    maxCoverage = coverage
        evalStats.BLEU = maxBLEU
        evalStats.BLEUSmooth = maxBLEUSmooth
        evalStats.ROUGE = maxROUGE
        evalStats.COVERAGE = maxCoverage
        evalStats.precision = maxPrecision
        return evalStats

# Then the NER eval stats
class NLGEvalStats:
    def __init__(self):
        super().__init__()
        self.BLEU = 0
        self.BLEUSmooth = 0
        self.BLEU_allRefs = 0
        self.BLEUSmooth_allRefs = 0
        self.ROUGE = 0
        self.COVERAGE = 0
        self.precision = 0.0
