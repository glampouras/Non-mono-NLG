# Gerasimos Lampouras, 2017:
from Action import Action
from collections import defaultdict
from imitation import StructuredInput

'''
 Internal representation of a Meaning Representation
'''
class MeaningRepresentation(StructuredInput):

    def __init__(self, predicate, attributeValues, MRstr, delexicalizationMap=False):
        # A MeaningRepresentation consists of a predicate, and a map of attributes to sets of values
        self.predicate = predicate
        self.attributeValues = attributeValues
        self.attributeSubjects = False
        # This variable stores the string describing the meaning representation in the dataset.
        # Used mostly for tracking "unique" instances of an MR in the dataset, when lexicalization and
        # attribute order are considered.
        self.MRstr = MRstr
        # A string representation of the MeaningRepresantation, used primarily to compare MeaningRepresantation objects.
        # We store the value, so we do not have to reconstruct it.
        self.abstractMR = str()

        # This variable maps the variable values (e.g. @x@attr), to the corresponding lexicalized string values.
        # It is populated during the initial delexicalization of the MR, and used after generation for post-processing
        # re-lexicalization of the variables.
        if delexicalizationMap:
            self.delexicalizationMap = delexicalizationMap
        else:
            self.delexicalizationMap = defaultdict()

    '''
     A string representation of the MeaningRepresantation, used primarily to compare MeaningRepresantation objects.
     We store the value, so we do not have to reconstruct it.
    '''
    def getAbstractMR(self, recalculate=False):
        if not self.abstractMR or recalculate:
            self.abstractMR = self.predicate + ":"

            sortedAttrs = sorted(self.attributeValues.keys())
            xCounts = defaultdict()
            for attr in sortedAttrs:
                xCounts[attr] = 0
                self.abstractMR += attr + "={"

                value = self.attributeValues.get(attr)
                if attr == "name" or attr == "near":
                    self.abstractMR += Action.TOKEN_X + attr + "_" + str(xCounts[attr])
                    xCounts[attr] = xCounts[attr] + 1
                else:
                    self.abstractMR += value
                self.abstractMR += "},"
        return self.abstractMR

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.predicate == other.predicate and self.attributeValues.__eq__(other.attributeValues)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.predicate) + hash(self.attributeValues)

