from structuredPredictionNLG.Action import Action

phrases = {}
phrases['area'] = [" river ", " city centre ", " centre of the city ", "center of the city", " city center ", "center of town", " riverside ", " city area ", " city "]
phrases['food'] = [" british ", " sushi and fish ", " sushi ", " wine and cheese ", " wines and cheese ", " fast - food "]
phrases['name'] = []
phrases['near'] = []
phrases['eattype'] = [' establishment ', ' restaurant ', ' coffee ship ', ' eating place ', ' café ', ' bar ', ' coffee shop ', ' coffee chop ', ' coffee house ', ' coffee shops ', ' caféteria ', ' public house ', ' pup ', ' coffee ', ' pub ', ' place ', ' shop ', ' house ']
phrases['pricerange'] = [" budget ", " fairly ", " regularly ", " l30 ", " l20 ", " l25 ", " l20 - l25 ", " l20 - 25 ", " l 30 ", " l 20 ", " l 25 ", " l 20 - l 25 ", " l 20 - 25 ", " lb30 ", " lb20 ", " lb25 ", " lb20 - lb25 ", " lb 30 ", " lb 20 ", " lb 25 ", " lb 20 - lb 25 ", " lb 20 - 25 ", " high end ", " 30l ", " 20l ", " 25l ", " 20l - 25l ", " 20 - 25l ", " 30 l ", " 20 l ", " 25 l ", " 20 l - 25 l", " 20 - 25 l", " 30lb ", " 20lb ", " 25lb ", " 20lb - 25lb ", " 20 - 25lb ", " 30 lb ", " 20 lb ", " 25 lb ", " 20 lb - 25 lb ", " 20 - 25 lb ", " frugally ", " well priced ", " reasonable ", " reasonably ", " under 30 pounds ", " under 30 pound ", " under 30 euros ", " under 30 euro ", " under 30 ", " under 25 pounds ", " under 25 pound ", " under 25 euros ", " under 25 euro ", " under 25 ", " under 20 pounds ", " under 20 pound ", " under 20 euros ", " under 20 euro ", " under 20 ", " under £30 ", " under £25 ", " under £20 ", " under £ 30 ", " under £ 25 ", " under £ 20 ", " twenty to twenty five pounds ", "fair", "£0", "£ 0", " twenty to twenty five pound ", " reduced ", " over 30 pounds ", " over 30 pound ", " over 30 euros ", " over 30 euro ", " over 30 ", " over 25 pounds ", " over 25 pound ", " over 25 euros ", " over 25 euro ", " over 25 ", " over 20 pounds ", " over 20 pound ", " over 20 euros ", " over 20 euro ", " over 20 ", " over £30 ", " over £25 ", " over £20 ", " over £ 30 ", " over £ 25 ", " over £ 20 ", " more than twenty pounds ", " more than twenty pound ", " more than twenty euros ", " more than twenty euro ", " more than twenty ", " more than thirty pounds ", " more than thirty pounds ", " more than thirty pound ", " more than thirty pound ", " more than thirty euros ", " more than thirty euros ", " more than thirty euro ", " more than thirty euro ", " more than thirty ", " more than thirty ", " more than 30 pounds ", " more than 30 pound ", " more than 30 euros ", " more than 30 euro ", " more than 30 ", " more than 25 pounds ", " more than 25 pound ", " more than 25 euros ", " more than 25 euro ", " more than 25 ", " more than 20 pounds ", " more than 20 pound ", " more than 20 euros ", " more than 20 euro ", " more than 20 ", " more than £30 ", " more than £25 ", " more than £20 ", " more than £ 30 ", " more than £ 25 ", " more than £ 20 ", " low cost ", " less than twenty pounds ", " less than twenty pound ", " less than twenty euros ", " less than twenty euro ", " less than twenty ", " less than thirty pounds ", " less than thirty pounds ", " less than thirty pound ", " less than thirty pound ", " less than thirty euros ", " less than thirty euros ", " less than thirty euro ", " less than thirty euro ", " less than thirty ", " less than thirty ", " less than 30 pounds ", " less than 30 pound ", " less than 30 euros ", " less than 30 euro ", " less than 30 ", " less than 25 pounds ", " less than 25 pound ", " less than 25 euros ", " less than 25 euro ", " less than 25 ", " less than 20 pounds ", " less than 20 pound ", " less than 20 euros ", " less than 20 euro ", " less than 20 ", " less than £30 ", " less than £25 ", " less than £20 ", " less than £ 30 ", " less than £ 25 ", " less than £ 20 ", " inexpensive ", " higher ", " expensive ", " cheap ", " cheap ", " between twenty to twenty five pounds ", " between twenty to twenty five pound ", " between twenty to twenty five euros ", " between twenty to twenty five euro ", " between 20 and 25 pounds ", " between 20 and 25 pound ", " between 20 and 25 euros ", " between 20 and 25 euro ", " between 20 and 25 ", " affordable ", " 30 ", " 25 ", " 20 - 25 ", " £30 ", " £25 ", " £20 - £25 ", " £20 - 25 ", " £20 ", " £ 30 ", " £ 25 ", " £ 20 - £ 25 ", " £ 20 - 25 ", " £ 20 ", " 30£ ", " 25£ ", " 20£ - 25£ ", " 20 - 25£ ", " 20£ ", " 30 £ ", " 25 £ ", " 20 £ - 25 £ ", " 20 - 25 £ ", " 20 £ ", " mid ", " 20 "]
phrases['customer_rating'] = [" 5 out of 5 ", " 3 out of 5 ", " 1 out of 5 ", " 5 out of 5 star ", " 3 out of 5 star ", " 1 out of 5 star ", " 5 out of 5 stars ", " 3 out of 5 stars ", " 1 out of 5 stars ", " poor ", " well ", " five out of five star ", " three out of five star ", " one out of five star ", " five out of five stars ", " three out of five stars ", " one out of five stars ", " five out of five ", " three out of fivestr ", " one out of five ", " 5 star ", " 3 star ", " 1 star  ", " 5 stars ", " 3 stars ", " 1 stars  ", " 5 - star ", " 3 - star ", " 1 - star  ", " 5 - stars ", " 3 - stars ", " 1 - stars  ", " five star ", " three star ", " one star ", " five stars ", " three stars ", " one stars ", " 5 ", " 3 ", " 1 ", " 5 ", " 3 ", " 1 "]
phrases['familyfriendly'] = [" adult ", " not family - friendly ", " not child - friendly ", " not children - friendly ", " not kids - friendly ", " not kid - friendly ", " not a family - friendly ", " not a child - friendly ", " not a children - friendly ", " not a kids - friendly ", " not a kid - friendly ", " family - friendly ", " child - friendly ", " children - friendly ", " kids - friendly ", " kid - friendly ", " a family - friendly ", " a child - friendly ", " a children - friendly ", " a kids - friendly ", " a kid - friendly ", " non family - friendly ", " non child - friendly ", " non children - friendly ", " non kids - friendly ", " non kid - friendly ", " not family friendly ", " not child friendly ", " not children friendly ", " not kids friendly ", " not kid friendly ", " not a family friendly ", " not a child friendly ", " not a children friendly ", " not a kids friendly ", " not a kid friendly ", " family friendly ", " child friendly ", " children friendly ", " kids friendly ", " kid friendly ", " a family friendly ", " a child friendly ", " a children friendly ", " a kids friendly ", " a kid friendly ", " non family friendly ", " non child friendly ", " non children friendly ", " non kids friendly ", " non kid friendly ", " not family ", " not child ", " not children ", " not kids ", " not kid ", " not a family ", " not a child ", " not a children ", " not a kids ", " not a kid ", " family ", " child ", " children ", " kids ", " kid ", " a family ", " a child ", " a children ", " a kids ", " a kid ", " non family ", " non child ", " non children ", " non kids ", " non kid ", " non family ", " non child ", " non children ", " non kids ", " non kid "]

for attr in phrases:
    phrases[attr] = [o for o in reversed(sorted(phrases[attr], key=len))]

def full_delexicalize_E2E(delexicalizedMap, attribute, v, attributeValues, refPart):
    for phrase in phrases[attribute]:
        if phrase in " " + refPart + " ":
            return phrase.strip()
            
    if attribute == 'name':
        if " " + v + "s " in " " + refPart + " ":
            return v + "s"
            
        elif v == 'cotto' and ' cotton ' in " " + refPart + " ":
            return 'cotton'
            
        elif v == 'fitzbillies' and ' fitzbilies ' in " " + refPart + " ":
            return 'fitzbilies'
            
        elif v == 'fitzbillies' and ' fitzbilies ' in " " + refPart + " ":
            return 'fitzbilies'
            
    elif attribute == 'near':
        if v == 'crowne plaza hotel' and ' crown plaza hotel ' in " " + refPart + " ":
            return 'crown plaza hotel'
            
        elif v == 'crowne plaza hotel' and ' crown plaza  ' in " " + refPart + " ":
            return 'crown plaza'
            
        elif v == 'raja indian cuisine' and ' raja ' in " " + refPart + " ":
            return 'raja'
            
        elif v == 'the portland arms' and ' portland arms ' in " " + refPart + " ":
            return 'portland arms'
            
        elif v == 'café brazil' and ' café brazilian ' in " " + refPart + " ":
            return 'café brazilian'
            
        elif v == 'yippee noodle bar' and ' yippe noodle bar ' in " " + refPart + " ":
            return 'yippe noodle bar'
            
        elif v == 'express by holiday inn' and ' holiday inns ' in " " + refPart + " ":
            return 'holiday inns'
            
        elif v == 'café sicilia' and ' café sicilian ' in " " + refPart + " ":
            return 'café sicilian'
            
        elif v == 'café rouge' and ' café rough ' in " " + refPart + " ":
            return 'café rough'
            
        elif v == 'burger king' and ' burger kings ' in " " + refPart + " ":
            return 'burger kings'
            
    elif attribute == 'food':
        if v == 'english' and ' pub food ' in " " + refPart + " ":
            return "pub"
    elif attribute == 'pricerange' and ('customer_rating' not in attributeValues or Action.TOKEN_X + 'customer_rating_0' in delexicalizedMap):
        for phrase in [" good ", " bad ", " better ", " worse ", " highest ", " higher ", " lowest ", " lower ", " great ", " high ", " low ", " moderate ", " moderates ", " average ", " averaged ", " medium "]:
            if phrase in " " + refPart + " ":
                return phrase.strip()
                
    elif attribute == 'customer_rating' and ('pricerange' not in attributeValues or Action.TOKEN_X + 'pricerange_0' in delexicalizedMap):
        for phrase in [" good ", " bad ", " better ", " worse ", " highest ", " higher ", " lowest ", " lower ", " great ", " high ", " low ", " moderate ", " moderates ", " average ", " averaged ", " medium "]:
            if phrase in " " + refPart + " ":
                return phrase.strip()

    return False


def covers_attr_E2E(attribute, predicted):
    for phrase in phrases[attribute]:
        if phrase in " " + predicted + " ":
            return True
    '''
    if attribute == 'name':
        if " " + v + "s " in " " + predicted + " ":
            return True

        elif v == 'cotto' and ' cotton ' in " " + predicted + " ":
            return True

        elif v == 'fitzbillies' and ' fitzbilies ' in " " + predicted + " ":
            return True

        elif v == 'fitzbillies' and ' fitzbilies ' in " " + predicted + " ":
            return True

    elif attribute == 'near':
        if v == 'crowne plaza hotel' and ' crown plaza hotel ' in " " + predicted + " ":
            return True

        elif v == 'crowne plaza hotel' and ' crown plaza  ' in " " + predicted + " ":
            return True

        elif v == 'raja indian cuisine' and ' raja ' in " " + predicted + " ":
            return True

        elif v == 'the portland arms' and ' portland arms ' in " " + predicted + " ":
            return True

        elif v == 'café brazil' and ' café brazilian ' in " " + predicted + " ":
            return True

        elif v == 'yippee noodle bar' and ' yippe noodle bar ' in " " + predicted + " ":
            return True

        elif v == 'express by holiday inn' and ' holiday inns ' in " " + predicted + " ":
            return True

        elif v == 'café sicilia' and ' café sicilian ' in " " + predicted + " ":
            return True

        elif v == 'café rouge' and ' café rough ' in " " + predicted + " ":
            return True

        elif v == 'burger king' and ' burger kings ' in " " + predicted + " ":
            return True

    elif attribute == 'food':
        if v == 'english' and ' pub food ' in " " + predicted + " ":
            return True
    
    elif attribute == 'pricerange' and (
            'customer_rating' not in attributeValues or Action.TOKEN_X + 'customer_rating_0' in delexicalizedMap):
        for phrase in [" good ", " bad ", " better ", " worse ", " highest ", " higher ", " lowest ", " lower ",
                       " great ", " high ", " low ", " moderate ", " moderates ", " average ", " averaged ",
                       " medium "]:
            if phrase in " " + predicted + " ":
                return True

    elif attribute == 'customer_rating' and (
            'pricerange' not in attributeValues or Action.TOKEN_X + 'pricerange_0' in delexicalizedMap):
        for phrase in [" good ", " bad ", " better ", " worse ", " highest ", " higher ", " lowest ", " lower ",
                       " great ", " high ", " low ", " moderate ", " moderates ", " average ", " averaged ",
                       " medium "]:
            if phrase in " " + predicted + " ":
                return True
    '''

    return False

