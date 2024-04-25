""" 
Embedding words for use by the spymaster bot.

Uses OpenAI's text-embedding-3-small model to embed words.
"""

from openai import OpenAI
import numpy as np
import pickle
import pprint
import tqdm
import pprint

import dotenv
import os
import gensim.downloader
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WORDS_FILE_PATH = "./data/words/official_10000_english_words.txt"

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

MAX_MEANINGS = 2
MAX_DEFINITIONS_PER_MEANING = 5

MAX_MAJOR_DEFS = 3
MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF = 2
MAX_MINOR_DEFS_PER_PART_OF_SPEECH = 1
MAX_DEFINITIONS = MAX_MAJOR_DEFS * MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF * MAX_MINOR_DEFS_PER_PART_OF_SPEECH


def get_definitions(word):
    """
    Get the definitions of a word from the dictionaryapi.dev API.

    Args:
        word (string) - the word to get the definitions of

    Returns:
        definitions (list) - list of definitions of the word
    """
    import requests
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    # url = f"https://dictionaryapi.com/api/v3/references/sd3/json/{word}?key=0d985e9f-44be-4b21-aab4-fbcd68a58ad6"
    # url = f"https://dictionaryapi.com/api/v3/references/sd4/json/{word}?key=0caae7d2-19ff-4764-aed9-c65c1bbe6cdf"
    response = requests.get(url)
    # print(response)
    try:
        definitions = []
        major_defs = response.json()
        # print(major_defs)
        # num_major_defs = len(major_defs)
        for major_def in major_defs[:MAX_MAJOR_DEFS]:
            parts_of_speech = major_def["meanings"]
            # num_parts_of_speech = len(parts_of_speech)
            for pos in parts_of_speech[:MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF]:
                minor_defs = pos["definitions"]
                # num_minor_defs = len(minor_defs)
                for minor_def in minor_defs[:MAX_MINOR_DEFS_PER_PART_OF_SPEECH]:
                    # example = minor_def["example"] if "example" in minor_def else ""
                    # definition = minor_def["definition"] + ((" " + example) if example else "")
                    definition = minor_def["definition"]
                    definitions.append(definition)
                
        
        major_def = major_defs[0]
        parts_of_speech = major_def["meanings"]
        while len(definitions) < MAX_DEFINITIONS and parts_of_speech:
            # greedily add definitions from the first major definition
            pos = parts_of_speech[0]
            parts_of_speech = parts_of_speech[1:]
            minor_defs = pos["definitions"]
            while len(definitions) < MAX_DEFINITIONS and minor_defs:
                definition = minor_defs[0]["definition"]
                minor_defs = minor_defs[1:]
                if definition not in definitions:
                    definitions.append(definition)
                
        # pprint.pp(definitions)
        return definitions
                
        # meanings = data[0]["meanings"]
        # # if len(data) > 1:
        # #     print(f"Multiple definitions for {word}")
        # all_definitions = []
        # # for meaning in meanings[:MAX_MEANINGS]:
        # for meaning in meanings:
        #     curr_definitions = meaning["definitions"]
        #     # for definition in curr_definitions[:MAX_DEFINITIONS_PER_MEANING]:
        #     for definition in curr_definitions:
        #         try:
        #             example = definition["example"] if "example" in definition else ""
        #             definition_text = definition["definition"] + ((" " + example) if example else "")
        #             all_definitions.append(definition_text)
        #         except:
        #             continue
                    
        # return all_definitions
        # pprint.pp(response.json())
    except Exception as e:
        # print(f"Error getting definitions for {word}: {e}")
        return None



def embed_words(words, model_name):
    """
    Embeds a list of words into numpy vectors.

    Args:
        words (list) - list of words to embed
        pickle_path (string) - Stores the results in pickle_path

    Returns:
        embeddings (dict) - dictionary mapping each word to its vector embedding
    """
    batch_size = 1000
    word_batches = []
    for i in range(0, len(words), batch_size):
        word_batches.append(words[i:i+batch_size])

    embeddings = {}
    
    
    if model_name == "openai":
        for batch in word_batches:
            response = client.embeddings.create(model="text-embedding-3-small", input=batch)
            response = response.data

            for i, word in enumerate(batch):
                embeddings[word] = np.array(response[i].embedding)
            print(f"Finished embedding {len(embeddings)} words")
    elif model_name == "glove300" or model_name == "word2vec300" or \
         model_name == "glove100" or model_name == "glovetwitter200" or model_name == "fasttext":
        if model_name == "glove300":
            model = gensim.downloader.load('glove-wiki-gigaword-300')
        elif model_name == "word2vec300":
            model = gensim.downloader.load('word2vec-google-news-300')
        elif model_name == "glove100":
            model = gensim.downloader.load('glove-wiki-gigaword-100')
        elif model_name == "glovetwitter200":
            model = gensim.downloader.load('glove-twitter-200')
        elif model_name == "fasttext":
            model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
        
        for batch in word_batches:
            for i, word in enumerate(batch):
                if word in model:
                    embeddings[word] = np.array(model[word])
            print(f"Finished embedding {len(embeddings)} words")
    elif model_name == "word2vec+glove300":
        model1 = gensim.downloader.load('word2vec-google-news-300')
        model2 = gensim.downloader.load('glove-wiki-gigaword-300')
        for batch in word_batches:
            for i, word in enumerate(batch):
                if word in model1 and word in model2:
                    vec1 = np.array(model1[word])
                    vec2 = np.array(model2[word])
                    embeddings[word] = np.concatenate((vec1, vec2))
            print(f"Finished embedding {len(embeddings)} words")

    pickle_path = f"./data/embeddings/{model_name}_word_embeddings.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

def load_embeddings(model_name):
    pickle_path = f"./data/embeddings/{model_name}_word_embeddings.pkl"
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def process_all_words():
    with open(WORDS_FILE_PATH, "r") as f:
        words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].strip().lower()
    return words

def embed_definitions(words):
    import time
    """
    Embeds a list of words into numpy vectors.

    Args:
        words (list) - list of words to embed
        pickle_path (string) - Stores the results in pickle_path

    Returns:
        embeddings (dict) - dictionary mapping each word to its vector embedding
    """
    print(f"Embedding definitions for {len(words)} words")

    word_to_def_embeddings = {}
    
    counts = {}

    try: 
        pickle_path = f"./data/embeddings/word_definitions.pkl"
        # read from pickle file if it exists
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                word_to_def_embeddings = pickle.load(f)
                num_words = len(word_to_def_embeddings)
                print(f"Loaded {num_words} word's embeddings from {pickle_path}")

        batch_size = 1000
        word_batches = []
        for i in range(0, len(words[num_words:]), batch_size):
            word_batches.append(words[num_words:][i:i+batch_size])

        for batch in tqdm.tqdm(word_batches):
            for word in tqdm.tqdm(batch):
                definitions = get_definitions(word)
                if definitions:
                    final_definitions = []
                    definition_embeddings = []
                    for definition in definitions:
                        try:
                            response = client.embeddings.create(model="text-embedding-3-small", input=[definition])
                            response = response.data[0].embedding
                            definition_embeddings.append(np.array(response))
                            final_definitions.append(definition)
                        except:
                            continue
                    definition_embeddings = np.array(definition_embeddings)
                    word_to_def_embeddings[word] = { "definitions": final_definitions, "embeddings": definition_embeddings }
                else:
                    word_to_def_embeddings[word] = { "definitions": [], "embeddings": [] }
                num_definitions = len(definitions) if definitions else 0
                counts[num_definitions] = counts.get(num_definitions, 0) + 1
                time.sleep(1)
            print(f"Finished embedding {len(word_to_def_embeddings)} words")
            print(f"Counts: {counts}")
            with open(pickle_path, "wb") as f:
                pickle.dump(word_to_def_embeddings, f)
                
    except KeyboardInterrupt:
        pass
    
    except Exception as _:
        pass

    sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    print("\n".join([f"{k}: {v}" for k, v in sorted_counts]))
    
    return word_to_def_embeddings


def fix_words():
    potentially_incorrect_words = [["africa", "africa"],
["america", "america"],
["antarctica", "antarctica"],
["atlantis", "atlantis"],
["australia", "australia"],
["aztec", "aztec"],
["beijing", "beijing"],
["berlin", "berlin"],
["bermuda", "bermuda"],
["canada", "canada"],
["cloak", "cloak"],
["czech", "czech"],
["egypt", "egypt"],
["england", "england"],
["europe", "europe"],
["france", "france"],
["germany", "germany"],
["greece", "greece"],
["himalayas", "himalayas"],
["hollywood", "hollywood"],
["india", "india"],
["jupiter", "jupiter"],
["kangaroo", "kangaroo"],
["london", "london"],
["mercury", "mercury"],
["mexico", "mexico"],
["moscow", "moscow"],
["net", "net"],
["olympus", "olympus"],
["rome", "rome"],
["saturn", "saturn"],
["shakespeare", "shakespeare"],
["tokyo", "tokyo"],
["unicorn", "unicorn"],
["washington", "washington"],
["for", "for"],
["do", "do"],
["be", "be"],
["your", "your"],
["was", "was"],
["our", "our"],
["their", "their"],
["every", "every"],
["hell", "hell"],
["old", "old"],
["woman", "woman"],
["hello", "hello"],
["look", "look"],
["stupid", "stupid"],
["christmas", "christmas"],
["touch", "touch"],
["jesus", "jesus"],
["double", "double"],
["saturday", "saturday"],
["is", "is"],
["taste", "taste"],
["tight", "tight"],
["friday", "friday"],
["chinese", "chinese"],
["ashame", "ashame"],
["soup", "soup"],
["sunday", "sunday"],
["monday", "monday"],
["thursday", "thursday"],
["tuesday", "tuesday"],
["o’clock", "o’clock"],
["carter", "carter"],
["suicide", "suicide"],
["italian", "italian"],
["eighty", "eighty"],
["causless", "causeless="],
["wednesday", "wednesday"],
["november", "november"],
["september", "september"],
["october", "october"],
["behalf", "behalf"],
["phase", "phase"],
["restrict", "restrict"],
["e-book", "e-book"],
["japanese", "japanese"],
["russian", "russian"],
["british", "british"],
["italy", "italy"],
["canadian", "canadian"],
["jewish", "jewish"],
["june", "june"],
["july", "july"],
["rape", "rape"],
["distrustul", "distrustful="],
["turk", "turk"],
["april", "april"],
["christian", "christian"],
["ignate", "ignite="],
["january", "january"],
["december", "december"],
["gemini", "gemini"],
["etc.", "etc."],
["february", "february"],
["fibber", "fibber"],
["olympics", "olympics"],
["dining-room", "dining-room"],
["reside", "reside"],
["allright", "allright"],
["dose", "dose"],
["tow", "tow"],
["vanilla", "vanilla"],
["diabete", "diabete"],
["calory", "calorie="],
["cue", "cue"],
["accustom", "accustom"],
["upbrining", "upbringing="],
["tea-pot", "tea-pot"],
["test-tube", "test-tube"],
["text-message", "text-message"],
["tyran", "tyran"],
["arab", "arab"],
["berch", "berch"],
["canvass", "canvass"],
["cartrige", "cartridge="],
["cod-liver", "cod-liver"],
["consession", "concession="],
["dane", "dane"],
["faciliate", "facilitate="],
["imprudence", "imprudence"],
["irriplaceable", "irreplaceable="],
["portuguese", "portuguese"],
["selfconfidence", "self confidence!"],
["selfcontrol", "selfcontrol"],
["selfemployed", "self employed!"],
["selfevident", "selfevident"],
["selfportrait", "self portrait!"],
["spaniard", "spaniard"],
["cupid", "cupid"],
["fewer", "fewer"],
["zomby", "zomby"],
["capricorn", "capricorn"],
["sideline", "sideline"],
["rhinoceros", "rhinoceros"],
["railing", "railing"],
["puberty", "puberty"],
["phantom", "phantom"],
["pesky", "pesky"],
["outrage", "outrage"],
["openly", "openly"],
["nominate", "nominate"],
["itching", "itching"],
["intuition", "intuition"],
["icky", "icky"],
["humility", "humility"],
["fishy", "fishy"],
["finch", "finch"],
["ferry", "ferry"],
["excessive", "excessive"],
["evolve", "evolve"],
["eligible", "eligible"],
["dosage", "dosage"],
["disrupt", "disrupt"],
["derange", "derange"],
["cuckoo", "cuckoo"],
["craziness", "craziness"],
["compatible", "compatible"],
["circumstantial", "circumstantial"],
["bunker", "bunker"],
["asthma", "asthma"],
["arise", "arise"],
["triad", "triad"],
["trashy", "trashy"],
["tilt", "tilt"],
["thorn", "thorn"],
["suburban", "suburban"],
["soothing", "soothing"],
["slumber", "slumber"],
["slayer", "slayer"],
["shindig", "shindig"],
["sentiment", "sentiment"],
["riddance", "riddance"],
["purity", "purity"],
["offshore", "offshore"],
["massacre", "massacre"],
["isolation", "isolation"],
["impersonal", "impersonal"],
["hoop", "hoop"],
["hamlet", "hamlet"],
["footprint", "footprint"],
["fluke", "fluke"],
["festivity", "festivity"],
["feisty", "feisty"],
["evacuate", "evacuate"],
["detain", "detain"],
["defender", "defender"],
["creeping", "creeping"],
["classics", "classics"],
["bud", "bud"],
["bitterness", "bitterness"],
["baloney", "baloney"],
["ashtray", "ashtray"],
["apocalypse", "apocalypse"],
["zillion", "zillion"],
["wan", "wan"],
["viable", "viable"],
["sympathize", "sympathize"],
["squirt", "squirt"],
["spade", "spade"],
["snappy", "snappy"],
["sleepover", "sleepover"],
["reunited", "reunited"],
["retainer", "retainer"],
["restroom", "restroom"],
["repercussion", "repercussion"],
["relive", "relive"],
["reef", "reef"],
["reconciliation", "reconciliation"],
["reconcile", "reconcile"],
["recognise", "recognise"],
["raven", "raven"],
["prevail", "prevail"],
["preach", "preach"],
["omen", "omen"],
["noose", "noose"],
["meatloaf", "meatloaf"],
["manicure", "manicure"],
["landlady", "landlady"],
["hypothesis", "hypothesis"],
["homesick", "homesick"],
["hive", "hive"],
["hectic", "hectic"],
["heartbreak", "heartbreak"],
["haunting", "haunting"],
["gall", "gall"],
["frown", "frown"],
["fingerprint", "fingerprint"],
["extract", "extract"],
["expire", "expire"],
["exhausting", "exhausting"],
["exceptional", "exceptional"],
["disregard", "disregard"],
["crisp", "crisp"],
["cooperative", "cooperative"],
["cling", "cling"],
["blender", "blender"],
["bitty", "bitty"],
["bead", "bead"],
["badger", "badger"],
["arch", "arch"],
["anticipation", "anticipation"],
["advocate", "advocate"],
["waterfront", "waterfront"],
["upstanding", "upstanding"],
["unprofessional", "unprofessional"],
["unhealthy", "unhealthy"],
["turmoil", "turmoil"],
["truthful", "truthful"],
["toothpaste", "toothpaste"],
["thoughtless", "thoughtless"],
["spun", "spun"],
["spook", "spook"],
["shady", "shady"],
["senseless", "senseless"],
["scooter", "scooter"],
["ringer", "ringer"],
["refuge", "refuge"],
["pun", "pun"],
["preposterous", "preposterous"],
["portable", "portable"],
["pastry", "pastry"],
["loner", "loner"],
["jogging", "jogging"],
["itchy", "itchy"],
["insinuate", "insinuate"],
["hospitality", "hospitality"],
["forthcoming", "forthcoming"],
["excess", "excess"],
["etiquette", "etiquette"],
["ending", "ending"],
["edit", "edit"],
["dunk", "dunk"],
["distinction", "distinction"],
["desist", "desist"],
["deprive", "deprive"],
["casserole", "casserole"],
["beaver", "beaver"],
["bearer", "bearer"],
["applaud", "applaud"],
["appalling", "appalling"],
["trump", "trump"],
["trench", "trench"],
["touchdown", "touchdown"],
["thaw", "thaw"],
["tailor", "tailor"],
["swoop", "swoop"],
["sticker", "sticker"],
["smite", "smite"],
["shameless", "shameless"],
["researching", "researching"],
["reclaim", "reclaim"],
["purposely", "purposely"],
["proxy", "proxy"],
["pineapple", "pineapple"],
["parasite", "parasite"],
["offspring", "offspring"],
["multiply", "multiply"],
["masculine", "masculine"],
["hoax", "hoax"],
["gunfire", "gunfire"],
["furnace", "furnace"],
["flyer", "flier="],
["engrave", "engrave"],
["duplicate", "duplicate"],
["dome", "dome"],
["designate", "designate"],
["delicatessen", "delicatessen"],
["cryptic", "cryptic"],
["creed", "creed"],
["condemn", "condemn"],
["colossal", "colossal"],
["clipper", "clipper"],
["cliché", "cliché"],
["clarity", "clarity"],
["banish", "banish"],
["argon", "argon"],
["uncanny", "uncanny"],
["treasury", "treasury"],
["technicality", "technicality"],
["static", "static"],
["scales", "scales"],
["satin", "satin"],
["rosebud", "rosebud"],
["relentless", "relentless"],
["rearrange", "rearrange"],
["plunge", "plunge"],
["obsolete", "obsolete"],
["moth", "moth"],
["lullaby", "lullaby"],
["layout", "layout"],
["knob", "knob"],
["flashy", "flashy"],
["epic", "epic"],
["encode", "encode"],
["dread", "dread"],
["doodle", "doodle"],
["dealing", "dealing"],
["cupcake", "cupcake"],
["console", "console"],
["conclusive", "conclusive"],
["bowel", "bowel"],
["aura", "aura"],
["ape", "ape"],
["announcing", "announcing"],
["abide", "abide"],
["wonderfully", "wonderfully"],
["warfare", "warfare"],
["wad", "wad"],
["violate", "violate"],
["suicidal", "suicidal"],
["skilled", "skilled"],
["sketchy", "sketchy"],
["shoplifting", "shoplifting"],
["quizmaster", "quizmaster"],
["profitable", "profitable"],
["politically", "politically"],
["peck", "peck"],
["needless", "needless"],
["momentarily", "momentarily"],
["midget", "midget"],
["leukemia", "leukemia"],
["idol", "idol"],
["fungus", "fungus"],
["extraction", "extraction"],
["explorer", "explorer"],
["chimera", "chimera"],
["cheery", "cheery"],
["cadet", "cadet"],
["benign", "benign"],
["artillery", "artillery"],
["apiece", "apiece"],
["abduction", "abduction"],
["virtual", "virtual"],
["unspeakable", "unspeakable"],
["unlimited", "unlimited"],
["unidentified", "unidentified"],
["twinkle", "twinkle"],
["transcript", "transcript"],
["temp", "temp"],
["surfing", "surfing"],
["stricken", "stricken"],
["stork", "stork"],
["sodium", "sodium"],
["snick", "snick"],
["sly", "sly"],
["retrieval", "retrieval"],
["quickie", "quickie"],
["playboy", "playboy"],
["observer", "observer"],
["mope", "mope"],
["moan", "moan"],
["mausoleum", "mausoleum"],
["intensive", "intensive"],
["infest", "infest"],
["incompetence", "incompetence"],
["horrified", "horrified"],
["grunt", "grunt"],
["fracture", "fracture"],
["formerly", "formerly"],
["fireball", "fireball"],
["firearm", "firearm"],
["fend", "fend"],
["examiner", "examiner"],
["elite", "elite"],
["duke", "duke"],
["dashing", "dashing"],
["curry", "curry"],
["crumb", "crumb"],
["conclude", "conclude"],
["boulevard", "boulevard"],
["bookie", "bookie"],
["baptize", "baptize"],
["astronaut", "astronaut"],
["sideway", "sideway"],
["exterminator", "exterminator"],
["flap", "flap"],
["fahrenheit", "fahrenheit"],
["scallion", "scallion"],
["alleviate", "alleviate"],
["caprize", "caprize"],
["egyptian", "egyptian"],
["finnish", "finnish"],
["freckless", "freckles="],
["lime-tree", "lime-tree"],
["mitten", "mitten"],
["montley", "motley="],
["phonebooth", "phonebooth"],
["rascal", "rascal"],
["selfservice", "selfservice"],
["sift", "sift"],
["abberance", "aberrance="],
["abberant", "aberrant="],
["abjection", "abjection"],
["abolishment", "abolishment"],
["aboninable", "abominable="],
["abonnement", "abonnement"],
["above-average", "above-average"],
["abruptness", "abruptness"],
["acheless", "acheless"],
["amendable", "amenable="],
["anoitment", "anointment="],
["antiterrorist", "antiterrorist"],
["archly", "archly"],
["armless", "armless"],
["assuredness", "assuredness"],
["aweless", "aweless"],
["bluffy", "bluffy"],
["ceasless", "ceaseless="],
["chastisement", "chastisement"],
["consecution", "consecution"],
["crimpy", "crimpy"],
["deliquency", "delinquency="],
["dimness", "dimness"],
["dirtiness", "dirtiness"],
["dish-water", "dish-water"],
["dog-collar", "dog-collar"],
["doormant", "dormant="],
["earlap", "earlap"],
["edacity", "edacity"],
["efforless", "effortless="],
["equability", "equability"],
["excruciation", "excruciation"],
["fire-extinguisher", "fire-extinguisher"],
["fixedly", "fixedly"],
["forceless", "forceless"],
["foster-child", "foster-child"],
["frigility", "frigility"],
["gorge", "gorge"],
["hardily", "hardily"],
["hermic", "hermic!"],
["icily", "icily"],
["illegimate", "illegitimate="],
["inapptitude", "ineptitude="],
["infractructure", "infrastructure="],
["lanentable", "lanentable"],
["lardy", "lardy"],
["lavishness", "lavishness"],
["leaseholder", "leaseholder"],
["lackless", "lackless"],
["muteness", "muteness"],
["objectless", "objectless"],
["overlive", "overlive"],
["palmful", "palmful"],
["pilotage", "pilotage"],
["placidity", "placidity"],
["reseptive", "receptive="],
["resedue", "residue="],
["revert", "revert"]]
    # all_words = process_all_words()
    pickle_path = f"./data/embeddings/word_definitions.pkl"
    # read from pickle file if it exists
    word_to_def_embeddings = None
    with open(pickle_path, "rb") as f:
        word_to_def_embeddings = pickle.load(f)
        for word, correct_word in potentially_incorrect_words:
            if "=" in correct_word:
                correct_word = correct_word[:-1]
                if word in word_to_def_embeddings:
                    word_to_def_embeddings[correct_word] = word_to_def_embeddings[word]
                    del word_to_def_embeddings[word]
                
            if "!" in correct_word:
                correct_word = correct_word[:-1]
                if word in word_to_def_embeddings:
                    del word_to_def_embeddings[word]
            
    # store the embeddings back in the pickle file
    with open(pickle_path, "wb") as f:
        pickle.dump(word_to_def_embeddings, f)
        
    # with open(WORDS_FILE_PATH, "w") as f:
    #     for word in all_words:
    #         f.write(f"{word}\n")

if __name__ == '__main__':
    pass

    # words = process_all_words()
    # pickle_path = f"./data/embeddings/word_definitions.pkl"
    # read from pickle file if it exists
    # with open(pickle_path, "rb") as f:
    #     word_to_def_embeddings = pickle.load(f)
    #     for word in words:
    #         if word not in word_to_def_embeddings:
    #             print(word)
            
    #     count = 0
    #     for word, data in word_to_def_embeddings.items():
    #         if len(data["definitions"]) == 0:
    #             print(word)
    #             count += 1
    #             # pprint.pp(data)
    #     print(count)
        
    # fix_words()
        
    # models = ["openai", "word2vec300", "glove300", "word2vec+glove300", "glove100", "glovetwitter200", "fasttext"]
    # model = "fasttext"
    # embed_words(words, model)
    # print(get_definitions("africa"))
    # get_definitions("die")

