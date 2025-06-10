import sys
import os
import platform
import numpy as np
from operator import itemgetter
import glob
import math
from collections import Counter
import spacy

class TAACOProcessor:
    def __init__(self, varDict, gui=False, source_text=False):
        # Initialize spacy
        print("Loading Spacy")
        print("Loading Spacy Model")
        self.nlp = spacy.load("en_core_web_sm")
        
        # System detection
        if platform.system() == "Darwin":
            self.system = "M"
        elif platform.system() == "Windows":
            self.system = "W"
        elif platform.system() == "Linux":
            self.system = "L"
            
        # Store input parameters
        self.varDict = varDict
        self.gui = gui
        self.source_text = source_text
        
        # Convert varDict values to boolean if coming from GUI
        if self.varDict["wordsAll"] not in [True, False]:
            for items in self.varDict:
                if self.varDict[items].get() == 1:
                    self.varDict[items] = True
                else:
                    self.varDict[items] = False
        
        # Calculate box checks
        self.overlap_box = self.checkBoxes(self.varDict, ["wordsAll","wordsContent","wordsFunction","wordsNoun","wordsPronoun","wordsArgument","wordsVerb","wordsAdjective","wordsAdverb"])
        self.segment_box = self.checkBoxes(self.varDict, ["overlapSentence","overlapParagraph"])
        self.adjacent_box = self.checkBoxes(self.varDict, ["overlapAdjacent","overlapAdjacent2"])
        self.ttr_box = self.overlap_box + self.checkBoxes(self.varDict, ["otherTTR"])
        self.all_boxes = self.checkBoxes(self.varDict, list(self.varDict.keys()))
        
        # POS tag lists
        self.noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        self.proper_n = ["NNP", "NNPS"]
        self.no_proper = ["NN", "NNS"]
        self.pronouns = ["PRP", "PRP$"]
        self.adjectives = ["JJ", "JJR", "JJS"]
        self.verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
        self.adverbs = ["RB", "RBR", "RBS"]
        self.content = ["NN", "NNS", "NNP", "NNPS","JJ", "JJR", "JJS"]
        self.prelim_not_function = ["NN", "NNS", "NNP", "NNPS","JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
        
        # Sentence linkers and connectives
        self.sentence_linkers = "nonetheless therefore although furthermore whereas nevertheless whatever however besides henceforth then if while but until because alternatively meanwhile when notwithstanding whenever moreover as consequently".split(" ")
        self.sentence_linkers_caveat = "for since after yet so"
        
        self.order = "next	first	firstly	second	secondly	finally	then".split("\t")
        self.order_caveat = "before	after".split("\t")
        self.order_ngram = "to begin with	in conclusion	above all".split("\t")
        
        self.reason_and_purpose = "therefore hence because as consequently".split(" ")
        self.reason_and_purpose_caveat = ["since", "so"]
        self.reason_and_purpose_ngram = ["that is why", "for this reason", "for that reason", "because of", "on account of", "so that"]

        self.all_causal = "although	arise	arises	arising	arose	because	cause	caused	causes	causing	condition	conditions	consequence	consequences	consequent	consequently	due to	enable	enabled	enables	enabling	even then	follow that	follow the	follow this	followed that	followed the	followed this	following that	follows the	follows this	hence	made	make	makes	making	nevertheless	nonetheless	only if	provided that	result	results	so	therefore	though	thus	unless	whenever".split("\t")
        self.all_causal_caveat = ["since"]
        
        self.positive_causal = "arise	arises	arising	arose	because	cause	caused	causes	causing	condition	conditions	consequence	consequences	consequent	consequently	due	enable	enabled	enables	enabling	even	follow	followed	following	follows	hence	if	made	make	makes	making	only	provided	result	results	then	therefore	this	thus".split("\t")
        self.positive_causal_caveat = "since	so".split("\t")
        
        self.all_logical = "actually	admittedly	also	alternatively	although	anyhow	anyway	because	besides	but	cause	caused	causes	causing	consequence	consequences	consequently	correspondingly	enable	enabled	enables	finally	for	fortunately	further	furthermore	hence	however	if	incidentally	instead	likewise	moreover	nevertheless	next	nonetheless	nor	or	otherwise	rather	secondly	similarly	summarizing	then	therefore	thereupon	though	thus	unless	whereas	while".split("\t")
        self.all_logical_caveat = "for since so".split(" ")
        
        self.positive_logical = "actually also anyway because besides cause caused causes causing consequence consequences consequently correspondingly enable enabled enables finally fortunately further furthermore hence if incidentally instead likewise moreover next secondly similarly summarizing then therefore thereupon thus while".split(" ")
        self.positive_logical_caveat = "for since so".split(" ")

        self.all_temporal = "earlier finally first further immediately instantly later meanwhile next presently previously secondly simultaneously since soon suddenly then until when whenever while".split(" ")
        self.all_temporal_caveat = "as after before".split(" ")

        self.positive_intentional = "desire	desired	desires	desiring	goal	goals	made	make	makes	making	purpose	purposes	want	wanted	wanting	wants".split("\t")
        self.positive_intentional_caveat = ["by", "so"]
        
        self.all_positive = "actually	also	and	anyway	arise	arises	arising	arose	as	because	besides	cause	caused	causes	causing	condition	conditions	consequence	consequences	consequent	consequently	correspondingly	desire	desired	desires	desiring	due	earlier	enable	enabled	enables	enabling	even	finally	first	follow	followed	following	follows	fortunately	further	furthermore	goal	goals	hence	if	immediately	incidentally	instead	likewise	made	make	makes	making	meanwhile	moreover	next	only	provided	presently	previously	purpose	purposes	result	results	secondly	similarly	simultaneously	since	so	soon	suddenly	summarizing	then	therefore	thereupon	throughout	thus	too	want	wanted	wanting	wants	when	whenever	while".split("\t")
        self.all_positive_caveat = "after	before	by".split("\t")
        
        self.all_connective = "actually	admittedly	again	also	alternatively	although	and	anyhow	anyway	arise	arises	arising	arose	as	because	besides	but	cause	caused	causes	causing	condition	conditions	consequence	consequences	consequent	consequently	correspondingly	desire	desired	desires	desiring	enable	enabled	enables	enabling	finally	first	fortunately	further	furthermore	goal	goals	hence	however	if	immediately	incidentally	instead	likewise	made	make	makes	making	meanwhile	moreover	nevertheless	next	nonetheless	nor	or	otherwise	presently	previously	rather	secondly	similarly	simultaneously	since	summarizing	then	therefore	thereupon	though	throughout	thus	too	unless	until	whenever	whereas	while	yet".split("\t")
        self.all_connective_caveat = "after	before	by so".split("\t")
        
        # Other connectives with no caveats needed
        self.basic_connectives = "and	nor	but	or	yet	so".split("\t")
        self.conjunctions = "and	but".split("\t")
        self.disjunctions = ["or"]
        self.coordinating_conjuncts = "yet	so	nor	however	therefore".split("\t")
        self.addition = "and	also	besides	further	furthermore	too	moreover	in addition	then	another	indeed	likewise".split("\t")
        self.opposition = "but	however	nevertheless	otherwise	on the other hand	on the contrary	yet	still	maybe	perhaps	instead	except for	in spite of	despite	nonetheless	apart from	unlike	whereas".split("\t")
        self.determiners = "a	an	the	this	that	these	those".split("\t")
        self.demonstratives = "this	that	these	those".split("\t")
        self.all_additive = "after all	again	all in all	also	alternatively	and	anyhow	as a final point	as well	at least	besides	but	by contrast	by the way	contrasted with	correspondingly	except that	finally	first	for example	for instance	fortunately	further	furthermore	however	in actual fact	in addition	in contrast	in fact	in other words	in sum	incidentally	instead	it follows	moreover	next	notwithstanding that	on one hand	on the contrary	on the one hand	on the other hand	or	otherwise	rather	secondly	similarly	summarizing	summing up	that is	thereupon	to conclude	to return to	to sum up	to summarize	to take an example	to these ends	to this end	too	well at any rate	whereas	yet".split("\t")
        self.negative_logical = "admittedly	alternatively	although	and conversely	anyhow	but	by contrast	contrasted with	despite the fact that	except that	however	in contrast	nevertheless	nonetheless	nor	notwithstanding that	on the contrary	on the other hand	or else	otherwise	rather	though	unless	whereas	yet".split("\t")
        self.all_negative = "admittedly	alternatively	although	and conversely	anyhow	but	by contrast	contrasted with	despite the fact that	except that	however	in contrast	nevertheless	nonetheless	nor	notwithstanding that	on the contrary	on the other hand	or	otherwise	rather	though	unless	until	whenever	whereas	yet".split("\t")

        # Givenness pronouns
        self.givenness_prp = "he she her him his they them their himself herself themselves his their it its".split(" ")

        # Add missing ngram lists
        self.all_logical_ngram = "after all	all in all	and conversely	arise from	arise out of	arises from	arises out of	arising from	arising out of	arose from	arose out of	as a final point	as a result	as well	at least	at this point	conditional upon	contrasted with	despite the fact that	due to	except that	follow that	follow the	follow this	followed that	followed the	followed this	following that	follows the	follows this	in actual fact	in any case	in any event	in case	in conclusion	in contrast	in fact	in order that	in other words	in short	in sum	it followed that	it follows	it follows that	notwithstanding that	on condition that	on one hand	on the condition that	on the contrary	on the one hand	on the other hand	once again	provided that	purpose of which	pursuant to	summing up	that is	that is to say	to conclude	to return(to	to sum up	to summarize	to take an example	to that end	to these ends	to this end	to those ends	well at any rate".split("\t")
        
        self.positive_logical_ngram = "after all	all in all	arise from	arise out of	arises from	arises out of	arising from	arising out of	arose from	arose out of	as a final point	as a result	as well	at least	at this point	conditional upon	due to	follow that	follow the	follow this	followed that	followed the	followed this	following that	follows the	follows this	in actual fact	in any case	in any event	in case	in conclusion	in fact	in order that	in other words	in short	in sum	it followed that	it follows	it follows that	on condition that	on one hand	on the condition that	on the one hand	once again	provided that	purpose of which	pursuant to	summing up	that is	that is to say	to conclude	to return to	to sum up	to summarize	to take an example	to that end	to these ends	to this end	to those ends	well at any rate".split("\t")

        self.all_temporal_ngram = "a consequence of	all this time	at last	at once	at the same time	at this moment	at this point	by this time	follow that	following that	from now on	in the meantime	it followed that	it follows	it follows that	now that	on another occasion	once more	so far	the consequence of	the consequences of	the last time	the previous moment	this time	throughout	to that end	up till that time	up to now".split("\t")

        self.positive_intentional_ngram = "in order	to that end	to these ends	to this end	to those ends".split("\t")
        
        self.all_positive_ngram = "all in all	at last	at least	at once	conditional upon	from now on	in actual fact	in addition	in any case	in any event	in case	in conclusion	in fact	in order	in other words	in short	in sum	on another occasion	on one hand	once more	summing up	well at any rate".split("\t")
        
        self.all_connective_ngram = "all in all	all this time	at last	at least	at once	at the same time	at this moment	at this point	conditional upon	contrasted with	despite the fact that	due to	except that	follow that	follow the	follow this	followed that	followed the	followed this	following that	follows the	follows this	from now on	in actual fact	in addition	in any case	in any event	in case	in conclusion	in contrast	in fact	in order	in other words	in short	in sum	in the end	in the meantime	it followed that	it follows	it follows that	notwithstanding that	now that	on another occasion	on one hand	on the contrary	on the one hand	on the other hand	once more	provided that	purpose of which	pursuant to	summing up	that is	the last time	the previous moment	this time	to conclude	to return to	to sum up	to summarize	to take an example	to that end	to these ends	to this end	to those ends	up till that time	up to now	well at any rate".split("\t")

        # Load dictionaries if needed
        if self.varDict.get("overlapLSA", False) or self.varDict.get("sourceLSA", False):
            print("Loading LSA vector space...")
            lsaFileList = [self.resource_path("COCA_newspaper_magazine_export_LSA_Small_A.csv"),
                          self.resource_path("COCA_newspaper_magazine_export_LSA_Small_B.csv"),
                          self.resource_path("COCA_newspaper_magazine_export_LSA_Small_C.csv"),
                          self.resource_path("COCA_newspaper_magazine_export_LSA_Small_D.csv"),
                          self.resource_path("COCA_newspaper_magazine_export_LSA_Small_E.csv")]
            self.lsa_dict = self.dicter_2_multi(lsaFileList, "\t", " ", lower=True)

        if self.varDict.get("overlapLDA", False) or self.varDict.get("sourceLDA", False):
            print("Loading LDA vector space...")
            self.lda_dict = self.dicter_2(self.resource_path("COCA_newspaper_magazine_export_LDA.csv"), "\t", " ", lower=True)

        if self.varDict.get("overlapWord2vec", False) or self.varDict.get("sourceWord2vec", False):
            print("Loading word2vec vector space...")
            word2vecFileList = [self.resource_path("COCA_newspaper_magazine_export_word2vec_Small_A.csv"),
                               self.resource_path("COCA_newspaper_magazine_export_word2vec_Small_B.csv"),
                               self.resource_path("COCA_newspaper_magazine_export_word2vec_Small_C.csv"),
                               self.resource_path("COCA_newspaper_magazine_export_word2vec_Small_D.csv"),
                               self.resource_path("COCA_newspaper_magazine_export_word2vec_Small_E.csv")]
            self.word2vec_dict = self.dicter_2_multi(word2vecFileList, "\t", " ", lower=True)

        # Load WordNet and adjective dictionaries
        self.wn_noun_dict = self.dicter(self.resource_path("wn_noun_2.txt"), "\t")
        self.wn_verb_dict = self.dicter(self.resource_path("wn_verb_2.txt"), "\t")
        self.adj_word_list = open("adj_lem_list.txt", errors="ignore").read().split("\n")[:-1]

        # Punctuation list
        self.punctuation = "' . , ? ! ) ( % / - _ -LRB- -RRB- SYM : ; _SP".split(" ")
        self.punctuation.append('"')

    @staticmethod
    def resource_path(relative):
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, relative)
        return os.path.join(relative)

    def dqMessage(self, text):
        if self.gui:
            dataQueue.put(text)
            root.update_idletasks()
        else:
            print(text)

    @staticmethod
    def checkBoxes(tDict, loKeys):
        nTrue = 0
        for x in loKeys:
            if tDict[x] == True:
                nTrue += 1
        return nTrue

    def dicter(self, spread_name, delimiter, lower=False):
        if lower == False:
            spreadsheet = open(self.resource_path(spread_name), errors="ignore").read().split("\n")
        if lower == True:
            spreadsheet = open(self.resource_path(spread_name), errors="ignore").read().lower().split("\n")
            
        dict = {}
        for line in spreadsheet:
            if line == "":
                continue
            if line[0] == "#":
                continue
            vars = line.split(delimiter)
            if len(vars) < 2:
                continue
            dict[vars[0]] = vars[1:]
        
        return dict

    def dicter_2(self, spread_name, delimiter1, delimiter, lower=False):
        if lower == False:
            spreadsheet = open(self.resource_path(spread_name), errors="ignore").read().split("\n")
        if lower == True:
            spreadsheet = open(self.resource_path(spread_name), errors="ignore").read().lower().split("\n")
            
        dict = {}
        for line in spreadsheet:
            if line == "":
                continue
            if line[0] == "#":
                continue
            head = line.split(delimiter1)[0]
            
            if len(line.split(delimiter1)) < 2:
                continue
            vars = line.split(delimiter1)[1].split(delimiter)
            if len(vars) < 2:
                continue
            dict[head] = vars[1:]
        
        return dict

    def dicter_2_multi(self, spread_names, delimiter1, delimiter, lower=False):
        spreadsheet = []
        for spread_name in spread_names:
            if lower == False:
                spreadsheet = spreadsheet + open(self.resource_path(spread_name), errors="ignore").read().split("\n")
            if lower == True:
                spreadsheet = spreadsheet + open(self.resource_path(spread_name), errors="ignore").read().lower().split("\n")
            
        tdict = {}
        for line in spreadsheet:
            if line == "":
                continue
            if line[0] == "#":
                continue
            head = line.split(delimiter1)[0]
            
            if len(line.split(delimiter1)) < 2:
                continue
            vars = line.split(delimiter1)[1].split(delimiter)
            if len(vars) < 2:
                continue
            tdict[head] = vars[1:]
        
        return tdict

    @staticmethod
    def dict_builder(database_file, number, log="n", delimiter="\t"):
        dict = {}
        data_file = database_file.lower().split("\n")
        for entries in data_file:  
            if entries == "":
                continue
            if entries[0] == '#':
                continue
        
            entries = entries.split(delimiter)
            if log == "n": 
                dict[entries[0]] = float(entries[number])
            if log == "y": 
                if not entries[number] == '0':
                    dict[entries[0]] = math.log10(float(entries[number]))

        return dict

    @staticmethod
    def indexer(results, name, index):
        results[name] = index

    @staticmethod
    def safe_divide(numerator, denominator):
        if denominator == 0:
            return 0
        return numerator / denominator

    def para_split(self, textString):
        para_t = textString
        while "\ufeff" in para_t:
            para_t = para_t.replace("\ufeff", "") #this deletes the BOM character that may occur at the beginning of some files

        while "\xa0" in para_t:
            para_t = para_t.replace("\xa0", " ") #this should be sufficient to replace all \xa0 instances and is probably the best course of action
        while "\t" in para_t:
            para_t = para_t.replace("\t", " ")
        #print "check1"			
        while "  " in para_t:
            para_t = para_t.replace("  ", " ")
        #print "check2"
        while "\t\t" in para_t:
            para_t.replace("\t\t","\t")
        #print "check3"			
        while "\n \n" in para_t:
            para_t = para_t.replace("\n \n", "\n")		
        #print "check4"
        while "\n\n" in para_t:
            para_t = para_t.replace("\n\n", "\n")

        if para_t[0] == "\n":
            para_t = para_t[1:]
        if para_t[-1] == "\n":
            para_t = para_t[:-1]
        para_t = para_t.split("\n") # this is a list of strings
        return(para_t)

    def single_givenness_counter(self, text):
        counter = 0
        for item in text:
            if text.count(item) == 1:
                counter += 1
        return counter

    def repeated_givenness_counter(self, text):
        counter = 0
        for item in text:
            if text.count(item) > 1:
                counter += 1
        return counter

    def n_grammer(self, text, length, list=None):
        counter = 0
        ngram_text = []
        for word in text:
            ngram = text[counter:(counter+length)]

            if len(ngram) > (length-1):
                ngram_text.append(" ".join(str(x) for x in ngram))
            counter += 1
        if list is not None:
            for item in ngram_text:
                list.append(item)
        else:
            return ngram_text

    def overlap_counter(self, results, name_suffix, list, seg_1, seg_2):
        n_segments = len(list)  # number of sentences or paragraphs
        
        # this next section deals with texts that only have one segment
        if n_segments < 2:
            if seg_1 == True:
                results["adjacent_overlap_" + name_suffix] = 0
                results["adjacent_overlap_" + name_suffix + "_div_seg"] = 0
                results["adjacent_overlap_binary_" + name_suffix] = 0
            
            if seg_2 == True:
                results["adjacent_overlap_2_"+name_suffix] = 0
                results["adjacent_overlap_2_" + name_suffix + "_div_seg"] = 0
                results["adjacent_overlap_binary_2_"+ name_suffix] = 0
        
        # this is the "normal" procedure
        else:
            single_overlap_denominator = 0
            double_overlap_denominator = 0
            
            overlap_counter_1 = 0
            overlap_counter_2 = 0
            binary_count_1 = 0
            binary_count_2 = 0
            
            for number in range(n_segments-1):
                next_item_overlap = []  # list so that overlap can be recovered for post-hoc
                next_two_item_overlap = []  # list so that overlap can be recovered for post-hoc

                if number < n_segments - 3 or number == n_segments - 3:  # Make sure we didn't go too far
                    for items in set(list[number]):
                        single_overlap_denominator += 1
                        double_overlap_denominator += 1
                        if items in list[number + 1]:
                            next_item_overlap.append(items)
                        if items in list[number + 1] or items in list[number + 2]:
                            next_two_item_overlap.append(items)
                else:  # Make sure we didn't go too far
                    for items in set(list[number]):
                        single_overlap_denominator += 1
                        if items in list[number + 1]:
                            next_item_overlap.append(items)
                
                overlap_counter_1 += len(next_item_overlap)
                overlap_counter_2 += len(next_two_item_overlap)
                if len(next_item_overlap) > 0: 
                    binary_count_1 += 1
                if len(next_two_item_overlap) > 0: 
                    binary_count_2 += 1
            
            if seg_1 == 1:
                overlap_1_nwords = self.safe_divide(overlap_counter_1, single_overlap_denominator)
                overlap_1_nseg = self.safe_divide(overlap_counter_1, n_segments - 1)
                binary_count_1_nsent = self.safe_divide(binary_count_1, n_segments - 1)
                
                results["adjacent_overlap_" + name_suffix] = overlap_1_nwords
                results["adjacent_overlap_" + name_suffix + "_div_seg"] = overlap_1_nseg
                results["adjacent_overlap_binary_" + name_suffix] = binary_count_1_nsent
            
            if seg_2 == 1:
                overlap_2_nwords = self.safe_divide(overlap_counter_2, double_overlap_denominator)
                overlap_2_nseg = self.safe_divide(overlap_counter_2, n_segments - 2)
                binary_count_2_nsent = self.safe_divide(binary_count_2, n_segments - 2)
                
                results["adjacent_overlap_2_"+name_suffix] = overlap_2_nwords
                results["adjacent_overlap_2_" + name_suffix + "_div_seg"] = overlap_2_nseg
                results["adjacent_overlap_binary_2_"+ name_suffix] = binary_count_2_nsent

    def wordnet_dict_build(self, target_list, syn_dict):
        counter = len(target_list)  # this is the number of paragraphs/sentences in the text
        
        # holder structure:
        target_syn_dict = {}
        
        # creates a version of the text where each word is a list of synonyms:
        for i in range(counter):  # iterates as many times as there are sentences/paragraphs in text
            if len(target_list[i]) < 1:
                target_syn_dict[i] = []
            else:
                syn_list1 = []
                for item in target_list[i]:  # for word in sentence/paragraph
                    if item in syn_dict:
                        syns = syn_dict[item]
                    else: 
                        syns = [item]
                    syn_list1.append(syns)
                target_syn_dict[i] = syn_list1
        
        return target_syn_dict

    def syn_overlap(self, results, name_suffix, list, syn_dict):
        counter = len(list)
        if counter < 2:
            syn_counter_norm = 0
        else:
            syn_counter = 0
            for i in range(counter-1):
                for items in set(list[i]):
                    for item in syn_dict[i+1]:
                        if items in item:
                            syn_counter += 1
            syn_counter_norm = self.safe_divide(syn_counter, counter-1)  # note these are divided by segments
        results["syn_overlap_" + name_suffix] = syn_counter_norm

    def multi_list_counter(self, results, word_list, target_list, nwords):
        for lines in word_list:
            if lines[0] == "#":
                continue
            line = lines.split("\t")
            counter = 0
            for words in line[1:]:
                if words == "":
                    continue
                word = " " + words + " "  # adds space to beginning and end to avoid over-counting
                for sentences in target_list:  # iterates through sentences to ensure that sentence boundaries are not crossed
                    sentence = " " + " ".join(sentences) + " "  # turns list of words into a string, adds a space to the beginning and end
                    counter += sentence.count(word)  # counts list instances in each sentence
            results[line[0]] = self.safe_divide(counter, nwords)  # appends normed index to results

    def ngram_counter(self, text, ngram_list):
        checker_text = " " + " ".join(text) + " "
        counter = 0
        new_ngram_list = []
        
        for item in ngram_list:
            new_item = " " + item + " "
            new_ngram_list.append(new_item)
        
        for items in new_ngram_list:
            counter += checker_text.count(items)
        return counter

    def mattr(self, text, window_length):
        if len(text) < (window_length + 1):
            mattr = self.safe_divide(len(set(text)), len(text))
            return mattr
        else:
            sum_ttr = 0
            denom = 0
            for x in range(len(text)):
                small_text = text[x:(x + window_length)]
                if len(small_text) < window_length:
                    break
                denom += 1
                sum_ttr += self.safe_divide(len(set(small_text)), float(window_length))
            mattr = self.safe_divide(sum_ttr, denom)
            return mattr

    def content_pos_dict_spacy(self, text, lemma=False):
        outd = {}
        doc = self.nlp(text)

        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        adjectives = ["JJ", "JJR", "JJS"]
        verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
        adverbs = ["RB", "RBR", "RBS"]
        verbs_nouns = ["NN", "NNS", "NNP", "NNPS","VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
        nouns_adjectives = ["NN", "NNS", "NNP", "NNPS","JJ", "JJR", "JJS"]

        s_noun_text = []
        s_adjective_text = []
        s_verb_text = []
        s_verb_noun_text = []
        s_adverb_text = []
        s_all_text = []

        for sent in doc.sents:
            noun_text = []
            adjective_text = []
            verb_text = []
            verb_noun_text = []
            adverb_text = []
            content_text = []
            all_text = []

            for token in sent:
                if lemma == False:
                    tok_item = token.text.lower()
                if lemma == True:
                    tok_item = token.lemma_.lower()
                if token.tag_ in self.punctuation:
                    continue
                all_text.append(tok_item)
                if token.tag_ in noun_tags:
                    noun_text.append(tok_item)
                    verb_noun_text.append(tok_item)
                if token.tag_ in adjectives:
                    adjective_text.append(tok_item)
                if token.tag_ in verbs:
                    verb_text.append(tok_item)
                    verb_noun_text.append(tok_item)
                if token.tag_ in adverbs:
                    adverb_text.append(tok_item)

            s_noun_text.append(noun_text)
            s_adjective_text.append(adjective_text)
            s_verb_text.append(verb_text)
            s_verb_noun_text.append(verb_noun_text)
            s_adverb_text.append(adverb_text)
            s_all_text.append(all_text)

        all_noun = [item for sublist in s_noun_text for item in sublist]
        all_adjective = [item for sublist in s_adjective_text for item in sublist]
        all_verb = [item for sublist in s_verb_text for item in sublist]
        all_verb_noun = [item for sublist in s_verb_noun_text for item in sublist]
        all_adverb = [item for sublist in s_adverb_text for item in sublist]
        all_all = [item for sublist in s_all_text for item in sublist]

        outd["s_all"] = s_all_text
        outd["noun"] = all_noun
        outd["adj"] = all_adjective
        outd["verb"] = all_verb
        outd["verb_noun"] = all_verb_noun
        outd["adv"] = all_adverb
        outd["all"] = all_all

        return outd

    def ngram_pos_dict_spacy(self, text, lemma=False):
        def dict_add(tdict, list, name, sent=False):
            if sent:
                if name in tdict:
                    tdict[name].append(list)
                else:
                    tdict[name] = [list]
            else:
                if name in tdict:
                    for items in list:
                        tdict[name].append(items)
                else:
                    tdict[name] = list

        def lemma_lister(sentence, constraint=None):
            list = []
            for token in sentence:
                if token.tag_ in self.punctuation:
                    continue
                if constraint is None:
                    if lemma:
                        list.append(token.lemma_.lower())
                    else:
                        list.append(token.text.lower())
                else:
                    if token.tag_ in constraint:
                        if lemma:
                            list.append(token.lemma_.lower())
                        else:
                            list.append(token.text.lower())
                    else:
                        list.append("x")
            return list

        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        adjectives = ["JJ", "JJR", "JJS"]
        verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
        adverbs = ["RB", "RBR", "RBS"]
        verbs_nouns = ["NN", "NNS", "NNP", "NNPS","VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
        nouns_adjectives = ["NN", "NNS", "NNP", "NNPS","JJ", "JJR", "JJS"]

        frequency_dict = {}
        doc = self.nlp(text)

        uni_list = []
        bi_list = []
        tri_list = []
        quad_list = []

        n_list_bi = []
        adj_list_bi = []
        v_list_bi = []
        v_n_list_bi = []
        a_n_list_bi = []

        n_list_tri = []
        adj_list_tri = []
        v_list_tri = []
        v_n_list_tri = []
        a_n_list_tri = []

        n_list_quad = []
        adj_list_quad = []
        v_list_quad = []
        v_n_list_quad = []
        a_n_list_quad = []

        for sent in doc.sents:
            word_list = lemma_lister(sent)

            for items in word_list:
                uni_list.append(items)

            n_list = lemma_lister(sent, noun_tags)
            adj_list = lemma_lister(sent, adjectives)
            v_list = lemma_lister(sent, verbs)
            v_n_list = lemma_lister(sent, verbs_nouns)
            a_n_list = lemma_lister(sent, nouns_adjectives)

            self.n_grammer(word_list, 2, bi_list)
            self.n_grammer(word_list, 3, tri_list)
            self.n_grammer(word_list, 4, quad_list)

            self.n_grammer(n_list, 2, n_list_bi)
            self.n_grammer(adj_list, 2, adj_list_bi)
            self.n_grammer(v_list, 2, v_list_bi)
            self.n_grammer(v_n_list, 2, v_n_list_bi)
            self.n_grammer(a_n_list, 2, a_n_list_bi)

            self.n_grammer(n_list, 3, n_list_tri)
            self.n_grammer(adj_list, 3, adj_list_tri)
            self.n_grammer(v_list, 3, v_list_tri)
            self.n_grammer(v_n_list, 3, v_n_list_tri)
            self.n_grammer(a_n_list, 3, a_n_list_tri)

            self.n_grammer(n_list, 4, n_list_quad)
            self.n_grammer(adj_list, 4, adj_list_quad)
            self.n_grammer(v_list, 4, v_list_quad)
            self.n_grammer(v_n_list, 4, v_n_list_quad)
            self.n_grammer(a_n_list, 4, a_n_list_quad)

        dict_add(frequency_dict, bi_list, "bi_list")
        dict_add(frequency_dict, tri_list, "tri_list")
        dict_add(frequency_dict, quad_list, "quad_list")

        dict_add(frequency_dict, n_list_bi, "n_list_bi")
        dict_add(frequency_dict, adj_list_bi, "adj_list_bi")
        dict_add(frequency_dict, v_list_bi, "v_list_bi")
        dict_add(frequency_dict, v_n_list_bi, "v_n_list_bi")
        dict_add(frequency_dict, a_n_list_bi, "a_n_list_bi")

        dict_add(frequency_dict, n_list_tri, "n_list_tri")
        dict_add(frequency_dict, adj_list_tri, "adj_list_tri")
        dict_add(frequency_dict, v_list_tri, "v_list_tri")
        dict_add(frequency_dict, v_n_list_tri, "v_n_list_tri")
        dict_add(frequency_dict, a_n_list_tri, "a_n_list_tri")

        dict_add(frequency_dict, n_list_quad, "n_list_quad")
        dict_add(frequency_dict, adj_list_quad, "adj_list_quad")
        dict_add(frequency_dict, v_list_quad, "v_list_quad")
        dict_add(frequency_dict, v_n_list_quad, "v_n_list_quad")
        dict_add(frequency_dict, a_n_list_quad, "a_n_list_quad")

        return frequency_dict

    def keyness(self, target_list, frequency_list_dict, top_perc=None, out_dir="", keyname=""):
        if top_perc is None:
            top_perc = 0.1

        keyness_dict = {}
        for word in target_list:
            if word in frequency_list_dict:
                keyness_dict[word] = frequency_list_dict[word]

        sorted_keyness = sorted(keyness_dict.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(sorted_keyness) * top_perc)

        if out_dir:
            with open(os.path.join(out_dir, keyname + "_keyness.txt"), "w") as f:
                for word, score in sorted_keyness[:top_n]:
                    f.write(f"{word}\t{score}\n")

        return sorted_keyness[:top_n]

    def simple_proportion(self, target_text, ref_text, type, results=None, index_name=None):
        if type == "word":
            target_count = len(target_text)
            ref_count = len(ref_text)
        elif type == "char":
            target_count = sum(len(word) for word in target_text)
            ref_count = sum(len(word) for word in ref_text)
        else:
            raise ValueError("Type must be 'word' or 'char'")

        proportion = self.safe_divide(target_count, ref_count)

        if index_name and results is not None:
            results[index_name] = proportion

        return proportion

    def lsa_similarity(self, text_list_1, text_list_2, lsa_matrix_dict, results=None, index_name=None, lsa_type="fwd", nvectors=300):
        def vector_av(text_list):
            n_items = 0
            l = []
            for i in range(nvectors):
                l.append(0)
    
            for items in text_list:
                if items not in lsa_matrix_dict:
                    continue
                else:
                    n_columns = 0
                    n_items+=1
                    for vector in lsa_matrix_dict[items]:
                        l[n_columns] += float(vector)
                        n_columns +=1

            #n_columns = 0
            #for items in l:
            #	l[n_columns] = l[n_columns]/n_items

            sum_count = 0
            for items in l:
                sum_count += math.pow(items,2)
            sqrt_sum = math.sqrt(sum_count)
        
    
            return([l, sqrt_sum])

        list1 = vector_av(text_list_1)
        list2 = vector_av(text_list_2)

        try:
            sum_count_2 = 0
            for items in range(len(list1[0])):
                sum_count_2+= (list1[0][items]*list2[0][items])


            cosine_sim = sum_count_2/(list1[1]*list2[1])

        except ZeroDivisionError:
            cosine_sim = "null"


        if index_name and results is not None:
            results[index_name] = cosine_sim

        return cosine_sim

    def lda_divergence(self, text_list_1, text_list_2, dict, nvectors=300):
        def vector_av(text_list):
            n_items = 0
            l = []
            for i in range(nvectors):
                l.append(0)
    
            for items in text_list:
                if items not in dict:
                    continue
                else:
                    n_columns = 0
                    n_items+=1
                    for vector in dict[items]:
                        l[n_columns] += float(vector)
                        n_columns +=1
            try:
                for x in range(len(l)): #normalize for number of words
                    l[x] = self.safe_divide(l[x],n_items)
                
                div = np.linalg.norm(l, ord = 1) #linear algebra normalization - all items in list sum to 1
                for x in range(len(l)):
                    l[x] = self.safe_divide(x,div)

            except ZeroDivisionError:
                l = "null"
            
            return(l)

        def jsdiv(P, Q):
            def _kldiv(A, B):
                return(np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)]))

            P = np.array(P)
            Q = np.array(Q)

            M = 0.5 * (P + Q)

            return(0.5 * (_kldiv(P, M) +_kldiv(Q, M)))

        list1 = vector_av(text_list_1)
        list2 = vector_av(text_list_2)
        #print list1[:20]
        if list1 == "null" or list2 == "null":
            divergence = "null"
        else:
            divergence = 1 - jsdiv(list1,list2)
            if divergence >= 0: #this silliness controls for some wacky output. This was in Mihai's code... not sure how the wackiness occurs (it is rare)
                divergence = divergence
            else:
                divergence = "null"
        return(divergence)

    def segment_compare(self, results, text_list_of_lists, seg, name_suffix, lsa_matrix_dict, type="lsa"):
        n_segments = len(text_list_of_lists)

        # Check if we have enough segments
        if n_segments < (seg + 1):
            results[name_suffix] = 0
            return

        denominator = 0
        counter = 0

        for number in range(n_segments - seg):
            item_list = text_list_of_lists[number]  # e.g., sentence 1
            comparison_list = text_list_of_lists[number + 1][:]  # e.g., sentence 2
            
            if seg == 2:
                # For seg=2, combine the next two segments
                for items in text_list_of_lists[number + 2]:
                    comparison_list.append(items)

            if type == "lsa":
                index = self.lsa_similarity(item_list, comparison_list, lsa_matrix_dict)
                if index == "null":
                    continue
                else:
                    counter += index
                    denominator += 1

            if type == "lda":
                index = self.lda_divergence(item_list, comparison_list, lsa_matrix_dict)
                if index == "null":
                    continue
                else:
                    counter += index
                    denominator += 1

        results[name_suffix] = self.safe_divide(counter, denominator)

    def process_text(self, text: str) -> dict:
        """Process a single text and return results as a dictionary."""
        if len(text.split()) <= 1:
            return {"error": "Text too short"}

        results = {}
        para_text = self.para_split(text)
        
        # Initialize tracking lists
        raw_text, lemma_text = [], []
        content_text, function_text = [], []
        noun_text, verb_text = [], []
        adj_text, adv_text = [], []
        prp_text, argument_text = [], []
        
        # For source text overlap
        all_verb_text = []
        all_verb_noun_text = []
        all_verb_x_text = []
        noun_x_text = []
        noun_verb_x_text = []
        adj_noun_x_text = []
        adj_x_test = []
        
        # For connectives
        subordinators = []
        sentence_linker_list = []
        order_list = []
        reason_and_purpose_list = []
        all_causal_list = []
        positive_causal_list = []
        all_logical_list = []
        positive_logical_list = []
        all_temporal_list = []
        positive_intentional_list = []
        all_positive_list = []
        all_connective_list = []
        attended_demonstratives_list = []
        unattended_demonstratives_list = []
        
        # For paragraph and sentence level analysis
        para_raw_list = []
        para_lemma_list = []
        para_content_list = []
        para_function_list = []
        para_pos_noun_list = []
        para_pos_verb_list = []
        para_pos_adj_list = []
        para_pos_adv_list = []
        para_pos_prp_list = []
        para_pos_argument_list = []
        
        sent_raw_list = []
        sent_lemma_list = []
        sent_content_list = []
        sent_function_list = []
        sent_pos_noun_list = []
        sent_pos_verb_list = []
        sent_pos_adj_list = []
        sent_pos_adv_list = []
        sent_pos_prp_list = []
        sent_pos_argument_list = []
        
        # Process text
        for paragraph in para_text:
            doc = self.nlp(paragraph)
            
            # Initialize paragraph lists
            raw_list_para = []
            lemma_list_para = []
            content_list_para = []
            function_list_para = []
            pos_noun_list_para = []
            pos_verb_list_para = []
            pos_adj_list_para = []
            pos_adv_list_para = []
            pos_prp_list_para = []
            pos_argument_list_para = []
            
            for sent in doc.sents:
                # Initialize sentence lists
                raw_list_sent = []
                lemma_list_sent = []
                content_list_sent = []
                function_list_sent = []
                pos_noun_list_sent = []
                pos_verb_list_sent = []
                pos_adj_list_sent = []
                pos_adv_list_sent = []
                pos_prp_list_sent = []
                pos_argument_list_sent = []
                
                for token in sent:
                    if token.tag_ in self.punctuation:
                        continue
                    
                    raw_token = token.text.lower()
                    lemma_token = token.lemma_.lower()
                    POS_token = token.tag_
                    
                    # Track words
                    raw_text.append(raw_token)
                    raw_list_para.append(raw_token)
                    raw_list_sent.append(raw_token)
                    
                    lemma_text.append(lemma_token)
                    lemma_list_para.append(lemma_token)
                    lemma_list_sent.append(lemma_token)
                    
                    # Track content/function words
                    if POS_token in self.content:
                        content_text.append(lemma_token)
                        content_list_para.append(lemma_token)
                        content_list_sent.append(lemma_token)
                    if POS_token not in self.prelim_not_function:
                        function_text.append(lemma_token)
                        function_list_para.append(lemma_token)
                        function_list_sent.append(lemma_token)
                    
                    # Track POS
                    if POS_token in self.verbs:
                        all_verb_text.append(lemma_token)
                        all_verb_x_text.append(lemma_token)
                        
                        if token.tag_[:2] in ["VB","MD"] and token.pos_ in ["AUX"]:
                            # Auxiliary verbs are function words
                            function_text.append(lemma_token)
                            function_list_para.append(lemma_token)
                            function_list_sent.append(lemma_token)
                        else:
                            # Main verbs are content words and go in verb_text
                            content_text.append(lemma_token)
                            content_list_para.append(lemma_token)
                            content_list_sent.append(lemma_token)
                            
                            verb_text.append(lemma_token)
                            pos_verb_list_para.append(lemma_token)
                            pos_verb_list_sent.append(lemma_token)
                    else:
                        all_verb_x_text.append("x")
                        
                    if POS_token in self.noun_tags:
                        noun_x_text.append(lemma_token)
                        noun_text.append(lemma_token)
                        pos_noun_list_para.append(lemma_token)
                        pos_noun_list_sent.append(lemma_token)
                    else:
                        noun_x_text.append("x")
                        
                    if POS_token in self.adjectives:
                        adj_x_test.append(lemma_token)
                        adj_text.append(lemma_token)
                        pos_adj_list_para.append(lemma_token)
                        pos_adj_list_sent.append(lemma_token)
                    else:
                        adj_x_test.append("x")
                        
                    if POS_token in self.adverbs:
                        adv_text.append(lemma_token)
                        pos_adv_list_para.append(lemma_token)
                        pos_adv_list_sent.append(lemma_token)
                        if lemma_token in self.adj_word_list or (lemma_token[-2:] == "ly" and lemma_token[:-2] in self.adj_word_list):
                            content_text.append(lemma_token)
                            content_list_para.append(lemma_token)
                            content_list_sent.append(lemma_token)
                        else:
                            function_text.append(lemma_token)
                            function_list_para.append(lemma_token)
                            function_list_sent.append(lemma_token)
                            
                    if POS_token in self.pronouns:
                        prp_text.append(lemma_token)
                        pos_prp_list_para.append(lemma_token)
                        pos_prp_list_sent.append(lemma_token)
                        
                    if POS_token in self.pronouns or POS_token in self.noun_tags:
                        argument_text.append(lemma_token)
                        pos_argument_list_para.append(lemma_token)
                        pos_argument_list_sent.append(lemma_token)
                        
                    if POS_token in self.verbs or POS_token in self.noun_tags:
                        noun_verb_x_text.append(lemma_token)
                        all_verb_noun_text.append(lemma_token)
                    else:
                        noun_verb_x_text.append("x")
                        
                    if POS_token in self.adjectives or POS_token in self.noun_tags:
                        adj_noun_x_text.append(lemma_token)
                    else:
                        adj_noun_x_text.append("x")
                    
                    # Track connectives
                    if token.dep_ in ["mark"]:
                        subordinators.append(raw_token)
                        
                    if raw_token in self.sentence_linkers:
                        sentence_linker_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.sentence_linkers_caveat:
                        sentence_linker_list.append(raw_token)
                        
                    # Track other connectives
                    if raw_token in self.order:
                        order_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.order_caveat:
                        order_list.append(raw_token)
                        
                    if raw_token in self.reason_and_purpose:
                        reason_and_purpose_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.reason_and_purpose_caveat:
                        reason_and_purpose_list.append(raw_token)
                        
                    # Track additional connectives
                    if raw_token in self.all_logical:
                        all_logical_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.all_logical_caveat:
                        all_logical_list.append(raw_token)

                    if raw_token in self.positive_logical:
                        positive_logical_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.positive_logical_caveat:
                        positive_logical_list.append(raw_token)

                    if raw_token in self.all_temporal:
                        all_temporal_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.all_temporal_caveat:
                        all_temporal_list.append(raw_token)

                    if raw_token in self.positive_intentional:
                        positive_intentional_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.positive_intentional_caveat:
                        positive_intentional_list.append(raw_token)

                    if raw_token in self.all_positive:
                        all_positive_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.all_positive_caveat:
                        all_positive_list.append(raw_token)

                    if raw_token in self.all_connective:
                        all_connective_list.append(raw_token)
                    if token.dep_ in ["mark"] and raw_token in self.all_connective_caveat:
                        all_connective_list.append(raw_token)

                    # Track demonstratives
                    if raw_token in self.demonstratives:
                        if token.dep_ in ["det"]:
                            attended_demonstratives_list.append(raw_token)
                        else:
                            unattended_demonstratives_list.append(raw_token)
                            prp_text.append(lemma_token)
                            pos_prp_list_para.append(lemma_token)
                            pos_prp_list_sent.append(lemma_token)
                
                # Add sentence lists to full sentence lists
                sent_raw_list.append(raw_list_sent)
                sent_lemma_list.append(lemma_list_sent)
                sent_content_list.append(content_list_sent)
                sent_function_list.append(function_list_sent)
                sent_pos_noun_list.append(pos_noun_list_sent)
                sent_pos_verb_list.append(pos_verb_list_sent)
                sent_pos_adj_list.append(pos_adj_list_sent)
                sent_pos_adv_list.append(pos_adv_list_sent)
                sent_pos_prp_list.append(pos_prp_list_sent)
                sent_pos_argument_list.append(pos_argument_list_sent)
            
            # Add paragraph lists to full paragraph lists
            para_raw_list.append(raw_list_para)
            para_lemma_list.append(lemma_list_para)
            para_content_list.append(content_list_para)
            para_function_list.append(function_list_para)
            para_pos_noun_list.append(pos_noun_list_para)
            para_pos_verb_list.append(pos_verb_list_para)
            para_pos_adj_list.append(pos_adj_list_para)
            para_pos_adv_list.append(pos_adv_list_para)
            para_pos_prp_list.append(pos_prp_list_para)
            para_pos_argument_list.append(pos_argument_list_para)
        
        # Calculate basic metrics
        nwords = len(raw_text)
        nprps = len(prp_text)
        nnouns = len(noun_text)
        nsentences = len(sent_lemma_list)
        nparagraphs = len(para_lemma_list)
        
        # Store basic metrics
        # results.update({
        #     "nwords": nwords,
        #     "nprps": nprps,
        #     "nnouns": nnouns,
        #     "nsentences": nsentences,
        #     "nparagraphs": nparagraphs
        # })
        
        # Calculate TTR metrics
        nlemmas = len(lemma_text)
        nlemma_types = len(set(lemma_text))
        ncontent_words = len(content_text)
        ncontent_types = len(set(content_text))
        nfunction_words = len(function_text)
        nfunction_types = len(set(function_text))
        
        results.update({
            "lemma_ttr": self.safe_divide(nlemma_types, nlemmas),
            "lemma_mattr": self.mattr(text=lemma_text, window_length=50),
            "content_ttr": self.safe_divide(ncontent_types, ncontent_words),
            "function_ttr": self.safe_divide(nfunction_types, nfunction_words),
            "function_mattr": self.mattr(text=function_text, window_length=50),
            "lexical_density_tokens": self.safe_divide(ncontent_words, nlemmas),
            "lexical_density_types": self.safe_divide(ncontent_types, nlemma_types)
        })
        
        # Calculate POS-specific TTR
        results.update({
            "noun_ttr": self.safe_divide(len(set(noun_text)), len(noun_text)),
            "verb_ttr": self.safe_divide(len(set(verb_text)), len(verb_text)),
            "adj_ttr": self.safe_divide(len(set(adj_text)), len(adj_text)),
            "adv_ttr": self.safe_divide(len(set(adv_text)), len(adv_text)),
            "prp_ttr": self.safe_divide(len(set(prp_text)), len(prp_text)),
            "argument_ttr": self.safe_divide(len(set(argument_text)), len(argument_text))
        })
        
        # Calculate N-gram metrics
        if self.varDict.get("overlapNgrams", False):
            bigram_lemma_text = self.n_grammer(lemma_text, 2)
            trigram_lemma_text = self.n_grammer(lemma_text, 3)
            
            results.update({
                "bigram_lemma_ttr": self.safe_divide(len(set(bigram_lemma_text)), len(bigram_lemma_text)),
                "trigram_lemma_ttr": self.safe_divide(len(set(trigram_lemma_text)), len(trigram_lemma_text))
            })
        
        # Calculate overlap metrics
        if self.varDict.get("overlapSentence", False):
            self.overlap_counter(results, "all_sent", sent_lemma_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "cw_sent", sent_content_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "fw_sent", sent_function_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "noun_sent", sent_pos_noun_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "verb_sent", sent_pos_verb_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "adj_sent", sent_pos_adj_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "adv_sent", sent_pos_adv_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "pronoun_sent", sent_pos_prp_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "argument_sent", sent_pos_argument_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
        
        if self.varDict.get("overlapParagraph", False):
            self.overlap_counter(results, "all_para", para_lemma_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "cw_para", para_content_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "fw_para", para_function_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "noun_para", para_pos_noun_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "verb_para", para_pos_verb_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "adj_para", para_pos_adj_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "adv_para", para_pos_adv_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "pronoun_para", para_pos_prp_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
            self.overlap_counter(results, "argument_para", para_pos_argument_list, self.varDict.get("overlapAdjacent", False), self.varDict.get("overlapAdjacent2", False))
        
        # Calculate WordNet overlap metrics
        if self.varDict.get("overlapSynonym", False):
            noun_sent_syn_lemma_dict = self.wordnet_dict_build(sent_pos_noun_list, self.wn_noun_dict)
            verb_sent_syn_lemma_dict = self.wordnet_dict_build(sent_pos_verb_list, self.wn_verb_dict)
            noun_para_syn_lemma_dict = self.wordnet_dict_build(para_pos_noun_list, self.wn_noun_dict)
            verb_para_syn_lemma_dict = self.wordnet_dict_build(para_pos_verb_list, self.wn_verb_dict)
            
            self.syn_overlap(results, "sent_noun", sent_pos_noun_list, noun_sent_syn_lemma_dict)
            self.syn_overlap(results, "sent_verb", sent_pos_verb_list, verb_sent_syn_lemma_dict)
            self.syn_overlap(results, "para_noun", para_pos_noun_list, noun_para_syn_lemma_dict)
            self.syn_overlap(results, "para_verb", para_pos_verb_list, verb_para_syn_lemma_dict)
        
        # Calculate semantic similarity metrics
        if self.varDict.get("overlapLSA", False):
            self.segment_compare(results, sent_lemma_list, 1, "lsa_1_all_sent", self.lsa_dict)
            self.segment_compare(results, sent_lemma_list, 2, "lsa_2_all_sent", self.lsa_dict)
            self.segment_compare(results, para_lemma_list, 1, "lsa_1_all_para", self.lsa_dict)
            self.segment_compare(results, para_lemma_list, 2, "lsa_2_all_para", self.lsa_dict)
        
        if self.varDict.get("overlapLDA", False):
            self.segment_compare(results, sent_lemma_list, 1, "lda_1_all_sent", self.lda_dict, "lda")
            self.segment_compare(results, sent_lemma_list, 2, "lda_2_all_sent", self.lda_dict, "lda")
            self.segment_compare(results, para_lemma_list, 1, "lda_1_all_para", self.lda_dict, "lda")
            self.segment_compare(results, para_lemma_list, 2, "lda_2_all_para", self.lda_dict, "lda")
        
        if self.varDict.get("overlapWord2vec", False):
            self.segment_compare(results, sent_lemma_list, 1, "word2vec_1_all_sent", self.word2vec_dict)
            self.segment_compare(results, sent_lemma_list, 2, "word2vec_2_all_sent", self.word2vec_dict)
            self.segment_compare(results, para_lemma_list, 1, "word2vec_1_all_para", self.word2vec_dict)
            self.segment_compare(results, para_lemma_list, 2, "word2vec_2_all_para", self.word2vec_dict)
        
        # Calculate connective metrics
        if self.varDict.get("otherConnectives", False):
            results.update({
                "basic_connectives": self.safe_divide(self.ngram_counter(raw_text, self.basic_connectives), nwords),
                "conjunctions": self.safe_divide(self.ngram_counter(raw_text, self.conjunctions), nwords),
                "disjunctions": self.safe_divide(self.ngram_counter(raw_text, self.disjunctions), nwords),
                "lexical_subordinators": self.safe_divide(len(subordinators), nwords),
                "coordinating_conjuncts": self.safe_divide(self.ngram_counter(raw_text, self.coordinating_conjuncts), nwords),
                "addition": self.safe_divide(self.ngram_counter(raw_text, self.addition), nwords),
                "sentence_linking": self.safe_divide(len(sentence_linker_list), nwords),
                "order": self.safe_divide((len(order_list) + self.ngram_counter(raw_text, self.order_ngram)), nwords),
                "reason_and_purpose": self.safe_divide((len(reason_and_purpose_list) + self.ngram_counter(raw_text, self.reason_and_purpose_ngram)), nwords),
                "all_causal": self.safe_divide(self.ngram_counter(raw_text, self.all_causal), nwords),
                "positive_causal": self.safe_divide(self.ngram_counter(raw_text, self.positive_causal), nwords),
                "opposition": self.safe_divide(self.ngram_counter(raw_text, self.opposition), nwords),
                "determiners": self.safe_divide(self.ngram_counter(raw_text, self.determiners), nwords),
                "all_demonstratives": self.safe_divide(self.ngram_counter(raw_text, self.demonstratives), nwords),
                "attended_demonstratives": self.safe_divide(len(attended_demonstratives_list), nwords),
                "unattended_demonstratives": self.safe_divide(len(unattended_demonstratives_list), nwords),
                "all_additive": self.safe_divide(self.ngram_counter(raw_text, self.all_additive), nwords),
                "all_logical": self.safe_divide((len(all_logical_list) + self.ngram_counter(raw_text, self.all_logical_ngram)), nwords),
                "positive_logical": self.safe_divide((len(positive_logical_list) + self.ngram_counter(raw_text, self.positive_logical_ngram)), nwords),
                "negative_logical": self.safe_divide(self.ngram_counter(raw_text, self.negative_logical), nwords),
                "all_temporal": self.safe_divide((len(all_temporal_list) + self.ngram_counter(raw_text, self.all_temporal_ngram)), nwords),
                "positive_intentional": self.safe_divide((len(positive_intentional_list) + self.ngram_counter(raw_text, self.positive_intentional_ngram)), nwords),
                "all_positive": self.safe_divide((len(all_positive_list) + self.ngram_counter(raw_text, self.all_positive_ngram)), nwords),
                "all_negative": self.safe_divide(self.ngram_counter(raw_text, self.all_negative), nwords),
                "all_connective": self.safe_divide((len(all_connective_list) + self.ngram_counter(raw_text, self.all_connective_ngram)), nwords)
            })
        
        # Calculate givenness metrics
        if self.varDict.get("otherGivenness", False):
            results.update({
                "pronoun_density": self.safe_divide((len(prp_text) + len(unattended_demonstratives_list)), nwords),
                "pronoun_noun_ratio": self.safe_divide((len(prp_text) + len(unattended_demonstratives_list)), nnouns),
                "repeated_content_lemmas": self.safe_divide(self.repeated_givenness_counter(content_text), nwords),
                "repeated_content_and_pronoun_lemmas": self.safe_divide(
                    (self.repeated_givenness_counter(content_text) + 
                     self.repeated_givenness_counter(prp_text) + 
                     self.repeated_givenness_counter(unattended_demonstratives_list)),
                    nwords
                )
            })
        
        return results
