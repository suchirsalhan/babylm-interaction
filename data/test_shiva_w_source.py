from TAACOnoGUI_edited import runTAACO

#set processing options
sampleVars = {"sourceKeyOverlap" : True, "sourceLSA" : True, "sourceLDA" : True, "sourceWord2vec" : True, "wordsAll" : True, "wordsContent" : True, "wordsFunction" : True, "wordsNoun" : True, "wordsPronoun" : True, "wordsArgument" : True, "wordsVerb" : True, "wordsAdjective" : True, "wordsAdverb" : True, "overlapSentence" : True, "overlapParagraph" : True, "overlapAdjacent" : True, "overlapAdjacent2" : True, "otherTTR" : True, "otherConnectives" : True, "otherGivenness" : True, "overlapLSA" : True, "overlapLDA" : True, "overlapWord2vec" : True, "overlapSynonym" : True, "overlapNgrams" : True, "outputTagged" : False, "outputDiagnostic" : False}

# In TAACO github it is said that Source overlap indices measure overlap between the source text (e.g., a reading passage) and the target text (e.g., an essay that references the source text).
# Here, I use prompts as source texts.

# Run TAACO on a folder of texts ("ELLIPSE_Sample/"), give the output file a name ("packageTest.csv), provide output for particular indices/options (as defined in sampleVars)
runTAACO("../generated_texts/", "../generated_texts_scores_w_source.csv", sampleVars, source_text="../source_inputs/")


