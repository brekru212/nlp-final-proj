from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz',
                       'stanford-ner-2018-02-27/stanford-ner.jar')
text = "'Inco Ltd., the Canadian nickel producer being bought by  Brazil 's Cia. Vale do Rio, said third-quarter profit soared 11-fold, boosted by surging metal prices and fees paid by Falconbridge Ltd. after a failed takeover. Net income jumped to $701 million, or $3.08 a share, from $64 million, or 29 cents, a year earlier, Toronto-based Inco said today in a statement. Results included $109 million in net fees from the failed deals with Falconbridge and Phelps Dodge Corp. Sales jumped to $2.32 billion from $1.08 billion. Inco sold nickel at double the price last year on average, and output jumped 13 percent. Demand for the metal, used in stainless steel, surged as global economic growth fueled demand, especially in  China . Mines have failed to keep pace, prompting a buying spree by producers seeking to bolster ore deposits. Vale outbid Phelps Dodge and Teck Cominco Ltd. with its $17.3 billion bid. ``Record quarterly earnings reflect the unprecedented sustained strength we've seen in the nickel market, combined with strong production,'' Inco Chief Executive Officer Scott Hand said in the statement."
tokenized = word_tokenize(text)
classified = st.tag(tokenized)
print classified