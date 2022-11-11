import re
import nltk
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords')

class Tagger:
    def __init__(self, topic) -> None:
        """
        Gets the text from a Wikipedia page

        Sends an HTTP request to get the html contents of a wikipedia page, extracts the text and cleans it. Saves text to the .text parameter of the instance.

        :param topic: The last part of a valid Wikipedia URL, for example "Ranomafana_National_Park".
        :type topic: str
        """
        self.url = "https://en.wikipedia.org/wiki/" + topic
        self.page = requests.get(self.url)
        self.soup = BeautifulSoup(self.page.content, "html.parser")
        self.title = self.soup.find("title").text.strip(" - Wikipedia")
        results = self.soup.find(id="mw-content-text")
        mw_parser_output = [result for result in results.children][0]
        text_list = [entry.text for entry in mw_parser_output.children if entry in mw_parser_output.find_all(['p', 'h2', 'ul','h3'])] # This gets all text that is not in a table!
        text = ' '.join(text_list)
        text = text.replace('\n', ' ') # Replace new lines with spaces
        text = re.sub(" +"," ",text) # Replace multiple spaces with just one
        text = re.sub("\[[0-9]+\]","",text) # take out reference numbers
        text = re.sub("\([0-9]+\)","",text) # take out (1), (34) etc.
        self.text = re.sub("\[[a-zA-Z]+\]","",text)

    def printStart(self, N = 2000):
        """
        Prints the first N characters of the Wikipedia article text.

        :param N: Number of characters to print, defaults to 2000.
        :type N: int, optional
        """
        print(self.text[:N])

    def _doc(self):
        self.nlp = spacy.load('en_core_web_md') # Loads the medium english core web spaCy model.
        self.doc = self.nlp(self.text) # Creates a document object from the Wikipedia article text.

    def getPOS(self, N=5):
        """
        Finds the N most frequent tokens along with their parts of speech and count.

        :param N: Number of tokens to return, defaults to 5.
        :type N: int, optional
        :return: N most frequent tokens along with their parts of speech and count.
        :rtype: pandas value_counts object
        """
        self._doc()
        tokens = {'token': [], 'pos': []}
        for token in self.doc:
            if token.text.lower() not in stopwords.words('english') and token.pos_ != "PUNCT":
                tokens["token"].append(token.text.lower())
                tokens["pos"].append(token.pos_)
        self.tokens = pd.DataFrame(tokens)
        return self.tokens.value_counts(["token", "pos"]).head(N)

    def getTags(self, N=10):
        """
        Finds N tags for the wikipedia article.

        Tags contain only letters and are the most common nouns and proper nouns in the article.

        :param N: Number of tags to return, defaults to 10
        :type N: int, optional
        :returns: N tags
        :rtype: list
        """
        self.getPOS()
        possible_tags = self.tokens[~self.tokens.token.str.contains('[^a-zA-Z]')]
        possible_tags = possible_tags[(self.tokens.pos == 'NOUN') | (self.tokens.pos == 'PROPN')].value_counts(["token", "pos"]).reset_index()
        self.tags = possible_tags['token'].values[:N]
        return self.tags