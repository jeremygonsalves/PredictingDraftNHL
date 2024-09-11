import re
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer



class NltkPreprocessor():
    """
    This class for a NLTK text preprocessing pipeline comes from
    https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
    """

    def __init__(self, reports : pd.Series):

        # merge all scouting reports into one report
        self.reports = reports
        self.report_name = reports.name
        self.tokens = None
        self._lowered = False

        self.stop_words = stopwords.words('english')

        self.pos_tag_dict = {
            'J' : wordnet.ADJ,
            'N' : wordnet.NOUN,
            'V' : wordnet.VERB,
            'R' : wordnet.ADV
        }
    
    def lower(self):
        self._lowered = True
        self.reports = self.reports.apply(lambda x: x.lower() if x is not np.nan else x)
        return self
    
    def remove_names(self, names : pd.Series):
        def _remove_names(row):
            if row[self.report_name] is np.nan:
                return row[self.report_name]
            report = row[self.report_name]
            for part in row['Name'].split(' '):
                part = part.lower() if self._lowered else part
                report = re.sub(
                    rf"{part}[\w']*", # patttern
                    '',               # replacement
                    report            # text
                )
            return report

        self.reports = pd.concat(
            (self.reports, names),
            axis=1
        ).apply(
            _remove_names,
            axis=1
        )
        return self
    
    def remove_whitespace(self):
        self.reports = self.reports.apply(lambda x: re.sub('\n', ' ', x) if x is not np.nan else x)
        self.reports = self.reports.apply(lambda x: re.sub('\r', '', x) if x is not np.nan else x)
        self.reports = self.reports.apply(lambda x: re.sub(' +', ' ', x) if x is not np.nan else x)
        return self
    
    def tokenize_text(self):
        self.tokens = self.reports.apply(lambda x: word_tokenize(x) if x is not np.nan else x)
        return self

    def remove_stopwords(self, hockey_words=None):
        if hockey_words is None:
            hockey_words = []

        if self.tokens is None:
            self.tokenize_text()

        self.tokens = self.tokens.apply(
            lambda x: [t for t in x 
                        if (t not in self.stop_words)
                            and (t not in hockey_words)]
                      if x is not np.nan else x
        )
            
        return self
    
    def remove_words(self, words):
        r"""
        This method removes any set of words from the text.

        Parameters
        ----------
        words : list-like
            The words to be removed.
        """

        if self.tokens is None:
            self.tokenize_text()

        self.tokens = self.tokens.apply(
            lambda x: [t for t in x
                       if (t not in words)]
                      if x is not np.nan else x
        )

        return self
    
    def normalize_words(self, normalization='porter'):
        r"""
        Parameters
        ----------
        normalization : {'porter', 'snowball', 'wordnet'}, default='porter'
            The normalization technique for each token.
        """
        if self.tokens is None:
            self.tokenize_text()

        # porter stemmer
        if normalization == 'porter':
            stemmer = PorterStemmer()
            self.tokens = self.tokens.apply(
                lambda x: [stemmer.stem(t) for t in x] 
                          if x is not np.nan else x
            )
        # snowball stemmer
        elif normalization == 'snowball':
            stemmer = SnowballStemmer()
            self.tokens = self.tokens.apply(
                lambda x: [stemmer.stem(t) for t in x]
                          if x is not np.nan else x
            )
        # wordnet lemmatizer
        elif normalization == 'wordnet':
            # TODO: lemmatizer also needs the part of speech (pos)
            #       otherwise it will default to nouns
            lemmatizer = WordNetLemmatizer()
            self.tokens = self.tokens.apply(
                lambda x: [lemmatizer.lemmatize(t) for t in x]
                          if x is not np.nan else x
            )
        else:
            ValueError(f"The normalization technique {normalization} is not supported.")

        return self
    
    def get_text(self):
        if self.tokens is not None:
            return self.tokens.apply(lambda x: ' '.join(x) if x is not np.nan else x)
        return self.reports



def preprocess(prospect_df):
    r"""
    Parameters
    ----------
    prospect_df : pandas.DataFrame
        The prospects data frame with raw scouting reports.

    Returns
    -------
    preprocessed_df : pandas.DataFrame
        The data frame with reports processed through NLTK methods.
    """

    # columns for the scouting reports
    mask = prospect_df.columns.str.match('Description')
    scouting_reports = prospect_df.columns[mask]
    
    preprocessed_df = prospect_df.copy()
    for report in scouting_reports:
        report_preprocessor = NltkPreprocessor(prospect_df[report])
        # preprocessed_df.loc[:,report] = report_preprocessor\
        #     .lower()\
        #     .remove_names(prospect_df['Name'])\
        #     .remove_whitespace()\
        #     .tokenize_text()\
        #     .remove_stopwords()\
        #     .normalize_words(normalization='porter')\
        #     .get_text()
        preprocessed_df.loc[:,report] = report_preprocessor\
            .remove_names(prospect_df['Name'])\
            .remove_whitespace()\
            .get_text()
        
    # merge all reports together
    preprocessed_df.loc[:,'all_reports'] = preprocessed_df[scouting_reports].apply(
        lambda row: ' '.join([row[i] for i in row.index if row[i] is not np.nan]),
        axis=1
    )

    return preprocessed_df



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file', help='the clean NHL prospects data file (.CSV)'
    )
    parser.add_argument(
        'output_file', help='the preprocessed NHL prospects data file (.CSV)'
    )
    args = parser.parse_args()

    clean_df = pd.read_csv(args.input_file)
    preprocessed_df = preprocess(clean_df)
    preprocessed_df.to_csv(args.output_file, index=False)
    