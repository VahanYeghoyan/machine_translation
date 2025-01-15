import re

class Preprocessing:
    def __init__(self, normalized_transcription, char_set = None, list_of_words = None):
        self.normalized_transcription = normalized_transcription
        #         self.new_sentences = set()
        self.new_sentences = list()
    def performe_preprocessing(self):
        pattern = r'.*\d.*'
        new_data = [sentence for sentence in self.normalized_transcription if not re.match(pattern, sentence)]
        self.normalized_transcription = new_data
        if " " in self.normalized_transcription:
            self.normalized_transcription.remove(" ")
        for sent in self.normalized_transcription:
            sent = sent.lower()
            new_sent = ""
            for i in sent:
                if ord(i) in range(ord("ա"), ord("և")+1) or i in {",", "։", " ", ":", ","}:
                    new_sent += i

            new_sent = new_sent.lower()

            if len(new_sent.strip())>1:
                if ":" in new_sent:
                    new_sent = new_sent.replace(":", "։")
                if new_sent[-1] == "։":
                    new_sent = new_sent.replace("։", "")
                new_sent = new_sent.replace("եվ", "և")
                new_sent = new_sent.replace("։", " ։")
                new_sent = new_sent.replace(",", ",")
                new_sent = new_sent.replace(",", " ,")
                new_sent = new_sent.replace(",", ", ")
                new_sent = " ".join(new_sent.split())

                self.new_sentences.append(new_sent)
        return self.new_sentences