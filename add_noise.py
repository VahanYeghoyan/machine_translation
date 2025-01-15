# makeing some changes

import pickle
import random

with open("/home/vahan/Documents/machine_translation/similar_sounding_chars.pkl", "rb") as f:
    similar_sounding_chars = pickle.load(f)


class DataAugmentor:
    
    def __init__(self, similar_sounding_chars):    
        self.similar_sounding_chars = similar_sounding_chars


    def space_removal(self, sentence):
        num_words = len([i for i in sentence.split() if i not in {",", "։"}])

        if num_words < 5:
            num_spaces_to_remove = 1
        elif num_words >= 5 and num_words < 10:
            num_spaces_to_remove = 2
        elif num_words >= 10 and num_words < 15:
            num_spaces_to_remove = 3
        else:
            num_spaces_to_remove = 4

        space_indices = [i for i, char in enumerate(sentence) if char == ' ']
        replace_indices = [random.choice(space_indices) for _ in range(num_spaces_to_remove)]

        while len(replace_indices) != len(set(replace_indices)):
            replace_indices = [random.choice(space_indices) for _ in range(num_spaces_to_remove)]

        new_sentence = ""
        for idx, ch in enumerate(sentence):
            if idx in replace_indices:
                new_sentence += ""
            else:
                new_sentence += ch

        return new_sentence
    
    def random_deletion(self, sentence):
        char_list = list(sentence)
        num_words = len([i for i in sentence.split() if i not in {",", "։"}])
        
        if num_words <= 3:
            rand_index = random.randint(0, len(char_list) - 1)
            while char_list[rand_index] == ' ':
                rand_index = random.randint(0, len(char_list) - 1)

            del char_list[rand_index]
            return ''.join(char_list)      
            
        min_char_to_delete = int(0.2*num_words)
        max_char_to_delete = int(0.4*num_words)   
        num_char_to_delete = random.randint(min_char_to_delete, max_char_to_delete)
        
        for _ in range(num_char_to_delete):
            rand_index = random.randint(0, len(char_list) - 1)
            while char_list[rand_index] == ' ':
                rand_index = random.randint(0, len(char_list) - 1)

            del char_list[rand_index]
        return ''.join(char_list)
    
    def random_insertion(self, sentence):
        char_list = list(sentence)
        char_indecies = [i for i, j in enumerate(sentence) if j not in  {",", "։", " "}]
        num_words = len([i for i in sentence.split() if i not in {",", "։"}])   
        if num_words <= 3:
            char_list.insert(random.choice(char_indecies), random.choice('աբգդեզէըթժիլխծկհձղճմյնշոչպջռցւփքևօֆ'))
            return ''.join(char_list)
            
        min_char_to_insert = int(0.2*num_words)
        max_char_to_insert = int(0.4*num_words)   
        num_char_to_insert = random.randint(min_char_to_insert, max_char_to_insert)
        
        for _ in range(num_char_to_insert):
            char_list.insert(random.choice(char_indecies), random.choice('աբգդեզէըթժիլխծկհձղճմյնշոչպջռցւփքևօֆ'))
            
        return ''.join(char_list)
    
    
    def random_change(self, sentence):
        num_words = len([i for i in sentence.split() if i not in {",", "։"}])
        if num_words <= 3:
            char_indices = [i for i, char in enumerate(sentence) if char in similar_sounding_chars]

            if not char_indices:
                return sentence

            change_indices = random.sample(char_indices, 1)
            new_sentence = ""
            for idx, ch in enumerate(sentence):
                if idx in change_indices:
                    similar_chars = similar_sounding_chars.get(ch, [ch])
                    
                    new_sentence += random.choice(similar_chars)
                else:
                    new_sentence += ch

            return new_sentence


        char_indices = [i for i, char in enumerate(sentence) if char in similar_sounding_chars]
        if not char_indices:
            return sentence

        min_char_to_replace = int(0.2 * num_words)
        max_char_to_replace = int(0.4 * num_words)


        num_char_to_change = random.randint(min_char_to_replace, max_char_to_replace)
        change_indices = random.sample(char_indices, min(num_char_to_change, len(char_indices)))

        new_sentence = ""
        for idx, ch in enumerate(sentence):
            if idx in change_indices:
                similar_chars = similar_sounding_chars.get(ch, [ch])
                
                new_sentence += random.choice(similar_chars)
            else:
                new_sentence += ch

        return new_sentence
    
    def perform_augmentation(self, sentences):
        corrupted_data = {}
        for original_sentence in sentences:
            cnt = 0
            for i in original_sentence.split():
                if i not in {",", "։"}:
                    cnt += 1
            if cnt <= 2:
                corrupted = self.random_insertion(original_sentence)
                corrupted = self.random_change(corrupted)
            else:
                corrupted = self.random_insertion(original_sentence)
                corrupted = self.random_change(corrupted)
                corrupted = self.random_deletion(corrupted)
                corrupted = self.space_removal(corrupted)
            corrupted_data[corrupted] = original_sentence
        return corrupted_data
