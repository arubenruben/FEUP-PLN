import os.path

lexicons = {}


def load_lexicons():
    with open(os.path.join('dataset', 'lexicons', 'sentilex.txt'), "r") as fp:
        for line in fp.readlines():
            index_first_comma = line.find(',')
            index_first_dot = line.find('.')

            first_polarity = 0

            index_first_polarity_value = line.find('N0=')

            if index_first_polarity_value != -1:
                index_first_semicolon = line.find(';', index_first_polarity_value + 1)
                first_polarity = int(line[index_first_polarity_value + 3:index_first_semicolon])

            index_second_polarity_value = line.find('N1=')

            if index_second_polarity_value != -1:
                index_second_semicolon = line.find(';', index_second_polarity_value + 1)
                second_polarity = int(line[index_second_polarity_value + 3:index_second_semicolon])
            else:
                second_polarity = first_polarity

            first_word = line[0:index_first_comma]
            second_word = line[index_first_comma + 1:index_first_dot]

            lexicons[first_word] = {
                'word': first_word,
                'polarity': first_polarity
            }

            lexicons[second_word] = {
                'polarity': second_polarity
            }


def get_polarity(word):
    return lexicons[word]['polarity']
