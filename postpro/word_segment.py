import re

SPACE_TOKEN = ' '


def segment_word_by_tags(text, tags, delimiters=['b', 's']):
    if SPACE_TOKEN in text:
        text = text.replace(' ', '')

    if len(text) != len(tags):
        raise ValueError('length of both unspaced text and tag seq are not the same!')

    segmented_text = ''

    if len(text) > 1:
        for s, t in zip(text, tags):
            if t in delimiters or t.lower() in delimiters:
                segmented_text += ' '
            segmented_text += s

        segmented_text = segmented_text.strip()
        segmented_text = re.sub(' +', ' ', segmented_text)

    return segmented_text
