from snorkel.labeling import labeling_function

ABSTAIN = -1
SUMMARY = 1
NOT_SUMMARY = 0

@labeling_function()
def is_short_sentence(x):
    return SUMMARY if len(x.text.split()) < 15 else ABSTAIN
