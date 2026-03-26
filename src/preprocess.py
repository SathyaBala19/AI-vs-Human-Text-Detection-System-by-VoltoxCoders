import re

# i handled the text cleaning part for our group project
# the idea is to make all the text look uniform before passing it to the model
# learned most of this from the sklearn documentation and a couple of medium articles

def clean_text(text):

    if not isinstance(text, str):
        return ""

    # lowercase everything first
    text = text.lower()

    # get rid of any http or www links in the text
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)

    # remove punctuation - only keep letters digits and whitespace
    # spent a while figuring out this regex, [^\w\s] means anything thats not word char or space
    text = re.sub(r'[^\w\s]', ' ', text)

    # fix multiple spaces that appear after removing stuff
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# applies clean_text to a whole pandas column
def preprocess_series(col):
    return col.apply(clean_text)


# ran this manually a few times to check if cleaning works right
if __name__ == "__main__":
    t1 = "HELLO World!! visit http://google.com for more info"
    t2 = "  too   many    spaces    here  "
    t3 = "don't won't - human writing has contractions"

    print("test 1:", clean_text(t1))
    print("test 2:", clean_text(t2))
    print("test 3:", clean_text(t3))
