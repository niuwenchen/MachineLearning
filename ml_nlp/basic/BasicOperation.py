#encoding:utf-8
#@Time : 2017/7/29 22:59
#@Author : JackNiu
from collections import Counter
def get_max_value(text):
    count= Counter([x for x in text.lower() if x.isalpha()])
    print(count)
    m=max(count.values())
    print(m)
    print(sorted([x for (x,y)  in count.items() if y ==m]))
    return sorted([x for (x,y)  in count.items() if y ==m])[0]

def get_max_value_1(text):
    import string
    text = text.lower()
    print(string.ascii_lowercase)
    # abcdefghijklmnopqrstuvwxyz
    return max(string.ascii_lowercase,key=text.count)

# print(get_max_value_1("abcdeabcdabcaba"))
import re

pattern = re.compile(r'\d{3}[-\.]\w{2}',re.M)
match = pattern.match("123.he "
                        "123-HE"
                        "sdfsdf"
                        "123.he"
                        "")
if match:
    print(match.groupdict())

regex_1 = re.compile(r"""\d+
                        \.
                        \d+""",re.X)
regex_2=re.compile(r"\d+\.\d*")
