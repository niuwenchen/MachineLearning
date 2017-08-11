#encoding:utf-8
#@Time : 2017/7/30 0:48
#@Author : JackNiu
import re

m = re.match(r'(\w+) (?P<aaa>\w+)(?P<sign>.*)','hello hangxiaoyang!')

print("m.string: ",m.string)
print('m.re:',m.re)
print('m.pos:',m.pos)
print('m.endpos:',m.endpos)
print('m.lastindex:',m.lastindex)
print('m.lastgroup:',m.lastgroup)

print('m.group(1,2)',m.group(1,2))
print('m.groups',m.groups())
print('m.start(2)',m.start(2))
# 给出别名的dict m.groupdict: {'aaa': 'hangxiaoyang', 'sign': '!'}
print('m.groupdict:',m.groupdict())
print('m.end(2)',m.end(2))
print('m.span(2):',m.span(2))
print(r"m.extend(r'\2 \1\3')",m.expand(r'\2 \1\3'))

p = re.compile(r'\d+')
print (p.split('one1two2three3four4'))

