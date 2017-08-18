#encoding:utf-8
#@Time : 2017/7/30 9:27
#@Author : JackNiu
from __future__ import unicode_literals
import sys,os
sys.path.append('./')
from whoosh.index import create_in,open_dir
from whoosh.fields import  *
from whoosh.qparser import  QueryParser

from jieba.analyse import  ChineseAnalyzer

analyzer= ChineseAnalyzer()
#
schema = Schema(title=TEXT(stored=True),path=ID(stored=True),content = TEXT(stored=True,analyzer=analyzer))
if not  os.path.exists("tmp"):
    os.mkdir("tmp")

ix= create_in("tmp",schema=schema)
writer = ix.writer()

writer.add_document(title="document1",path="/a",content ="This is the first document we've added")
writer.add_document(
    title="document2",
    path="/b",
    content="The second one 你 中文测试中文 is even more interesting! 吃水果"
)

writer.add_document(
    title="document3",
    path="/c",
    content="买水果然后来世博园。"
)

writer.add_document(
    title="document4",
    path="/c",
    content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
)
writer.add_document(
    title="document4",
    path="/c",
    content="咱俩交换一下吧。"
)

writer.add_document(
    title="document5",
    path="/e",
    content="先生说我们都是好学生。"
)

writer.commit()
searcher = ix.searcher()
parser = QueryParser("content",schema=ix.schema)

for keyword in ("水果世博园","你","first","中文","交换机","交换","学生"):
    print("result of ",keyword)
    q= parser.parse(keyword)
    result = searcher.search(q)
    for hit in result:
        print(hit['title'],hit.highlights("content"))
    print("="*10)

for t in analyzer("我的好朋友是李明;我爱北京天安门;IBM和Microsoft; I have a dream. this is intetesting and interested me a lot"):
    print(t.text)