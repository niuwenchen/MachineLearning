#encoding:utf-8
#@Time : 2017/7/30 10:24
#@Author : JackNiu
import jieba
import jieba.analyse as analyse
analyse.set_stop_words('stop.txt')
lines=open('21.txt',encoding='utf-8').read()
# allowPOS 指定词性，形容词或名词之类的
print("  ".join(analyse.extract_tags(lines,topK=30,withWeight=False,allowPOS=())))

# 艺术  艺术创作  精品  创作  艺术作品  生产  优秀  实践  正确  作品  社会主义  繁荣  思想  政治  正确处理  演出  经济效益  坚持  精神  社会  艺术家  社会效益  文艺  人民  抓好  重视  作为  思想性  邓小平  保证

# 取出了我们这个停用词
