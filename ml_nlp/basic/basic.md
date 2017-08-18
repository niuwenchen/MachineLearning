## 基本技能
(1) 简单的排序，wordcount

    from collections import Counter
    def get_max_value(text):
        count= Counter([x for x in text.lower() if x.isalpha()])
        print(count)
        # Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1})
        m=max(count.values())
        print(m)
        # 5
        print(count.items())
        # dict_items([('e', 1), ('c', 3), ('b', 4), ('a', 5), ('d', 2)])
        return sorted([x for (x,y)  in count.items() if y ==m])[0]
        # sorted 返回一个new list
        
    
    string函数和max 结合
    def get_max_value_1(text):
    import string
    text = text.lower()
    print(string.ascii_lowercase)
    # abcdefghijklmnopqrstuvwxyz
    return max(string.ascii_lowercase,key=text.count)
    
    
正则比表达式

http://regexr.com


    通俗理解
    .   匹配任意字符，除\n换行符   a.c    abc
    \   转义字符，改变如上面.的意义 a\.c    a.c
    [...] 字符集,其中的字符可以逐个列出，也可以给出范围, [abc],[a-c] 
        [^abc] 不是abc的其他字符   a[bcd]e     abe ace ade
        所有的特殊字符在字符集中都失去了原有的特殊含义，在字符集中如果想要使用] -或^
        可以在前面加上\, 或把] -放在第一个字符，^放在非第一个字符
        a[-a^a]c   a-c   a^c

        
    数字\d [0-9]      a\dc    a1c
    非数字\D           a\Dc    abc
    空白字符 \s  [空格\t\r\n\f\v]  a\sc  a c
    非空白字符   \S  [^\s]   a\Sc    abc
    单词字符:\w 字母+数字  [A-Za-Z0-9]    a\wc  abc
    非单词字符\W   a\Wc   a c
    
    指定数字个基本字符   a[-a^a]{1,4}c   a-c  a^c    a^ac
    不指定个数
    *       匹配*前一位的字符0次或多次   abc*   ab  abcccc
    +       匹配+前一位字符1次或多次     abc+   abc   abccc
    ?       匹配?前一位字符0次或1次      abc?   ab    abc
    
    边界匹配
    ^   匹配字符串开头，比如判断text中是不是以xxx开头      ^abc   abcd中只会匹配abc
    $   匹配字符串末尾，给定的字符串                      abc$    abc
        
        ^W\w*e$ ^W\S+e$      Wxxxe  这样来，可以查询单词的匹配
    \A  仅匹配给定字符串作为开头    \Aabc   abc   \Aabc\w*   abcdadsa
    \Z  仅匹配字符串末尾            abc\Z       abc
    \b   匹配\w和\W之间             a\b!bc       a!bc
    
    逻辑 分组
    |     任意匹配一个，一旦匹配成功则跳过右边的表达式 abc(d+|e+)def
    (...) 分组，分组表达式表示一个整体，后接数量词
    allochieally  (...).*\1  \1 表示后面一定会出现前面匹配的三个字符
    
    
    练习 Regex  Golf
    
re模块

    import re

    pattern = re.compile(r'\d{3}[-\.]\w{2}')
    match = pattern.match("123.he 123-he")
    if match:
        print(match.group())
    
    re.complie(strpattern,flag)
    re.complie('pattern',re.l|re.M)== re.complie('(?im)pattern')
    flag可选的模式
        re.l(re.IGNORECASE): 忽略大小写
        re.M(MULTIPLU):多行模式，改变^和$的行为
        re.S(DOTALL)   点任意匹配模式
        re.L(LOCALE): 使预定字符类\w\W\b\B\s\S取决于当前区域设定
        re.u(UNICODE): 取决于unicode定义的字符属性
        re.X(VERBOSE),详细模式,正则表达式可以是多行，忽略空白字符，并可以加入注释。
        
        regex_1 = re.compile(r"""\d+
                        \.
                        \d+""",re.X)
        regex_2=re.compile(r"\d+\.\d*")
        
        pattern = re.compile(r'\d{3}[-\.]\w{2}',re.I)
        match = pattern.findall("123.he "
                        "123-HE"
                        "sdfsdf"
                        "123.he"
                        "")
        #  ['123.he', '123-HE', '123.he']
    
    
    match对象
    group()获得一个或多个分组的字符串
    
http://www.imooc.com/article/details/id/1141

    分组后就可以使用组的编号了(...)　分组的(?:...) 不分组的
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
    
    output
    m.string:  hello hangxiaoyang!
    m.re: re.compile('(\\w+) (?P<aaa>\\w+)(?P<sign>.*)')
    m.pos: 0
    m.endpos: 19
    m.lastindex: 3
    m.lastgroup: sign
    m.group(1,2) ('hello', 'hangxiaoyang')
    m.groups ('hello', 'hangxiaoyang', '!')
    m.start(2) 6
    m.groupdict: {'aaa': 'hangxiaoyang', 'sign': '!'}
    m.end(2) 18
    m.span(2): (6, 18)
    m.extend(r'\2 \1\3') hangxiaoyang hello!
    
    p = re.compile(r'\d+')
    print (p.split('one1two2three3four4'))
    # ['one', 'two', 'three', 'four', '']
    
    
结巴分词

    seg_list=jieba.cut("他毕业于上海交通大学，在百度深度学习研究院进行研究学习")
    print('/'.join(seg_list))
    # 他/毕业/于/上海交通大学/，/在/百度/深度/学习/研究院/进行/研究/学习
    seg_list=jieba.cut_for_search("他毕业于上海交通大学，在百度深度学习研究院进行研究学习")
    print('/'.join(seg_list))
    # 他/毕业/于/上海/交通/大学/上海交通大学/，/在/百度/深度/学习/研究/研究院/进行/研究/学习
    
    全模式: 会识别词包
    精确模式: 只是加隔板，不会找出词包
    
    添加用户自定义词典
    需要针对自己的场景进行分词，会有一些邻域内的专有词汇
        可以用jieba.load_userdict(file_name)
        少量的词汇可以手动添加
            add_word(word,freq=None,tag=None)
            del_word(word)
            
            suggest_freq(segment,tune=Tune) 可调节单个词语的词频，使其能被分出来。
            但是有的却是不能拆分
            jieba.suggest_freq(("今天","天气"),True)
            terms = jieba.cut('今天天气不错',HMM=False)
            print('/'.join(terms))
    
    字典格式:
        词典格式和 dict.txt 一样，一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，
        顺序不可颠倒。file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。
        
        自定义词典
        ﻿云计算 5
        李小福 2 nr
        创新办 3 i
        easy_install 3 eng
        好用 300
        韩玉赏鉴 3 nz
        八一双鹿 3 nz
        台中
        凱特琳 nz
        Edu Trust认证 2000
    
    
    HMM=False,自己添加用户词典必须设计HMM 为false
    
    print('/'.join(jieba.cut('如果放到旧字典中将出错。',HMM=False)))
    # 如果/放到/旧/字典/中将/出错/。
    # 将连在一起的分成两个词
    jieba.suggest_freq(('中','将'),True)
    print('/'.join(jieba.cut('如果放到旧字典中将出错。',HMM=False)))
    #如果/放到/旧/字典/中/将/出错/。
    jieba.add_word('旧字典')
    print('/'.join(jieba.cut('如果放到旧字典中将出错。',HMM=False)))
    # 如果/放到/旧字典/中/将/出错/。
    
    
    停用词:  必须是字典格式
    
    
关键词提取

计算所词性说明: 

    基于Tf-IDF算法的关键词抽取
    TF: 在该文章中出现的次数
    IDF: 逆向文件频率
    
        import jieba.analyse as analyse
        lines=open('NBA.txt').read()
        # allowPOS 指定词性，形容词或名词之类的
        print("  ".join(analyse.extract_tags(lines,topK=20,withWeight=False,allowPOS=())))
    
    关键词使用的逆向文件频率(IDF文本语料库可以切换成自定义语料库的路径
        Jieba.analyse.set_idf_path(file_name) # 自定义语料库
        关键词所使用的停用次stop_words
        jieba.analyse.set_stop_words(file_name)
        

基于TextRank算法的关键词抽取

    jieba.analyse.textrank()
    print(" ".join(analyse.textrank(lines,topK=20,withWeight=False,allowPOS=('ns','n','vn','v')
                                )))
                    
词性标注

    jieba.posseg.POSTTokenizer() 新建自定义分词器，可指定内部使用的Tokenizer分词器。
    
    import jieba.posseg as pseg
    words = pseg.cut('我爱自然语言处理')
    for word,flag in words:
        print(word,flag)
    
    我 r
    爱 v
    自然语言 l
    处理 v
    
并行分词

    jieba.enable_parallel(4)
    

chineseAnalyzer for whooser搜索引擎
Whoosh是一个索引文本和搜索文本的类库，他可以为你提供搜索文本的服务，比如如果你在创建一个博客的软件，你可以用whoosh为它添加添加一个搜索功能以便用户来搜索博客的入口

    可以在文档中检索某些词
    
    result of  水果世博园
    买<b class="match term0">水果</b>然后来<b class="match term1">世博园</b>

        for hit in result:
        print(hit['title'])
    
    这个工具主要使用的意义是: 去整个文档库中去检索出现的词语
    content= content.replace("\r\n","").strip() #删除换行和多余的空格
    
加载用户字典

    其他词典

    占用内存较小的词典文件 https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.small

    支持繁体分词更好的词典文件 https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big

下载你所需要的词典，然后覆盖 jieba/dict.txt 即可；或者用 jieba.set_dictionary('data/dict.txt.big')


工具
汉语分词系统  http://ictclas.nlpir.org/
python -m jieba [options] filename
