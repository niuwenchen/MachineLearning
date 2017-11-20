##Web 前端基础
1 HTTP标准

    HTTP头部信息
    Accept	*/*
    Accept-Encoding	    gzip, deflate
    Accept-Language	    zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3
    Connection	    keep-alive
    Content-Length	    487
    Content-Type	    application/json
    Cookie	    Hm_lvt_6bcd52f51e9b3dce32bec4a3997715ac=1505721402,1505721413,1505727057,1505747474; uuid_tt_dd=-5906265661481595238_20170615; UN=u010651137; UE="540051856@qq
                .com"; BT=1505699691374; UM_distinctid=15d1b955e2e5be-02fc92fb2143738-12646f4a-100200-15d1b955e2f4bb
                ; __utma=17226283.1753235598.1500455447.1500455447.1500455447.1; __utmz=17226283.1500455447.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; 
                Hm_lpvt_6bcd52f51e9b3dce32bec4a3997715ac=1505747474; dc_tos=owhe6v; dc_session_id=1505747479802_0.6598235450574795
    Host	flume.csdn.net
    Origin	http://blog.csdn.net
    Referer	http://blog.csdn.net/
    User-Agent	    Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0
    
    
    User-Agent: 里面包含发出请求的用户信息，其中有使用的浏览器型号、版本和操作系统的信息。这个头域经常用来作
    为反爬虫的措施。
    
Cookie状态管理

    Cookie和Session都用来保存状态信息，但是爬虫开发更加关系cookie，因为cookie保存在客户端，而session保存在服务端。
    Cookie是服务器在本地机器上存储的小段文本并随每一个请求发送到同一个服务器。网络服务器用HTTP向客户端发送Cookie，客户端就会解析这些Cookie并将他们保存为一个本地文件，会自动将同一服务的任何请求绑定上这些cokie
    
    Cookie的工作方式： 服务器给每个Session分配惟一一个JSESSIONID，并通过Cookie发送给客户端，当客户端发起新的请求的时候，将在Cookie头中携带这个JSESSIONID。这样服务器就能够找到这个客户端对应的Session
    实际上： Cookie -- JSESSIONID--> SESSION
    