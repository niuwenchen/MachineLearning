#encoding:utf-8
#@Time : 2017/7/26 13:59
#@Author : JackNiu
'''
Minimal character-level vanilla rnn model
'''
import numpy as np

# data IO
data = open('input.txt','r').read()
chars = list(set(data))
data_size,vocab_size = len(data),len(chars)
print("data has %d characters,%d unique."%(data_size,vocab_size))
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

# hyperparmaters
hidden_size=100  # size of hidden layer of neurous
seq_length = 25 # number of steps to unroll the RNN for
learning_rate=1e-1


#model parameters
Wxh=np.random.randn(hidden_size,vocab_size)*0.01 #
Whh  = np.random.randn(hidden_size,hidden_size)*0.01
Why = np.random.randn(vocab_size,hidden_size)*0.01
bh = np.zeros((hidden_size,1))
by=np.zeros((vocab_size,1))

def sample(h, seed_ix, n):
    '''
    Sample a sequence of integers from the model , h is memory state
    seed_ix is seed  letter for first time step
    :param h:
    :param seed_ix:
    :param n:
    :return:
    '''
    x=np.zeros((vocab_size,1))
    x[seed_ix]=1
    ixes=[]
    for t in range(n):
        h=np.tanh(np.dot(Wxh,x)+np.dot(Whh,h)+bh)
        #y是输出结果
        y=np.dot(Whh,h)+by
        # 概率
        p=np.exp(y)/np.sum(np.eye(y))
        # ix说明:
        ix = np.random.choice(range(vocab_size),p=p.ravel())
        x=np.zeros((vocab_size,1))
        x[ix]=1
        ixes.append(ix)
    return ixes


def lossFun(inputs,targets,hprev):
    xs,hs,ys,ps={},{},{},{}
    hs[-1]=np.copy(hprev)
    loss=0
    for t in range(len(inputs)):
        xs[t]=np.zeors((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t]=np.tanh(np.dot(Wxh,xs[t])+np.dot(Whh,hs[t-1])+bh)
        ys[t]=np.dot(Why,hs[t])+by
        ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t]],0)
        # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy=np.copy(ps[t])
        dy[targets[t]] -=1
        dWhy += np.dot(dy.hs[t].T)
        dby += dy
        dh=np.dot(Why.T,dy)
        dhraw=(1-hs[t]*hs[t].T)
        dbh +=dhraw
        dWxh += np.dot(dhraw,xs[t].T)
        dWhh += np.dot(dhraw,hs[t-1].T)
        dhnext = np.dot(Whh.T,dhraw)

    for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
        np.clip(dparam,-5,5,out=dparam)
    return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]


# mainloop
n,p=0,0
mWxh,mWhh,mWhy= np.zeros_like(Wxh),np.zeros_like(Whh),np.zeros_like(Why)
mbh,mby = np.zeros_like(bh),np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length
while True:
    # prepare inputs
    if p+seq_length+1>=len(data) or n==0:
        hprev = np.zeros((hidden_size,1))
        p=0
    # inputs 是输入，targets 是预测
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    #sample  from the model now and then
    if n%100 ==0:
        sample_ix = sample(hprev,inputs([0]),200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n%s\n----'%(txt,))

    #
    loss,dWxh,dWhh,dWhy,dbh,dby,hprev=lossFun(inputs,targets,hprev)
    smooth_loss =smooth_loss*0.999+loss*0.001
    if n%100 ==0:print('iter %d, loss: %f' % (n, smooth_loss))
    for param,dparam,mem in zip([Wxh,Whh,Why,bh,by],
                                [dWxh,dWhh,dWhy,dbh,dby],
                                [mWxh,mWhh,mWhy,mbh,mby]):
        mem += dparam *dparam
        param += -learning_rate * dparam/np.sqrt(mem+ 1e-8)

    p+= seq_length
    n+=1


