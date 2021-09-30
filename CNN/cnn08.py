import numpy as np
import struct
from scipy.stats import truncnorm

def load_labels(file):   # 加载数据
    with open(file, "rb") as f :
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)

def load_images(file):   # 加载数据
    with open(file,"rb") as f :
        data = f.read()
    magic_number , num_items, rows, cols = struct.unpack(">iiii",data[:16])
    return  np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items,-1)

def make_onehot(arrayx,class_num = 10):
    data_lens = len(arrayx)
    result = np.zeros((data_lens,class_num),dtype='float32')
    for index,num in enumerate(arrayx):
        result[index][num] = 1
    return result

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis = 1, keepdims = True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return 2 * sigmoid(2*x) - 1

def L2NormPartial(l, theta):
    return theta * l

class MyDataSet():
    def __init__(self,imgdata,imglabel,batch_size,shuffle):
        self.imgdata = imgdata
        self.imglabel = imglabel
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.imgdata)

    def __iter__(self):
        return MyDataLoader(self)

class MyDataLoader():
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        else:
            indexs = np.arange(len(self.dataset))
            if self.dataset.shuffle:
                np.random.shuffle(indexs)
            index = indexs[self.cursor:self.cursor+self.dataset.batch_size]
            X = self.dataset.imgdata[index]
            Y = self.dataset.imglabel[index]
            self.cursor += self.dataset.batch_size
            return X,Y

class Module:
    def __init__(self, name, mode='train'):
        self.name = name
        self.mode = mode

    def __call__(self,*args):
        return self.forward(*args)

class Parameter():
    def __init__(self,vlaue):
        self.value = vlaue
        self.grad = 0


class Initializer:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        return self.apply(*args)


class GaussInitializer(Initializer):
    def __init__(self, mean, std):
        super().__init__("Gauss")
        self.mean = mean
        self.std = std

    def apply(self,value):
        value[...] = np.random.normal(self.mean, self.std, value.shape)
        return value

class XavierInitializer(Initializer):
    def __init__(self,low,upp,fan_in,fan_out):
        super().__init__("Xavier")
        mean = 0
        std = np.sqrt(2 / (fan_in + fan_out))
        self.tn = get_truncated_normal(mean,std,low,upp)

    def apply(self,value):
        arrtn = self.tn.rvs(value.size)
        value[...] = np.array(arrtn).reshape(value.shape)
        return value

class KaimingInitializer(Initializer):
    def __init__(self,low,upp,fan_in):
        super().__init__("Kaiming")
        mean = 0
        std = np.sqrt(2 / (fan_in))
        self.tn = get_truncated_normal(mean,std,low,upp)

    def apply(self,value):
        arrtn = self.tn.rvs(value.size)
        value[...] = np.array(arrtn).reshape(value.shape)
        return value

class LecunInitializer(Initializer):
    def __init__(self,low,upp,fan_in):
        super().__init__("Lecun")
        mean = 0
        std = np.sqrt(1 / (fan_in))
        self.tn = get_truncated_normal(mean,std,low,upp)

    def apply(self,value):
        arrtn = self.tn.rvs(value.size)
        value[...] = np.array(arrtn).reshape(value.shape)
        return value

class Sigmoid(Module):
    def __init__(self):
        super().__init__("Sigmoid")

    def forward(self,x):
        self.result = sigmoid(x)
        return self.result

    def backward(self,G):
        return G * (self.result) * (1 - self.result)

class Tanh(Module):
    def __init__(self):
        super().__init__("Tanh")

    def forward(self,x):
        self.result = tanh(x)
        return self.result

    def backward(self,G):
        return G * (1-self.result ** 2)

class Softmax(Module):
    def __init__(self):
        super().__init__("Softmax")

    def forward(self,x):
        return softmax(x)

    def backward(self,G):
        return G

class ReLU(Module):
    def __init__(self):
        super().__init__("ReLU")

    def forward(self,x):
        self.negative = x < 0
        x[self.negative] = 0
        return x

    def backward(self,G):
        G[self.negative] = 0
        return G

class PReLU(Module):
    def __init__(self , exp = 0.1):
        super().__init__("PReLU")
        self.exp = exp

    def forward(self,x):
        self.negative  = x < 0
        x[self.negative] *= self.exp

        return x

    def backward(self,G):
        G[self.negative]  *= self.exp

        return G

class Linear(Module):
    def __init__(self,size = ()):
        super().__init__("Linear")
        self.weight = Parameter(np.random.normal(0,np.sqrt(2 / size[0]),size=size))
        self.bias = Parameter(np.zeros((1,size[1])))

    def forward(self,x):
        self.x = x
        return self.x @ self.weight.value + self.bias.value

    def backward(self,G):

        self.weight.grad += self.x.T @ G
        self.bias.grad += np.sum(G,axis = 0 , keepdims = True)

        delta_x = G @ self.weight.value.T

        #self.weight -= lr * self.weight.grad
        #self.bias -= lr * self.bias.grad

        return delta_x

class Conv2d(Module):
    def __init__(self,initializer,out_channel,in_channel,kernel_size=(),stride = 1):
        super().__init__("Conv2d")
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_h,self.kernel_w = kernel_size
        self.stride = stride
        self.kernel = Parameter(initializer(np.empty((out_channel,in_channel,*kernel_size))))

    def forward(self,X):
        img_n,img_c,img_h,img_w = X.shape
        self.x_shape = X.shape

        self.c_h = (img_h - self.kernel_h)//self.stride + 1
        self.c_w = (img_w - self.kernel_w)//self.stride + 1

        self.col = self.kernel.value.reshape(self.out_channel,-1)

        self.column = np.zeros((img_n,self.col.shape[1],self.c_w * self.c_h))

        self.bias = Parameter(np.zeros((self.out_channel,self.c_h*self.c_w)))

        c = 0

        for h in range(self.c_h):
            for w in range(self.c_w):
                self.column[:,:,c] = X[...,h*self.stride:h*self.stride+self.kernel_h,w*self.stride:w*self.stride+self.kernel_w].reshape(img_n,-1)
                c+=1

        result = self.col @ self.column + self.bias.value

        return result.reshape(img_n,self.out_channel,self.c_h,self.c_w)

    def backward(self,G):
        G = G.reshape(G.shape[0],self.out_channel,-1)
        delta_col = np.sum(G @ self.column.transpose(0,2,1),axis=0)
        self.kernel.grad = delta_col.reshape(self.out_channel,self.in_channel,self.kernel_h,self.kernel_w)
        self.bias.grad = np.sum(G,axis=0)
        delta_column = self.col.T @ G
        back_G = np.zeros(self.x_shape)

        c = 0

        for h in range(self.c_h):
            for w in range(self.c_w):
                back_G[..., h*self.stride:h*self.stride + self.kernel_h, w*self.stride:w*self.stride + self.kernel_w] = delta_column[...,c].reshape(G.shape[0],self.in_channel,self.kernel_h,self.kernel_w)
                c+=1
        return back_G

class MaxPooling(Module):
    def __init__(self, pool_size, stride):
        super().__init__("MaxPool2d")
        self.pool_size = pool_size
        self.stride = stride
        self.flag = True

    def forward(self, x):
        self.x_shape = x.shape
        self.p_h = (x.shape[-2] - self.pool_size) // self.stride + 1
        self.p_w = (x.shape[-1] - self.pool_size) // self.stride + 1

        pool = np.zeros((*x.shape[:2], self.p_h, self.p_w))
        self.argmax = np.zeros_like(pool,dtype='int')

        for h in range(self.p_h):
            for w in range(self.p_w):
                h_, w_ = h * self.stride, w * self.stride
                pool[...,h,w] = np.max(x[...,h_:h_+self.pool_size,w_:w_+self.pool_size],axis= (2,3))
                self.argmax[...,h,w] = np.argmax(x[..., h_:h_ + self.pool_size, w_:w_ + self.pool_size].reshape(-1,self.pool_size * self.pool_size), axis=(1)).reshape(*x.shape[:2])
        return pool

    def backward(self,G):
        new_g = np.zeros(self.x_shape)
        for h in range(self.p_h):
            for w in range(self.p_w):
                h_, w_ = h * self.stride, w * self.stride
                multi = (np.arange(G.shape[0]*G.shape[1])*(self.pool_size*self.pool_size)).reshape(*G.shape[:2])
                index = np.unravel_index(multi+self.argmax[...,h,w],(*G.shape[:2],self.pool_size,self.pool_size))
                new_g[...,h_:h_+self.pool_size,w_:w_+self.pool_size][index] = G[...,h,w]
        return new_g

class BatchNorm2d(Module):
    def __init__(self, in_channel, eps=1e-05, momentum=0.5):
        super().__init__("BatchNorm2d")
        self.in_channel = in_channel
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones((in_channel, 1)))
        self.beta = Parameter(np.zeros((in_channel, 1)))
        self.running_mean = 0
        self.running_var = 0

    def forward(self, x):
        x = x.transpose(1, 0, 2, 3)
        self.x_shape = x.shape
        self.x = x.reshape(self.in_channel, -1)

        if self.mode == 'train':
            self.x_mean = np.mean(self.x, axis=1, keepdims=True)
            self.x_var = np.var(self.x, axis=1, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.x_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.x_var
            self.x_hat = (self.x - self.x_mean) / np.sqrt(self.x_var + self.eps)
        elif self.mode == "test":
            self.x_hat = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % self.mode)

        x = self.gamma.value * self.x_hat + self.beta.value
        x = x.reshape(self.x_shape).transpose(1, 0, 2, 3)
        return x

    def backward(self, G):
        G = G.transpose(1, 0, 2, 3).reshape(self.in_channel, -1)

        # ------------------------------ BatchNorm2d backward -----------------------------------
        delta_gamma = np.sum(G * self.x_hat, axis=1, keepdims=True)
        delta_beta = np.sum(G, axis=1, keepdims=True)

        delta_x_hat = self.gamma.value * G
        if self.mode == 'train':
            delta_x_var = -0.5 * np.sum(delta_x_hat * (self.x - self.x_mean), axis=1, keepdims=True) * np.power(self.x_var + self.eps, -1.5)
            delta_x_mean = -np.sum(delta_x_hat / np.sqrt(self.x_var + self.eps), axis=1, keepdims=True) - 2.0 * delta_x_var * np.sum(self.x - self.x_mean, axis=1, keepdims=True) / G.shape[1]
            delta_x = delta_x_hat / np.sqrt(self.x_var + self.eps) + (2.0 * delta_x_var * (self.x - self.x_mean) + delta_x_mean) / G.shape[1]
        elif self.mode == 'test':
            delta_x_var = -0.5 * np.sum(delta_x_hat * (self.x - self.running_mean), axis=1, keepdims=True) * np.power(self.running_var + self.eps, -1.5)
            delta_x_mean = -np.sum(delta_x_hat / np.sqrt(self.running_var + self.eps), axis=1, keepdims=True) - 2.0 * delta_x_var * np.sum(self.x - self.running_mean, axis=1, keepdims=True) / G.shape[1]
            delta_x = delta_x_hat / np.sqrt(self.running_var + self.eps) + (2.0 * delta_x_var * (self.x - self.running_mean) + delta_x_mean) / G.shape[1]
        self.gamma.grad += delta_gamma
        self.beta.grad += delta_beta

        delta_x = delta_x.reshape(self.x_shape).transpose(1, 0, 2, 3)
        return delta_x


class Flatten(Module):
    def __init__(self):
        super().__init__("Flatten")

    def forward(self,X):
        self.x_shape = X.shape
        return X.reshape(X.shape[0],-1)

    def backward(self,G):
        return G.reshape(self.x_shape)

class Padding(Module):
    def __init__(self,pad_size):
        super().__init__("Padding")
        self.pad_size = pad_size

    def forward(self,x):
        batch,out,h,w = x.shape
        pad = np.zeros((batch,out,h+self.pad_size,w+self.pad_size))
        pad[...,:h,:w] = x
        return pad

    def backward(self,G):
        return G[...,:(G.shape[2]-self.pad_size),:(G.shape[3]-self.pad_size)]

class Optimizer():
    def __init__(self,params,lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

class SGD(Optimizer):
    def __init__(self,params,lr = 0.01):
        super().__init__(params,lr)

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad

class L2SGD(Optimizer):
    def __init__(self,params,lr = 0.01,r=0.001):
        super().__init__(params,lr)
        self.r = r

    def step(self):
        for param in self.params:
            #bias不需要正则 懒得改了
            param.value -= self.lr * param.grad + L2NormPartial(self.r,param.value)

class MomentumSGD(Optimizer):
    def __init__(self,params,lr = 0.01,momentum = 0.9):
        super().__init__(params, lr)
        self.momentum = momentum

        for param in self.params:
            param.pre_delta = 0

    def step(self):
        for param in self.params:
            delta = param.pre_delta * self.momentum - self.lr * param.grad *(1 - self.momentum)
            param.value += delta

            param.pre_delta = delta

class Adam(Optimizer):
    def __init__(self,params,lr=0.001,beta1 = 0.9,beta2 = 0.999, e = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e

        self.t = 0

        for param in self.params:
            param.mt = 0
            param.vt = 0

    def step(self):
        self.t += 1
        for param in self.params:
            gt = param.grad
            param.mt = self.beta1 * param.mt + (1-self.beta1) * gt
            param.vt = self.beta2 * param.vt + (1-self.beta2) * (gt ** 2)
            mt_ = param.mt / ( 1- self.beta1 ** self.t)
            vt_ = param.vt / ( 1- self.beta2 ** self.t)

            param.value -= self.lr * mt_ / (np.sqrt(vt_) + self.e)


class Dropout(Module):
    def __init__(self, drop_rate=0.5, inplace=True):
        super().__init__("Dropout")
        assert 0 < drop_rate <= 1
        self.drop_rate = drop_rate
        self.prob_keep = 1 - drop_rate
        self.inplace = inplace
        if self.mode == 'test':
            self.inplace = False

    def forward(self, x):

        self.mask = np.random.binomial(size=x.shape, p= self.prob_keep, n=1)
        if not self.inplace:
            x = x.copy()

        x *= self.mask
        x *= 1 / self.prob_keep
        return x

    def backward(self, G):
        if not self.inplace:
            G = G.copy()
        G *= self.mask
        G *= 1 / self.prob_keep
        return G

class Model(Module):
    def __init__(self,*args):
        self.layers = args

    def forward(self,x,label = None):
        self.label = label
        for layer in self.layers:
            x = layer(x)
        self.pre = x
        return x

    def backward(self):
        G = (self.pre - self.label)/len(self.label)
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    @property
    def predict(self):
        return np.argmax(self.pre,axis = 1)

    @property
    def modules(self):
        ms = []
        attrs = self.__dict__
        for att in attrs:
            value = attrs[att]
            if isinstance(value,Module):
                ms.append(value)
            elif "__iter__" in dir(value):
                for i in value:
                    if isinstance(i,Module):
                        ms.append(i)
                    else:
                        break
        return ms

    def get_parameters(self):
        ps = []
        for layers in self.modules:
            atts = layers.__dict__
            for att in atts:
                value = atts[att]
                if isinstance(value,Parameter):
                    ps.append(value)
                elif "__iter__" in dir(value):
                    for i in value:
                        if isinstance(i,Parameter):
                            ps.append(i)
                        else:
                            break
        return ps

    def __repr__(self):
        names = [m.name for m in self.modules]
        return "\n".join(names)


if __name__ == '__main__':
    train_data = load_images("E:\\ai\\9-7\\datas\\train-images.idx3-ubyte")[:10000]

    train_label = load_labels("E:\\ai\\9-7\\datas\\train-labels.idx1-ubyte")[:10000]

    train_label = make_onehot(train_label)

    test_data = load_images("E:\\ai\\9-7\\datas\\t10k-images.idx3-ubyte")[:2000]

    test_label = load_labels("E:\\ai\\9-7\\datas\\t10k-labels.idx1-ubyte")[:2000]

    standard = 255

    train_data = train_data/ standard

    test_data = test_data / standard

    lrmax = 0.5

    lrmin = 0.05

    epoch = 50

    batch_size = 50

    shuffle = True

    train_num, feature_num = train_data.shape
    test_num = test_data.shape[0]

    train_data = train_data.reshape(train_num, 1, 28, 28)
    test_data = test_data.reshape(test_num,1,28,28)
    dataset = MyDataSet(train_data,train_label,batch_size,shuffle)


    class_num = 10
    hidden_num1 = 128
    hidden_num2 = 64
    step_num = 1
    #gauss = GaussInitializer(0, 0.1)
    #xavier = XavierInitializer(0,0.1,1,5)
    kaiming = KaimingInitializer(0,0.1,1)
    #lecun = LecunInitializer(0,0.1,1)

    model = Model(
        Padding(pad_size=2),
        Conv2d(kaiming,out_channel=3,in_channel=1,kernel_size=(3,3)), #50*1*30*30 --> 50*3*28*28
        BatchNorm2d(3),
        ReLU(),
        MaxPooling(2, 2),
        Conv2d(kaiming,out_channel=27,in_channel=3,kernel_size=(3,3)),
        BatchNorm2d(27),
        ReLU(),
        MaxPooling(2,2),
        #Dropout(0.4),
        Flatten(),
        Linear(size=((27*6*6),600)),
        ReLU(),
        Linear(size=(600,10)),
        Softmax()
    )

    #print(model)

    for e in range(epoch):
        params = model.get_parameters()
        lr = lrmax + 1 / 2 * (lrmax - lrmin) * (1 + np.cos( e / 5 * np.pi )) #余弦退火
        # opt = SGD(params,lr)
        # opt = L2SGD(params, lr,r=0.001)
        opt = MomentumSGD(params, lr)
        # opt = Adam(params)

        for batch_index,(x,y) in enumerate(dataset):

            x = model(x, y)

            loss = -np.sum( y * np.log(x)  ) / batch_size

            model.backward()

            if (batch_index+1) % step_num == 0:

                opt.step()

                opt.zero_grad()

        model.mode = 'test'

        x = model(test_data)

        acc = np.sum(model.predict == test_label) / test_num * 100

        print(f"epoch:{e}, acc : {acc:.3f} % ,loss : {loss:.3f}")








