import os
import cv2
from matplotlib import pyplot as plt 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from pykalman import KalmanFilter


def get_peak(data, n_peak=2, size=5):
    tmp = 0
    cnt = 0
    value = []
    index = []
    for i in range(len(data)-size):
        peak = max(data[i: i+size])
        if peak==tmp: 
            cnt+=1
            if cnt >= size-1:
                value.append(peak)
                ind = list(data[i:i+size]).index(peak)
                index.append(ind+i)
        else:
            cnt =0
        tmp = peak
    if len(value)<n_peak:
        print('Only one peak.')
    value_sorted = np.array(sorted(value, reverse=True))
    value_ind = np.argsort(np.array(value))
    index_sorted = np.array(index)[value_ind][::-1]
    i=0
    while value_sorted[0]==value_sorted[i] and abs(index_sorted[0]-index_sorted[i])<size:
        i += 1
    return [value_sorted[0],value_sorted[i]], [index_sorted[0], index_sorted[i]]

def fit_lane(img, ind, peak, peak_ind):
    data=[[],[],[],[]]
    for x in range(ind, min(img.shape[0]-10, ind+100)):
        img[x, :][:4]=255
        img[x, :][-4:]=255
        p = 255-img[x, :]
        _, y = get_peak(p)
        if _[0]>20 and _[1]>20 and abs(y[0]-y[1])>120:
            data[0].append([x])
            data[1].append([min(y)])
            data[2].append([x])
            data[3].append([max(y)])
    
    if len(data[0])>30:
        poly_reg1 =PolynomialFeatures(degree=2)
        X_ploy =poly_reg1.fit_transform(data[0])
        regr1 = linear_model.LinearRegression()
        regr1.fit(X_ploy, data[1])
        y1 = round(regr1.predict(poly_reg1.fit_transform([[ind]]))[0][0])
    else:
        y1 = 0

    if len(data[2])>30:
        poly_reg2 = PolynomialFeatures(degree=2)
        X_ploy = poly_reg2.fit_transform(data[2])
        regr2 = linear_model.LinearRegression()
        regr2.fit(X_ploy, data[3])
        y2 = round(regr2.predict(poly_reg2.fit_transform([[ind]]))[0][0])
    else:
        y2 = 0 

    if y1==0 or y2==0:
        y1, y2 = peak_ind, peak_ind
    else:
        if abs(y1-peak_ind)<abs(y2-peak_ind): y1 = peak_ind
        else: y2 = peak_ind
    
    return [y1, y2]

def iou(box11, box22):
    box1 = [int(float(x)) for x in box11]
    box2 = [int(float(x)) for x in box22]
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou

def get_box(path):
    c = ['bike', 'people']
    l = [[], []]
    with open(path, 'r') as fr:
        for line in fr.readlines():
            line = line.strip().split()
            if line[0]==c[0]:   l[0].append(line)
            elif line[0]==c[1]: l[1].append(line)
    for k in range(2):
        repeat = []
        for i in range(len(l[k])):
            for j in range(i+1, len(l[k])):
                if iou(l[k][i][2:], l[k][j][2:]) > 0:
                    repeat.append(j)
    
        for i, n in enumerate(sorted(set(repeat))):
            del l[k][n-i]
    
    return l[0]+l[1]
            
def get_timesq(split='train'):
    yolo_path = '/home/zhangruiwen/01research/02toyota/03code/PyTorch-YOLOv3/output1/'+split
    scnn_path = '/home/zhangruiwen/01research/02toyota/03code/lanedetect/output_'+split

    save_path = '/home/zhangruiwen/01research/02toyota/03code/direction/'+split
    img_save_path = '/home/zhangruiwen/01research/02toyota/03code/img/'+split
    os.makedirs(save_path, exist_ok=True)
    for sq in sorted(os.listdir(yolo_path)):
        os.makedirs(os.path.join(img_save_path, sq), exist_ok=True)
        with open(os.path.join(save_path, sq+'.txt'), 'w') as fw:
            for n, imglist in enumerate(sorted(os.listdir(os.path.join(yolo_path, sq)))):
                img = cv2.imread(os.path.join(scnn_path, 'probmap'+sq, imglist.replace('.txt', '_0_avg.png')), 0)
                for line in get_box(os.path.join(yolo_path, sq, imglist)):            
                    ind = int(float(line[5]))
                    img[ind, :][:4]=255
                    img[ind, :][-4:]=255
                    p = 255-img[ind, :]
                    v, i = get_peak(p)
                    if v[1]<15 or abs(i[0]-i[1])<120:
                        i = fit_lane(img, ind, v[0], i[0])
                    fw.write(f'{n+1}\t{line[0]}\t{round(float(line[2]))}\t{round(float(line[4]))}\t{min(i)}\t{max(i)}\n')

                    print(imglist, v,i)
                    plt.scatter(i,v,c='black', label='peak')
                    plt.xlim([0,800])
                    plt.ylim([0,255])
                    plt.xlabel('pixel_w_coordinate')
                    plt.ylabel('pixel_value')
                    plt.title('Lane_img_width_pixel')
                    if line[0]=='people':
                        plt.plot(p, c='black')
                        plt.vlines(int(float(line[2])),0, 100, colors='r')
                        plt.vlines(int(float(line[4])),0, 100, colors='r')
                        plt.hlines(100, int(float(line[2])), int(float(line[4])), colors='r', label='people')
                    else:
                        plt.plot(p, c='y')
                        plt.vlines(int(float(line[2])),0, 70, colors='g')
                        plt.vlines(int(float(line[4])),0, 70, colors='g')
                        plt.hlines(70, int(float(line[2])), int(float(line[4])), colors='g', label='bike')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(img_save_path, sq, imglist.replace('.txt', '.png')))
                plt.close()

def smooth(data, step=7):
    for i in range(0, len(data)-step+1, 2):
        m, s = np.mean(sorted(data[i: i+step])[1:-1]), np.std(data[i: i+step])
        for j, v in enumerate(data[i: i+step]):
            if i+j==0:
                if abs(v-data[1])>50:
                    data[0] = data[1]
                    v = data[1]
            else:
                if abs(v-data[i+j-1])>50:
                    data[i+j] = data[i+j-1]
                    v = data[i+j-1]
            if abs(v-m)>3*s:
                data[i+j]=m
                # if s>60 or abs(v-m)>60:
                #     if i+j==1:
                #         data[i+j] = (data[i+j-1]+data[i+j+1])/2
                #     elif i+j==0:
                #         data[i+j] = (data[i+j+1]+data[i+j+2])/2
                #     else:
                #         data[i+j] = (data[i+j-1]+data[i+j-2])/2
                # else:
                #     data[i+j]=m
            
    return data


class get_location():
    def __init__(self, split='train', 
                    fpath='/home/zhangruiwen/01research/02toyota/03code/direction',
                    spath='/home/zhangruiwen/01research/02toyota/03code/direction/result'):
        self.file_path = os.path.join(fpath, split)
        self.flist = sorted(os.listdir(self.file_path))
        self.save_path = os.path.join(spath, split)
        self.save_path_txt = os.path.join(spath, split+'_txt')
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path_txt, exist_ok=True)

    def Kalman_F(self, data):
        m = np.mean(sorted(data[0][:6])[2:-2])
        kf = KalmanFilter(transition_matrices=np.array([[1,0], [0,1]]),
                  observation_matrices=np.array([[1,0],[0,1]]),
                  transition_covariance= 0.03*np.eye(2),
                  initial_state_mean=np.array([m, data[1][0]]))
        
        x = np.vstack((data[0], data[1]))
        y = kf.filter(x.T)[0]
        return [y[:,0], y[:,1]]

    def adjust(self, idata):
        img_num = np.argwhere(idata[1:,0]==0)[0][0]
        o_ind = int((idata.shape[1]-1)/3)
        o_data = np.sort(idata[1:, 1:5])[:, ::-1]
        l_data1 = np.sort(idata[1:, 5:9])[:, ::-1]
        l_data2 = np.sort(idata[1:, 9:13])[:, ::-1]
        x = np.zeros(o_data.shape[1])
        y = [[],[],[],[]]
        for i in range(o_data.shape[1]):
            y[i] = np.argwhere(o_data[:,i] !=0)
            x[i] = len(y[i])
        row = np.argwhere(x/img_num<0.25)[0][0]
        
        if row==1:
            ind = y[0].flatten()+1
            o_data = o_data[y[0].flatten(),:]
            for i in range(1, o_data.shape[0]):
                t = np.where(o_data[i,:]>0)[0]
                if len(t)>1:
                    min_ind = np.argmin(abs(o_data[i,t]-o_data[i-1,0]))
                    o_data[i,0] = o_data[i,min_ind]
            for i in range(0, o_data.shape[0]-1)[::-1]:
                t = np.where(o_data[i,:]>0)[0]
                if len(t)>1:
                    min_ind = np.argmin(abs(o_data[i,t]-o_data[i+1,0]))
                    o_data[i,0] = o_data[i,min_ind]

            l_data1 = l_data1[y[0].flatten(),:]
            for i in range(1, l_data1.shape[0]):
                t = np.where(l_data1[i,:]>0)[0]
                if len(t)>1:
                    min_ind = np.argmin(abs(l_data1[i,t]-l_data1[i-1,0]))
                    l_data1[i,0] = l_data1[i,min_ind]
            for i in range(0, l_data1.shape[0]-1)[::-1]:
                t = np.where(l_data1[i,:]>0)[0]
                if len(t)>1:
                    min_ind = np.argmin(abs(l_data1[i,t]-l_data1[i+1,0]))
                    l_data1[i,0] = l_data1[i,min_ind]

            l_data2 = l_data2[y[0].flatten(),:]
            for i in range(1, l_data2.shape[0]):
                t = np.where(l_data2[i,:]>0)[0]
                if len(t)>1:
                    min_ind = np.argmin(abs(l_data2[i,t]-l_data2[i-1,0]))
                    l_data2[i,0] = l_data2[i,min_ind]
            for i in range(0, l_data2.shape[0]-1)[::-1]:
                t = np.where(l_data2[i,:]>0)[0]
                if len(t)>1:
                    min_ind = np.argmin(abs(l_data2[i,t]-l_data2[i+1,0]))
                    l_data2[i,0] = l_data2[i,min_ind]
            l1 = self.Kalman_F([l_data1[:,0], idata[ind, 0]])[0].flatten()
            l2 = self.Kalman_F([l_data2[:,0], idata[ind, 0]])[0].flatten()
            o1 = self.Kalman_F([o_data[:,0], idata[ind, 0]])[0].flatten()
            lane_center = (l1+l2)/2
            lane_width = np.abs(l1-l2)
            return [[(o1-lane_center)/lane_width, idata[ind, 0]]]
        elif row==0:
            return []
        elif row==2:
            ind = y[0].flatten()+1
            o_data = o_data[y[0].flatten(),:]
            for i in range(1, o_data.shape[0]):
                t = np.where(o_data[i,:]>0)[0]
                if len(t)>1:
                    min_ind0 = np.argmin(abs(o_data[i,t]-o_data[i-1,0]))
                    min_ind1 = np.argmin(abs(o_data[i,t]-o_data[i-1,1]))
                    if min_ind0==min_ind1:
                        another_ind = list(set(t)-set([min_ind0]))[0]
                        if abs(o_data[i,min_ind0]-o_data[i-1,0])>abs(o_data[i,min_ind0]-o_data[i-1,1]):
                            tmp0, tmp1 = o_data[i,another_ind], o_data[i,min_ind1]
                        else:
                            tmp0, tmp1 = o_data[i,min_ind0], o_data[i,another_ind]
                        o_data[i,0], o_data[i,1] = tmp0, tmp1
                    else:
                        tmp0, tmp1 = o_data[i,min_ind0], o_data[i,min_ind1]
                        o_data[i,0], o_data[i,1] = tmp0, tmp1
                else:
                    min_ind = np.argmin(abs(o_data[i,t[0]]-o_data[i-1,:2]))
                    o_data[i, min_ind] = o_data[i,0]
            for i in range(0, o_data.shape[0]-1)[::-1]:
                t = np.where(o_data[i,:]>0)[0]
                if len(t)>1:
                    min_ind0 = np.argmin(abs(o_data[i,t]-o_data[i+1,0]))
                    min_ind1 = np.argmin(abs(o_data[i,t]-o_data[i+1,1]))
                    if min_ind0==min_ind1:
                        another_ind = list(set(t)-set([min_ind0]))[0]
                        if abs(o_data[i,min_ind0]-o_data[i+1,0])>abs(o_data[i,min_ind0]-o_data[i+1,1]):
                            tmp0, tmp1 = o_data[i,another_ind], o_data[i,min_ind1]
                        else:
                            tmp0, tmp1 = o_data[i,min_ind0], o_data[i,another_ind]
                        o_data[i,0], o_data[i,1] = tmp0, tmp1
                    else:
                        tmp0, tmp1 = o_data[i,min_ind0], o_data[i,min_ind1]
                        o_data[i,0], o_data[i,1] = tmp0, tmp1
                else:
                    min_ind = np.argmin(abs(o_data[i,t[0]]-o_data[i+1,:2]))
                    o_data[i, min_ind] = o_data[i,0]
            
            l_data1 = l_data1[y[0].flatten(),:]
            for i in range(1, l_data1.shape[0]):
                t = np.where(l_data1[i,:]>0)[0]
                if len(t)>1:
                    min_ind0 = np.argmin(abs(l_data1[i,t]-l_data1[i-1,0]))
                    min_ind1 = np.argmin(abs(l_data1[i,t]-l_data1[i-1,1]))
                    if min_ind0==min_ind1:
                        another_ind = list(set(t)-set([min_ind0]))[0]
                        if abs(l_data1[i,min_ind0]-l_data1[i-1,0])>abs(l_data1[i,min_ind0]-l_data1[i-1,1]):
                            tmp0, tmp1 = l_data1[i,another_ind], l_data1[i,min_ind1]
                        else:
                            tmp0, tmp1 = l_data1[i,min_ind0], l_data1[i,another_ind]
                        l_data1[i,0], l_data1[i,1] = tmp0, tmp1
                    else:
                        tmp0, tmp1 = l_data1[i,min_ind0], l_data1[i,min_ind1]
                        l_data1[i,0], l_data1[i,1] = tmp0, tmp1
                else:
                    min_ind = np.argmin(abs(l_data1[i,t[0]]-l_data1[i-1,:2]))
                    l_data1[i, min_ind] = l_data1[i,0]
            for i in range(0, l_data1.shape[0]-1)[::-1]:
                t = np.where(l_data1[i,:]>0)[0]
                if len(t)>1:
                    min_ind0 = np.argmin(abs(l_data1[i,t]-l_data1[i+1,0]))
                    min_ind1 = np.argmin(abs(l_data1[i,t]-l_data1[i+1,1]))
                    if min_ind0==min_ind1:
                        another_ind = list(set(t)-set([min_ind0]))[0]
                        if abs(l_data1[i,min_ind0]-l_data1[i+1,0])>abs(l_data1[i,min_ind0]-l_data1[i+1,1]):
                            tmp0, tmp1 = l_data1[i,another_ind], l_data1[i,min_ind1]
                        else:
                            tmp0, tmp1 = l_data1[i,min_ind0], l_data1[i,another_ind]
                        l_data1[i,0], l_data1[i,1] = tmp0, tmp1
                    else:
                        tmp0, tmp1 = l_data1[i,min_ind0], l_data1[i,min_ind1]
                        l_data1[i,0], l_data1[i,1] = tmp0, tmp1
                else:
                    min_ind = np.argmin(abs(l_data1[i,t[0]]-l_data1[i+1,:2]))
                    l_data1[i, min_ind] = l_data1[i,0]
            
            l_data2 = l_data2[y[0].flatten(),:]
            for i in range(1, l_data2.shape[0]):
                t = np.where(l_data2[i,:]>0)[0]
                if len(t)>1:
                    min_ind0 = np.argmin(abs(l_data2[i,t]-l_data2[i-1,0]))
                    min_ind1 = np.argmin(abs(l_data2[i,t]-l_data2[i-1,1]))
                    if min_ind0==min_ind1:
                        another_ind = list(set(t)-set([min_ind0]))[0]
                        if abs(l_data2[i,min_ind0]-l_data2[i-1,0])>abs(l_data2[i,min_ind0]-l_data2[i-1,1]):
                            tmp0, tmp1 = l_data2[i,another_ind], l_data2[i,min_ind1]
                        else:
                            tmp0, tmp1 = l_data2[i,min_ind0], l_data2[i,another_ind]
                        l_data2[i,0], l_data2[i,1] = tmp0, tmp1
                    else:
                        tmp0, tmp1 = l_data2[i,min_ind0], l_data2[i,min_ind1]
                        l_data2[i,0], l_data2[i,1] = tmp0, tmp1
                else:
                    min_ind = np.argmin(abs(l_data2[i,t[0]]-l_data2[i-1,:2]))
                    l_data2[i, min_ind] = l_data2[i,0]
            for i in range(0, l_data2.shape[0]-1)[::-1]:
                t = np.where(l_data2[i,:]>0)[0]
                if len(t)>1:
                    min_ind0 = np.argmin(abs(l_data2[i,t]-l_data2[i+1,0]))
                    min_ind1 = np.argmin(abs(l_data2[i,t]-l_data2[i+1,1]))
                    if min_ind0==min_ind1:
                        another_ind = list(set(t)-set([min_ind0]))[0]
                        if abs(l_data2[i,min_ind0]-l_data2[i+1,0])>abs(l_data2[i,min_ind0]-l_data2[i+1,1]):
                            tmp0, tmp1 = l_data2[i,another_ind], l_data2[i,min_ind1]
                        else:
                            tmp0, tmp1 = l_data2[i,min_ind0], l_data2[i,another_ind]
                        l_data2[i,0], l_data2[i,1] = tmp0, tmp1
                    else:
                        tmp0, tmp1 = l_data2[i,min_ind0], l_data2[i,min_ind1]
                        l_data2[i,0], l_data2[i,1] = tmp0, tmp1
                else:
                    min_ind = np.argmin(abs(l_data2[i,t[0]]-l_data2[i+1,:2]))
                    l_data2[i, min_ind] = l_data2[i,0]

            
            l11 = self.Kalman_F([l_data1[:,0], idata[ind, 0]])[0].flatten()
            l12 = self.Kalman_F([l_data1[:,1], idata[ind, 0]])[0].flatten()
            l21 = self.Kalman_F([l_data2[:,0], idata[ind, 0]])[0].flatten()
            l22 = self.Kalman_F([l_data2[:,1], idata[ind, 0]])[0].flatten()
            o1 = self.Kalman_F([o_data[:,0], idata[ind, 0]])[0].flatten()
            o2 = self.Kalman_F([o_data[:,1], idata[ind, 0]])[0].flatten()
            lane_center1, lane_center2 = (l11+l22)/2, (l12+l21)/2
            lane_width1, lane_width2 = np.abs(l11-l22), np.abs(l12-l21)
            return [[(o1-lane_center1)/lane_width1, idata[ind, 0]], [(o2-lane_center2)/lane_width2, idata[ind, 0]]]
        else:
            print(x/img_num, '||', np.argwhere(x/img_num<0.25))
            print('#############################################')
        res = []
        return res

    def normalize(self):
        for l in self.flist:
            print(l)
            location = {'bike':[], 'people':[]}
            loc = [[],[],[]]
            loca = {'bike':np.zeros((250, 13)), 'people':np.zeros((250, 13))}
            lane = [[], []]
            with open(os.path.join(self.file_path, l)) as fr:
                for line in fr.readlines():
                    line = line.strip().split()
                    
                    tmp = [int(float(x)) for x in line[2:]]
                    line_center = (tmp[2]+tmp[3])/2
                    line_wight = abs(tmp[2]-tmp[3])
                    object_center = (tmp[0]+tmp[1])/2

                    cnt = int(line[0])
                    loca['bike'][cnt,0] = cnt
                    loca['people'][cnt,0] = cnt
                    for k in range(1,5):
                        if loca[line[1]][cnt,k]==0:
                            loca[line[1]][cnt,k] = object_center
                            loca[line[1]][cnt,k+4] = tmp[2]
                            loca[line[1]][cnt,k+8] = tmp[3]
                            break
            
            color = ['r', 'g', 'b', 'c', 'm', 'y']  
            plt.figure(figsize=(6, 6))
            '''
            for k, lo in enumerate(self.adjust(loca['bike'])):
                if len(lo[0])>20:
                    plt.plot(lo[1], lo[0], c=color[k], label='bike')
                    with open(os.path.join(self.save_path_txt, l.replace('.txt','_bike.txt')), 'w') as fw:
                        for i in range(len(lo[0])):
                            fw.write(f'{lo[1][i]}\t{lo[0][i]}\n')
            '''
            for k, lo in enumerate(self.adjust(loca['people'])):
                if len(lo[0])>20:
                    plt.plot(lo[1], lo[0], c='r', label='people')
                    with open(os.path.join(self.save_path_txt, l.replace('.txt','_people.txt')), 'w') as fw:
                        for i in range(len(lo[0])):
                            fw.write(f'{lo[1][i]}\t{lo[0][i]}\n')
            #'''
            plt.hlines(0.5, 0, 120, linestyles='--', label='lane')
            plt.hlines(-0.5, 0, 120, linestyles='--')
            plt.hlines(1.5, 0, 120, label='lane')
            plt.hlines(-1.5, 0, 120)
            plt.xlim([0, 120])
            plt.ylim([-2, 2])
            if l.find('along')>0: plt.title('along')
            elif l.find('r2l')>0: plt.title('right-left')
            elif l.find('l2r')>0: plt.title('left-right')

            plt.xlabel('time-index')
            plt.ylabel('relative_position')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.save_path, l.replace('txt','png')))
            plt.close()

if __name__=='__main__':
    x = 'train'
    get_timesq(x)
    s = get_location(split=x)
    s.normalize()

