from multiprocessing.connection import wait
from pkgutil import get_data
from tkinter import *
import tkinter as tk
from PIL import ImageTk,Image 
import datetime
from itertools import takewhile
import math
import sys 
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from matplotlib.font_manager import FontProperties
from multiprocessing.pool import ThreadPool          #C:/Users/Mohamed/Desktop/PFE M2/benchmarks/Simulation Exacte/n15/E15k2.xls
from matplotlib import rc
import time
from time import perf_counter
from sklearn.neighbors import NearestNeighbors

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 'family':'serif','serif':['Times'] Serif
rc('font',**{'family':'sans-serif','sans-serif':['Serif'],'weight':'bold'})
rc('text', usetex=True)
#C:/Users/Mohamed/Desktop/PFE M2/interface1.jpg
window =Tk()
window.resizable(width=False, height=False)
window.title("Waste Collection")
window.geometry('1200x700')

# Create a photoimage object of the image in the path
image1 = Image.open("C:/Users/Mohamed/Desktop/PFE M2/interfacecopy9.jpg")
test = ImageTk.PhotoImage(image1)

label1 = tk.Label(image=test)
label1.image = test

# Position image
label1.place(x=0, y=0)

img = ImageTk.PhotoImage(Image.open("C:/Users/Mohamed/Desktop/PFE M2/cdtalogocopy.png"))  
logo=Label(image=img)
logo.pack(side = "left",fill='both', expand = "no")
img2 = ImageTk.PhotoImage(Image.open("C:/Users/Mohamed/Desktop/PFE M2/usthbcopy.jpg"))  
logo2=Label(image=img2)
logo2.pack(side = "right",fill='both', expand = "no")


window.iconbitmap("C:/Users/Mohamed/Desktop/PFE M2/interfacecopy8.ico")#FFFAFA #343B3D
lb1=Label(window, text="Dynamic Waste Collection",bg="#FFFAFA",fg="black", font=("Bahnshrift", 30,"bold","italic"), padx=5, pady=5)
lb1.place(relx=.5, rely=0.06,anchor="center")

lb2=Label(window, text="Case study: BabaHassen",font=("Bahnshrift", 12,"bold","italic"), bg="#FFFAFA",fg="black",  padx=5, pady=5)
lb2.place(relx=.5, rely=0.11,anchor="center")

def clear():
    textbox_1.delete(1.0, END)
    textbox_3.delete(1.0,END)

def heat(old_path, node_w, m,cap, D):
    """
    heating function
     :param D: distance between nodes
     :param m: number of vehicles
     :param node_w: (initially defined) weight of each point
     :param old_path: original solution, i.e. any solution set in initialization
     :return: return the initial temperature T0 and the old path old_path
    """
    dc = np.zeros(4000)  # Set the number of heating times. In this example, the 2-opt algorithm is used to heat the initial solution 4000 times (randomly scrambled)

    for i in range(4000):
        new_path = new_paths(old_path,node_w,m,cap)  # generate new paths
        dis1 = total_dis(old_path, D)  # Calculate the distance of the old path
        dis2 = total_dis(new_path, D)  # Calculate the distance of the new path
        dc[i] = abs(dis2 - dis1)  # Distance deviation between old and new paths
        old_path = new_path

    T0 =10*max(dc)  # set the initial temperature to 20 times the max deviation

    return T0, old_path


def generate_new_path(old_path):  
    N = len(old_path)
    a, b = np.random.randint(1, N), np.random.randint(1, N)  # Generate a random integer between, ensure that the beginning and the end are 0
    random_left, random_right = min(a, b), max(a, b)  # Sort the resulting integers
    rever = old_path[random_left:random_right]  # Randomly drawn path of the middle part of old_path
    new_path = old_path[:random_left] + rever[::-1] + old_path[random_right:]  # 2-opt algorithm, flip splicing into a new path

    return new_path

def new_paths(old_path,node_w,m,cap):
    optimal_found=False
    while not optimal_found:
        optimal_found=True
        new_path=generate_new_path(old_path)
        address_index = [i for i in range(len(new_path)) if new_path[i] == 0]  
        C= [0] * (len(address_index))
        for i in range(len(address_index) - 1): 
            for j in range(address_index[i], address_index[i + 1], 1):
                C[i] += node_w[new_path[j]]  
            #print(new_path)
            if C[i] > cap[0]:
                optimal_found=False
                #print("camion numero ",i,Charge[i]," >= ",cap[i])
                break 
    return new_path



def total_dis(path,D):
    """
    To calculate the distance function, here is a little trick, that is, to calculate the distance in advance to form a two-dimensional list, and then you only need to look up the list to calculate the distance, which greatly saves time.
     :param D: distance between nodes
     :param m: number of vehicles
     :param path: the path to calculate
     :param node_w: (initially defined) weight of each point
     :return: The objective function, that is, the distance value of the current path + the combination of the penalty term
    """
    dis= 0
    for i in range(len(path) - 1):  # Find the distance between two points in the path path, and return the distance value by looking up a list
        dis += D[path[i]][path[i + 1]]

    return dis


def init():
    def get_data():
        def column_len(sheet, index):
            col_values = sheet.col_values(index)
            col_len = len(col_values)
            for _ in takewhile(lambda x: not x, reversed(col_values)):
                col_len -= 1
            return col_len
        #data = read_excel("C:/Users/Mohamed/Desktop/PFE M2/Data.xlsx",engine='openpyxl') C:/Users/Mohamed/Desktop/PFE M2/Data4.
        def waithere():
            var = IntVar()
            window.after(5000, var.set, 1)
            print("waiting...")
            window.wait_variable(var)
        while True:
            try:
                msg = textbox_1.get("1.0", "end")[:-1]
                data = xlrd.open_workbook(msg)                
                break
            except FileNotFoundError:
                print('File not found')
                textbox_3.insert(tk.END,'File not found')
                textbox_3.insert(END, '\n')
                textbox_3.insert(tk.END,'Please provide the correct file path')
                #time.sleep(5)
                waithere()
                textbox_3.delete("1.0",tk.END)
                textbox_1.delete("1.0",tk.END)
                break
                
        #data = xlrd.open_workbook(path)  # Open the selected excel table and assign it to 
        
        table = data.sheet_by_index(0)  #Read a form in excel, 0 is the default preferred form
        m=column_len(table, 3)
        return table.col_values(0), table.col_values(1), table.col_values(2),table.col_values(3),m,table.col_values(4)
    node_x, node_y, node_w,cap,m,var = get_data()
    #x_cor,y_cor,capacity=node_x,node_y,node_w
    def strategy(var,node_x, node_y, node_w):
        x_cor,y_cor,capacity=[],[],[]
        x_cor.append(node_x[0])
        y_cor.append(node_y[0])
        capacity.append(node_w[0])
        if (var[0]<50):
            for i in range(0,len(node_x)):
                if(node_w[i]>1600):
                    x_cor.append(node_x[i])
                    y_cor.append(node_y[i])
                    capacity.append(node_w[i])
        if (var[0]>=50):
            for i in range(0,len(node_x)):
                if(node_w[i]>1300):
                    x_cor.append(node_x[i])
                    y_cor.append(node_y[i])
                    capacity.append(node_w[i])
        path=[]
        for j in range(len(node_w)):
            for i in range(len(capacity)):
                if((capacity[i]==node_w[j])):
                    path.append(j)
        final_path = list(dict.fromkeys(path))
        return final_path

    final_path= strategy(var,node_x, node_y, node_w) 
    n=len(final_path)-1  
    return  node_x, node_y, node_w,cap,var,m,final_path,n

def solution_initial(m,node_w,cap,final_path):
    c=[0]*m
    path=[]
    capacity=cap[0]
    for i in range(len(c)):
        #initial_path.append([])
        for j in final_path:
            if ((c[i]+node_w[j])<=capacity) and (j not in path):
                c[i]+=node_w[j]
                path.append(j)
                #initial_path[i].append(j)
        path.append(0)
    return path

#C:/Users/Mohamed/Desktop/PFE M2/benchmarks/Set B/B/B51.xls
def pic(node_x, node_y):
    def background():
        """
        Using plt to draw a coordinate system background, including setting information such as coordinates and dimensions
        """
        plt.figure(figsize=(21, 21.5),facecolor='#5CB3FF') #C0C0C0 #3D3C3A #BCC6CC' #52595D
        ax=plt.axes()
        ax.set_xlabel('Coordonnées X',fontsize=16, fontweight='bold')
        ax.set_ylabel('Coordonnées Y',fontsize=16, fontweight='bold')
        ax.set_facecolor('#EBF4FA')#98AFC7 #728FCE #00BFFF #2B65EC #EBF4FA
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.annotate (xy=(100,100),text=('VAR=',var,'%'), xytext=(10,10))
    x_list= [] # Define the x-coordinate of the node for plt drawing
    y_list = [] # Define the y-coordinate of the node for plt drawing
    background()
    #plt.scatter(node_x, node_y, c='y', s=50, alpha=1)  # Draw a scale plot of the store's
    #address_index = [i for i in range(len(best_path)) if best_path[i] == 0 ]  # Find the coordinates of 0 ( depot ) in the path
    for i in range(len(node_x)):
        x_list.append(node_x[i])
        y_list.append(node_y[i])
    
    plt.scatter(x_list,y_list, color='#228B22',s=50,label="conteneur")    #800080  #2E8B57 #00A36C #3EB489
    plt.plot(x_list[0], y_list[0], 'k*', markersize=20, label="Dépot") 

    plt.title('Illustration des conteneurs', fontsize=18 ,fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(color='black', linestyle='-', linewidth=0.15, alpha=0.5)
    plt.show()

def pict(node_x, node_y,node_w,var):
    def background():
        """
        Using plt to draw a coordinate system background, including setting information such as coordinates and dimensions
        """
        plt.figure(figsize=(21, 21.5),facecolor='#EBF4FA') 
        ax=plt.axes()
        ax.set_xlabel('Coordonnées X',fontsize=16, fontweight='bold')
        ax.set_ylabel('Coordonnées Y',fontsize=16, fontweight='bold')
        ax.set_facecolor('#7BCCB5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.annotate (xy=(100,100),text=('VAR=',var,'%'), xytext=(10,10))
    x_list,x_lis,x_li= [],[],[] # Define the x-coordinate of the node for plt drawing
    y_list,y_lis,y_li = [],[],[] # Define the y-coordinate of the node for plt drawing
    background()
    #plt.scatter(node_x, node_y, c='y', s=50, alpha=1)  # Draw a scale plot of the store's
    #address_index = [i for i in range(len(best_path)) if best_path[i] == 0 ]  # Find the coordinates of 0 ( depot ) in the path
    for i in range(0,len(node_x)):
        if (node_w[i]<=1300):
            x_list.append(node_x[i])
            y_list.append(node_y[i])
    for i in range(0,len(node_x)):
        if (1300 <node_w[i]<=1600): 
            x_lis.append(node_x[i])
            y_lis.append(node_y[i])
    for i in range(0,len(node_x)):
        if (node_w[i]>1600):
            x_li.append(node_x[i])
            y_li.append(node_y[i])
    plt.scatter(x_list,y_list, color='#0000FF',s=60,label="bin blue")
    plt.scatter(x_lis,y_lis, color='#FC9A04',s=60,label="bin orange") 
    plt.scatter(x_li,y_li, color='#FF0000',s=60,label="bin rouge")       
    plt.plot(x_list[0], y_list[0], 'k*', markersize=20, label="Dépot") 
    per=f"{var[0]}%"
    #plt.plot(x_l,y_l,c=random_color())
    plt.annotate('VAR= {} '.format(per),xy=(100,100),fontweight='bold')
    plt.title('Illustration des états des conteneurs', fontsize=18 ,fontweight='bold')
    #x_list = []  # Emptying
    #y_list = []  # Emptying
    plt.legend(loc='lower left')
    plt.grid(color='black', linestyle='-', linewidth=0.15, alpha=0.5)
    plt.show()

def metropolis(old_path, new_path, T, D):
    """
    The metropolis criterion, which is the probability of accepting a new solution in the simulated annealing algorithm, is needed to prevent falling into a local optimum, so it is necessary to accept a new solution with a certain probability
    :param old_path: old path
    :param new_path: new path
    :param node_w: the amount of goods demanded by the store
    :param m: number of vehicles
    :param T: the current temperature under the simulated annealing outer loop
    :param D: distance between nodes
    :return: return the current optimal solution and the corresponding objective function (distance) under the metropolis criterion judgment
    """
    dis1 = total_dis(old_path , D)  # Distance of the old path
    dis2 = total_dis(new_path, D)  # Distance of the new path

    dc = dis2 - dis1  # Difference between the two

    if dc < 0 or np.exp(-abs(dc) / T) > np.random.random():  # metropolis criteria, two cases of accepting the new solution: 1. the distance of the new solution is smaller; 2. a certain probability of accepting
        path = new_path
        path_dis = dis2
    else:
        path = old_path
        path_dis = dis1

    return path, path_dis


def picture(node_x, node_y, best_path,m):
    """
    The function to map the access path of a vehicle
    :param node_x: the x-coordinate of the store's location
    :param node_y: y-coordinate of the store's location
    :param best_path: the optimal path
    :return:
    """

    def background():
        """
        Using plt to draw a coordinate system background, including setting information such as coordinates and dimensions
        """
        plt.figure(figsize=(21, 21.5),facecolor='#EBF4FA') 
        ax=plt.axes()
        ax.set_xlabel('Coordonnées X',fontsize=16, fontweight='bold')
        ax.set_ylabel('Coordonnées Y',fontsize=16, fontweight='bold')
        ax.set_facecolor('#7BCCB5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    x_list= [] # Define the x-coordinate of the node for plt drawing
    y_list = [] # Define the y-coordinate of the node for plt drawing
    background()
    index = {}
    for j in range(m):
        index[j+1] = [0 for i in range(len(best_path))]
    
    plt.scatter(node_x, node_y, c='w', s=60, alpha=1)  # Draw a scale plot of the store's
    address_index = [i for i in range(len(best_path)) if best_path[i] == 0 ]  # Find the coordinates of 0 ( depot ) in the path
    for i in range(len(address_index) - 1):
        for j in range(address_index[i], address_index[i + 1] + 1, 1):
            x_list.append(node_x[best_path[j]])
            y_list.append(node_y[best_path[j]])
            #plt.plot( x_list[0], y_list[0], c='g',marker='D')
    for i in range(len(address_index) - 1):
            #index[d+1][i]=best_path[address_index[i]:address_index[i + 1] + 1]
        #print('list of index is= ',best_path[address_index[i]:address_index[i + 1] + 1])
        index[i+1]=best_path[address_index[i]:address_index[i + 1] + 1]
    loc_x, loc_y = {}, {}
    N=len(node_x)
    for j in range(m):
        LEN_ROUTE = len(index[j+1])
        loc_x[j], loc_y[j] = [], []
        for i in range(LEN_ROUTE):
                if index[j+1][i] < N:
                    loc_x[j].append(node_x[index[j+1][i]])
                    loc_y[j].append(node_y[index[j+1][i]])
                else:
                    loc_x[j].append(node_x[0])
                    loc_y[j].append(node_y[0])
    
    #print('list of index is= ',index) #Set1 #hsv
    cmap = plt.cm.get_cmap('Dark2', m+1)
    for j in range(m):
        plt.plot(loc_x[j], loc_y[j],
                color=cmap(j),
                marker='x',
                linewidth=.1,
                markersize=9,markeredgecolor=cmap(j), markeredgewidth=4,label='Conteneur collecté par le véhicule %s' %(j+1)) 
        N = len(loc_x[j])-1
        for i in range(N):
            plt.annotate('',
                        xy=(loc_x[j][i+1], loc_y[j][i+1]),
                        xytext=(loc_x[j][i], loc_y[j][i]),label='%s Road of vehicle ' %j,
                        arrowprops=dict(
                            arrowstyle="->",
                            color=cmap(j),
                            lw=1.75
                        )
            )
         
    plt.plot(node_x[0], node_y[0], 'k*', markersize=22, label="Dépot") #ko
    #plt.plot(x_l,y_l,c=random_color())
    #plt.annotate('$dépot$',(x_list[0]+1, y_list[0]))
    plt.title('Illustration de la tournée pour chaque véhicule:', fontsize=18 ,fontweight='bold')
    #x_list = []  # Emptying
    #y_list = []  # Emptying
    plt.legend(loc='lower left')
    plt.grid(color='black', linestyle='-', linewidth=0.15, alpha=0.5)
    plt.savefig('result.png')  # Save vehicle path of travel visualization
    plt.show()

def matricedistance(node_x,node_y,init_path):
    D = [[0] * len(node_x) for i in range(len(node_x))]  # Define a two-dimensional list to hold the distances between nodes
    for i in init_path:
            for j in init_path:
                D[i][j] = np.sqrt((node_x[i] - node_x[j]) ** 2 + (node_y[i] - node_y[j]) ** 2)
    return D


def CVRP_SA1():
    node_x, node_y, node_w,cap,var,m,final_path,n= init()  # In this example there are n stores, 1 depot and 8 
    init_path=solution_initial(m,node_w,cap,final_path)

    C,totale_cap=0,0
    for i in range(m):
        totale_cap+=cap[i]
    for i in final_path:
        C+=node_w[i]
  
    if totale_cap < C:
        raise ValueError("La quantité totale à collecter doit etre inferieur à la capacité des véhicules disponibles") 
    else:
        start_time = datetime.datetime.now()
        #for i in init_path:
            #for j in init_path:
                #D[i][j] = np.sqrt((node_x[i] - node_x[j]) ** 2 + (node_y[i] - node_y[j]) ** 2)  # Calculate the paths
        D=matricedistance(node_x,node_y,init_path)
        T0, old_path = heat(init_path, node_w, m,cap, D)  # Initial temperature, (the path after the heating process as) the initial path

        T_down_rate = 0.994  # Temperature drop rate
        T_end = 0.01  # Ending temperature
        K = 1200  # Number of internal loops

        #count = math.ceil(math.log(T_end / T0, T_down_rate))  # Number of external loops
        #print('count= ',count)
        #dis_T = np.zeros(count + 1)  # Optimal solution under each loop
        best_path = init_path  # Set the optimal path as the initial path
        shortest_dis = np.inf  # Set the initial optimal distance to infinity
        n = 0
        T = T0  # Temperature value under current loop

        while T > T_end:  # Repeat until the temperature is less than the ending temperature 
            for i in range(K):
                new_path = new_paths(old_path,node_w,m,cap)  # Generate a new path
                old_path, path_dis = metropolis(old_path, new_path, T, D)  # Determine the current optimal path by the metropolis criteria
                if path_dis <= shortest_dis:  # If the optimal path is better than the previous calculated result, accept
                    shortest_dis = path_dis
                    best_path = old_path

            #dis_T[n] = shortest_dis  # Optimal solution under each loop
            #print('dis',dis_T[n])
            n += 1
            T *= T_down_rate  # After each cycle, the temperature is cooled down at a certain rate of decrease
            #print('best current dis is:', shortest_dis)  # Print out the execution process so that the T value and the current optimal distance can be monitored at all times
            #print('number of iterations is= ', n)
            #print('current temperature is =',T)
        #print('dis',dis_T)
        end_time = datetime.datetime.now()

        print('')
        
        textbox_3.insert(tk.END,'Best path= ')
        textbox_3.insert(tk.END, '[')
        textbox_3.insert(tk.END,best_path)
        textbox_3.insert(tk.END, ']')
        textbox_3.insert(END, '\n')
        print('best_path', best_path)  # Output the optimal path solution
        
        print('')#math.ceil
        textbox_3.insert(tk.END,'Total distance= ')
        textbox_3.insert(tk.END, round(total_dis(best_path,D),3))
        textbox_3.insert(tk.END, 'km')
        textbox_3.insert(END, '\n')
        print('The result of the SA algorithm to calculate the CVRP problem is= ',  (total_dis(best_path,D)))  # Output the optimal calculation result, i.e. the sum of the formal paths of the vehicles
        textbox_3.insert(END, '\n')
        print('')
        address_index = [i for i in range(len(best_path)) if best_path[i] == 0]  # Find the coordinates of 0 ( depot ) in the path
        textbox_3.insert('end -1 chars', '')
        for i in range(len(address_index) - 1):  # Output the path of each vehicle
            print('The path of the {} vehicle is：'.format(i + 1))
            textbox_3.insert(tk.END,'The path of the {} vehicle is：'.format(i + 1))
            print(best_path[address_index[i]:address_index[i + 1] + 1])
            textbox_3.insert(tk.END, '[')
            textbox_3.insert(tk.END, best_path[address_index[i]:address_index[i + 1] + 1])
            textbox_3.insert(tk.END, ']')
            textbox_3.insert(END, '\n')
        print('')
        print('capacity of the vehicles is= ', totale_cap)
        print('')
        print('amount of waste to be collected is= ',C)
        print('')
        textbox_3.insert(END, '\n')
        textbox_3.insert(tk.END,'Running time= ')
        textbox_3.insert(tk.END, end_time - start_time)
        
        print('The program run time is：', end_time - start_time)
        
        pic(node_x, node_y)
        pict(node_x, node_y,node_w,var)
        picture(node_x, node_y, best_path,m)
         # Visualization of vehicle paths of travel
    return best_path


def matricedist(node_x,node_y):
    D = [[0] * len(node_x) for i in range(len(node_x))]  # Define a two-dimensional list to hold the distances between nodes
    for i in range(len(node_x)):
            for j in range(len(node_x)):
                D[i][j] = np.sqrt((node_x[i] - node_x[j]) ** 2 + (node_y[i] - node_y[j]) ** 2)
    return D



def heatio(old_path, node_w, m,cap, D):
    """
    heating function
     :param D: distance between nodes
     :param m: number of vehicles
     :param node_w: (initially defined) weight of each point
     :param old_path: original solution, i.e. any solution set in initialization
     :return: return the initial temperature T0 and the old path old_path
    """
    dc = np.zeros(4000)  # Set the number of heating times. In this example, the 2-opt algorithm is used to heat the initial solution 4000 times (randomly scrambled)

    for i in range(4000):
        new_path = neww_paths(old_path)  # generate new paths
        dis1 = total_distance(old_path, node_w, m,cap, D)  # Calculate the distance of the old path
        dis2 = total_distance(new_path, node_w, m,cap, D)  # Calculate the distance of the new path
        dc[i] = abs(dis2 - dis1)  # Distance deviation between old and new paths
        old_path = new_path

    T0 =10*max(dc)  # set the initial temperature to 20 times the max deviation

    return T0, old_path
    
def neww_paths(old_path):
    """
   Generate a new path function. In this example, the 2-opt algorithm is used to generate a new path
     :param old_path: old path
     :return: new path
    """
    
    N = len(old_path)
    a, b = np.random.randint(1, N), np.random.randint(1, N)  # Generate a random integer between, ensure that the beginning and the end are 0
    random_left, random_right = min(a, b), max(a, b)  # Sort the resulting integers
    rever = old_path[random_left:random_right]  # Randomly drawn path of the middle part of old_path
    new_path = old_path[:random_left] + rever[::-1] + old_path[random_right:]  # 2-opt algorithm, flip splicing into a new path

    return new_path

def total_distance(path, node_w, m,cap,D):
    """
    To calculate the distance function, here is a little trick, that is, to calculate the distance in advance to form a two-dimensional list, and then you only need to look up the list to calculate the distance, which greatly saves time.
     :param D: distance between nodes
     :param m: number of vehicles
     :param path: the path to calculate
     :param node_w: (initially defined) weight of each point
     :return: The objective function, that is, the distance value of the current path + the combination of the penalty term
    """
    dis= 0
    for i in range(len(path) - 1):  # Find the distance between two points in the path path, and return the distance value by looking up a list
        dis += D[path[i]][path[i + 1]]
    address_index = [i for i in range(len(path)) if path[i] == 0]  # Find the coordinates of 0 (depot) in the path
    C,M = [0] * m ,[0] * m # capacity per vehicle
                           # Set a penalty item, if the maximum capacity of each vehicle exceeds 100, a penalty will be imposed
    #d=1
    #print('add= ',address_index)
    for i in range(len(address_index) - 1):  # The number between the depot coordinates (0) is the route each vehicle travels
        for j in range(address_index[i], address_index[i + 1], 1):
            C[i] += node_w[path[j]]  # Calculate the current capacity of each vehicle to ensure that the maximum capacity limit of 200 is not exceeded
        #print('cap',C[i])
        if C[i] >= cap[0]:
            M[i] =1000*(C[i] - cap[0])# Penalty term to prevent the capacity of the vehicle from exceeding the maximum limit of 200, penalty term factor 20 can be modified
        sum_M = sum(M)  # Sum of penalty terms
            #d+=1

    return dis + sum_M

def init():
    def get_data():
        def column_len(sheet, index):
            col_values = sheet.col_values(index)
            col_len = len(col_values)
            for _ in takewhile(lambda x: not x, reversed(col_values)):
                col_len -= 1
            return col_len
        #data = read_excel("C:/Users/Mohamed/Desktop/PFE M2/Data.xlsx",engine='openpyxl') C:/Users/Mohamed/Desktop/PFE M2/Data4.
        def waithere():
            var = IntVar()
            window.after(5000, var.set, 1)
            print("waiting...")
            window.wait_variable(var)
        while True:
            try:
                msg = textbox_1.get("1.0", "end")[:-1]
                data = xlrd.open_workbook(msg)                
                break
            except FileNotFoundError:
                print('File not found')
                textbox_3.insert(tk.END,'File not found')
                textbox_3.insert(END, '\n')
                textbox_3.insert(tk.END,'Please provide the correct file path')
                #time.sleep(5)
                waithere()
                textbox_3.delete("1.0",tk.END)
                textbox_1.delete("1.0",tk.END)
                break
                
        #data = xlrd.open_workbook(path)  # Open the selected excel table and assign it to 
        
        table = data.sheet_by_index(0)  #Read a form in excel, 0 is the default preferred form
        m=column_len(table, 3)
        return table.col_values(0), table.col_values(1), table.col_values(2),table.col_values(3),m,table.col_values(4)
    node_x, node_y, node_w,cap,m,var = get_data()
    #x_cor,y_cor,capacity=node_x,node_y,node_w
    def strategy(var,node_x, node_y, node_w):
        x_cor,y_cor,capacity=[],[],[]
        x_cor.append(node_x[0])
        y_cor.append(node_y[0])
        capacity.append(node_w[0])
        if (var[0]<50):
            for i in range(0,len(node_x)):
                if(node_w[i]>1600):
                    x_cor.append(node_x[i])
                    y_cor.append(node_y[i])
                    capacity.append(node_w[i])
        if (var[0]>=50):
            for i in range(0,len(node_x)):
                if(node_w[i]>1300):
                    x_cor.append(node_x[i])
                    y_cor.append(node_y[i])
                    capacity.append(node_w[i])
        path=[]
        for j in range(len(node_w)):
            for i in range(len(capacity)):
                if((capacity[i]==node_w[j])):
                    path.append(j)
        final_path = list(dict.fromkeys(path))
        return final_path

    final_path= strategy(var,node_x, node_y, node_w) 
    n=len(final_path)-1  
    return  node_x, node_y, node_w,cap,var,m,final_path,n

def nearest_neighbor(n,D,m,node_w,cap,final_path):
    knn = NearestNeighbors(n_neighbors=(n))
    knn.fit(D)
    distance_mat, neighbours_mat = knn.kneighbors(D)
    #print('nn=',neighbours_mat)
    con=np.concatenate(neighbours_mat)
    #print('nei=', con)
    neib=[]
    for i in con:
        if i not in neib:
            neib.append(i)
    #print('final=',final_path)
    #print('orig=',neib)
    nefroha=[]
    for i in neib:
        #print('i=',i)
        if (i in final_path ):
            nefroha.append(i)
    #print('nefroha=',nefroha)
    #print('neib=',neib)
    charge=[0]*m
    neibo=[]
    for i in range(0,m):
        for j in nefroha:
            if(charge[i]+node_w[j]<=cap[0]) and (j not in neibo):
                charge[i]+=node_w[j]
                neibo.append(j)
        neibo.append(0)
    return neibo

def metropolisio(old_path, new_path, node_w, m,cap, T, D):
    """
    The metropolis criterion, which is the probability of accepting a new solution in the simulated annealing algorithm, is needed to prevent falling into a local optimum, so it is necessary to accept a new solution with a certain probability
    :param old_path: old path
    :param new_path: new path
    :param node_w: the amount of goods demanded by the store
    :param m: number of vehicles
    :param T: the current temperature under the simulated annealing outer loop
    :param D: distance between nodes
    :return: return the current optimal solution and the corresponding objective function (distance) under the metropolis criterion judgment
    """
    dis1 = total_distance(old_path, node_w, m,cap, D)  # Distance of the old path
    dis2 = total_distance(new_path, node_w, m,cap, D)  # Distance of the new path

    dc = dis2 - dis1  # Difference between the two

    if dc < 0 or np.exp(-(dc) / T) >= np.random.random():  # metropolis criteria, two cases of accepting the new solution: 1. the distance of the new solution is smaller; 2. a certain probability of accepting
        path = new_path
        path_dis = dis2
    else:
        path = old_path
        path_dis = dis1

    return path, path_dis

def CVRP_SA2():
    node_x, node_y, node_w,cap,var,m,final_path,n= init()  # In this example there are n stores, 1 depot and 8 
   
    C,totale_cap=0,0
    Distancia = matricedist(node_x,node_y)
    init_path=nearest_neighbor(n,Distancia,m,node_w,cap,final_path)
    for i in range(m):
        totale_cap+=cap[i]
    for i in final_path:
        C+=node_w[i]
    if totale_cap < C:
        raise ValueError("La quantité totale à collecter doit etre inferieur à la capacité des véhicules disponibles")  
    else:
        start_time = datetime.datetime.now()
        #for i in init_path:
            #for j in init_path:
                #D[i][j] = np.sqrt((node_x[i] - node_x[j]) ** 2 + (node_y[i] - node_y[j]) ** 2)  # Calculate the paths
        T0, old_path = heatio(init_path, node_w, m,cap, Distancia)  # Initial temperature, (the path after the heating process as) the initial path

        T_down_rate = 0.994  # Temperature drop rate
        T_end = 0.01  # Ending temperature
        K = 1200  # Number of internal loops

        #count = math.ceil(math.log(T_end / T0, T_down_rate))  # Number of external loops
        #print('count= ',count)
        #dis_T = np.zeros(count + 1)  # Optimal solution under each loop
        best_path = init_path  # Set the optimal path as the initial path
        shortest_dis = np.inf  # Set the initial optimal distance to infinity
        n = 0
        T = T0  # Temperature value under current loop

        while T > T_end:  # Repeat until the temperature is less than the ending temperature 
            for i in range(K):
                new_path = neww_paths(old_path)  # Generate a new path
                old_path, path_dis = metropolisio(old_path, new_path, node_w, m,cap, T, Distancia)  # Determine the current optimal path by the metropolis criteria
                if path_dis <= shortest_dis:  # If the optimal path is better than the previous calculated result, accept
                    shortest_dis = path_dis
                    best_path = old_path

            #dis_T[n] = shortest_dis  # Optimal solution under each loop
            #print('dis',dis_T[n])
            n += 1
            T *= T_down_rate  # After each cycle, the temperature is cooled down at a certain rate of decrease
            #print('best current dis is:', shortest_dis)  # Print out the execution process so that the T value and the current optimal distance can be monitored at all times
            #print('number of iterations is= ', n)
            #print('current temperature is =',T)
        #print('dis',dis_T)
        end_time = datetime.datetime.now()
        if totale_cap < C:
            textbox_3.insert(tk.END,'La quantité à collecter doit etre <= à la capacité totale des véhhicules ')
        print('')
        
        textbox_3.insert(tk.END,'Best path= ')
        textbox_3.insert(tk.END, '[')
        textbox_3.insert(tk.END,best_path)
        textbox_3.insert(tk.END, ']')
        textbox_3.insert(END, '\n')
        print('best_path', best_path)  # Output the optimal path solution
        
        print('')#math.ceil
        textbox_3.insert(tk.END,'Total distance= ')
        textbox_3.insert(tk.END, round(total_distance(best_path, node_w, m,cap, Distancia),3))
        textbox_3.insert(tk.END, 'km')
        textbox_3.insert(END, '\n')
        print('The result of the SA algorithm to calculate the CVRP problem is= ',  (total_distance(best_path, node_w, m,cap, Distancia)))  # Output the optimal calculation result, i.e. the sum of the formal paths of the vehicles
        textbox_3.insert(END, '\n')
        print('')
        address_index = [i for i in range(len(best_path)) if best_path[i] == 0]  # Find the coordinates of 0 ( depot ) in the path
        textbox_3.insert('end -1 chars', '')
        for i in range(len(address_index) - 1):  # Output the path of each vehicle
            print('The path of the {} vehicle is：'.format(i + 1))
            textbox_3.insert(tk.END,'The path of the {} vehicle is：'.format(i + 1))
            print(best_path[address_index[i]:address_index[i + 1] + 1])
            textbox_3.insert(tk.END, '[')
            textbox_3.insert(tk.END, best_path[address_index[i]:address_index[i + 1] + 1])
            textbox_3.insert(tk.END, ']')
            textbox_3.insert(END, '\n')
        print('')
        print('capacity of the vehicles is= ', totale_cap)
        print('')
        print('amount of waste to be collected is= ',C)
        print('')
        textbox_3.insert(END, '\n')
        textbox_3.insert(tk.END,'Running time= ')
        textbox_3.insert(tk.END, end_time - start_time)
        
        print('The program run time is：', end_time - start_time)
        
        pic(node_x, node_y)
        pict(node_x, node_y,node_w,var)
        picture(node_x, node_y, best_path,m)

    return

label1 = Label(window, text="Enter the file path:", font=("Bahnshrift", 12,"bold","italic"), bg="#FFFAFA",fg="black",)
#label2 = Label(window, text="Clé:", font=("Bahnshrift", 12,"bold","italic"), bg="#FFFAFA",fg="#343B3D",)
#label3 = Label(window, text="Text Crypté/Décrypté ", font=("Bahnshrift", 12,"bold","italic"),bg="#FFFAFA",fg="#343B3D",)

textbox_1 = Text(window, height=1, width=55, font="Bahnshrift 12")
textbox_3 = Text(window, height=10.5, width=55, font="Bahnshrift 13")
radio_button1 = Button(window, text="Strategy 1",font=("Bahnshrift", 13,"bold","italic"),bg="#FFFAFA",fg="black", command=CVRP_SA1)
radio_button2 = Button(window, text="Strategy 2",font=("Bahnshrift", 13,"bold","italic"),bg="#FFFAFA",fg="black",command=CVRP_SA2)
#radio_button2 = Button(window, text="Decrypt",font=("Bahnshrift", 11,"bold","italic"),bg="#FFFAFA",fg="#343B3D",command=decrypt)

#clear_button = Button(window, text='Clear',font=("Bahnshrift", 12,"bold","italic"),bg="#FFFAFA",fg="#343B3D", command=clear)

label1.place(relx=.5, rely=.17, anchor="center")
#label2.place(relx=.5, rely=.41, anchor="center")
#label3.place(relx=.5, rely=.66, anchor="center")

textbox_1.place(relx=.5, rely=.20, anchor="center")
textbox_3.place(relx=.5, rely=.43, anchor="center")
clear_button = Button(window, text='Clear',font=("Bahnshrift", 12,"bold","italic"),bg="#FFFAFA",fg="black", command=clear)
radio_button1.place(relx=.55, rely=.25, anchor="center")
radio_button2.place(relx=.45, rely=.25, anchor="center")
clear_button.place(relx=.53, rely=.93, anchor="center")
#radio_button2.place(relx=.55, rely=.55, anchor="center")
#clear_button.place(relx=.5, rely=.87, anchor="center")

window.mainloop()