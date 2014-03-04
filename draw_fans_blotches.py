import os,sys
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import MySQLdb
import random
from numpy import genfromtxt
from math import pi
from matplotlib.patches import Ellipse

def p4_overplot(image_id):

    db=MySQLdb.connect(db='mysql',host='localhost', user='root')
    cursor=db.cursor()
    cursor.execute("""use P4""")

    cmd='select image_location from HiRISE_images where zooniverse_id='+'"'+image_id+'"'
    cursor.execute(cmd)
    fetch=cursor.fetchall()
    awslocation=fetch[0][0]
    os.system('wget '+awslocation+' -O tmp.jpg')
    os.system('convert ./tmp.jpg tmp.png')
    print  awslocation
    img=mpimg.imread('tmp.png')
    os.remove('tmp.png')
    os.remove('tmp.jpg')

    maxy=len(img[:,0])
    maxx=len(img[0,:])
    #print maxx


    f=plt.figure()
    plt.subplots_adjust(top=0.85)
    f.add_subplot(2, 2, 1)

    imgplot = plt.imshow(img)

    cmd='select distinct classification_id from annotations where image_id='+'"'+image_id+'"'
    cursor.execute(cmd)
    users=cursor.fetchall()
    print len(users)
    j=0

    fans=np.zeros(len(users),dtype=np.int)


    plt.title(image_id)
    for i in users:
        #print i[0]

        cmd='select count(marking) from annotations where image_id='+'"'+image_id+'"'+' and classification_id="'+i[0]+'" and marking="fan"'
        cursor.execute(cmd)
        fetch=cursor.fetchall()

        fans[j]=fetch[0][0]
        j=j+1
    
        cmd='select x,y,distance, angle, spread from annotations where image_id='+'"'+image_id+'"'+' and classification_id="'+i[0]+'" and X IS NOT NULL and marking="fan"'
        cursor.execute(cmd)
        fetch=cursor.fetchall() 
        #print fetch
        if (len(fetch) ==0):
            continue 
        index=np.asarray(fetch)
        fetch=0
        x=np.copy(index)
        x=np.delete(x, 1,1)
        x=np.delete(x, 1,1)
        x=np.delete(x, 1,1)
        x=np.delete(x, 1,1) 
        x=np.asarray(x, dtype=np.float)         

        y=np.copy(index)
        y=np.delete(y, 0,1)
        y=np.delete(y, 1,1)
        y=np.delete(y, 1,1)
        y=np.delete(y, 1,1)
        y=np.asarray(y, dtype=np.float)
        
        distance=np.copy(index)
        distance=np.delete(distance, 0,1)
        distance=np.delete(distance, 0,1)
        distance=np.delete(distance, 1,1)
        distance=np.delete(distance, 1,1)   
        print distance 
        distance=np.asarray(distance, dtype=np.float)
    
        angle=np.copy(index)
        angle=np.delete(angle, 0,1)
        angle=np.delete(angle, 0,1)
        angle=np.delete(angle, 0,1)
        angle=np.delete(angle, 1,1)
        print angle 
        angle=np.asarray(angle, dtype=np.float)

        spread=np.copy(index)
        spread=np.delete(spread, 0,1)
        spread=np.delete(spread, 0,1)
        spread=np.delete(spread, 0,1)
        spread=np.delete(spread, 0,1)   
        print spread    
        spread=np.asarray(spread, dtype=np.float)
        # since it's an isosceles triangle we actually want to use half of spread
        # so let's just do that now
        spread=spread/2.0


        x21=np.zeros(len(x), dtype=np.float)
        y21=np.zeros(len(y), dtype=np.float)

        x22=np.zeros(len(x), dtype=np.float)
        y22=np.zeros(len(y), dtype=np.float)

    
        x21=np.cos(angle*pi/180.0)-(np.tan(spread*pi/180.0)*np.sin(angle*pi/180.0))
        x21=x21*distance    
        
        y21=np.sin(angle*pi/180.0)+(np.tan(spread*pi/180.0)*np.cos(angle*pi/180.0))

        y21=distance*y21
        
        x22=np.cos(angle*pi/180.0)+(np.tan(spread*pi/180.0)*np.sin(angle*pi/180.0))
        x22=x22*distance

        y22=np.sin(angle*pi/180.0)-(np.tan(spread*pi/180.0)*np.cos(angle*pi/180.0))


        y22=y22*distance
                
        # okay now need to move to the frame of the image before assuming that the origin was (0,0) but it's (x,y) on the image     
        x21=x+x21
        y21=y+y21

        x22=x+x22
        y22=y+y22           

        #print y

        for i in np.arange(len(x)):
            plt.plot([x[i], x21[i]], [y[i], y21[i]], color='blue')
            plt.plot([x[i], x22[i]], [y[i], y22[i]], color='blue')  


        plt.ylim([maxy,0])
        plt.xlim([0,maxx])

        f.add_subplot(2, 2, 2)

        imgplot = plt.imshow(img)

    
        f.add_subplot(2, 2, 3)

        imgplot = plt.imshow(img)
    

    blotches=np.zeros(len(users),dtype=np.int)
    j=0
    for i in users:
            #print i[0]

        cmd='select count(marking) from annotations where image_id='+'"'+image_id+'"'+' and classification_id="'+i[0]+'" and marking="blotch"'
        cursor.execute(cmd)
        fetch=cursor.fetchall()

        blotches[j]=fetch[0][0]
        j=j+1
        cmd='select x,y,radius_1,radius_2, angle from annotations where image_id='+'"'+image_id+'"'+' and classification_id="'+i[0]+'" and X IS NOT NULL and marking="blotch"'
        cursor.execute(cmd)
        fetch=cursor.fetchall() 
        #print fetch
        if (len(fetch) ==0):
                continue 

        index=np.asarray(fetch)
        fetch=0
        x=np.copy(index)
        x=np.delete(x, 1,1)
        x=np.delete(x, 1,1)
        x=np.delete(x, 1,1)
        x=np.delete(x, 1,1)
        x=np.asarray(x, dtype=np.float)

        y=np.copy(index)
        y=np.delete(y, 0,1)
        y=np.delete(y, 1,1)
        y=np.delete(y, 1,1)
        y=np.delete(y, 1,1)
        y=np.asarray(y, dtype=np.float)

        radius_1=np.copy(index)
        radius_1=np.delete(radius_1, 0,1)
        radius_1=np.delete(radius_1, 0,1)
        radius_1=np.delete(radius_1, 1,1)
        radius_1=np.delete(radius_1, 1,1)
        print radius_1
        radius_1=np.asarray(radius_1, dtype=np.float)

        radius_2=np.copy(index)
        radius_2=np.delete(radius_2, 0,1)
        radius_2=np.delete(radius_2, 0,1)
        radius_2=np.delete(radius_2, 0,1)
        radius_2=np.delete(radius_2, 1,1)
        radius_2=np.asarray(radius_2, dtype=np.float)

        angle=np.copy(index)
        angle=np.delete(angle, 0,1)
        angle=np.delete(angle, 0,1)
        angle=np.delete(angle, 0,1)
        angle=np.delete(angle, 0,1)
        angle=np.asarray(angle, dtype=np.float)

        e1 = Ellipse((x, y), radius_1, radius_2, angle=angle, linewidth=2, fill=False, zorder=2)

        ax = plt.gca()
        for i in np.arange(len(x)):
            e1 = Ellipse((x[i], y[i]), radius_1[i], radius_2[i], angle=angle[i], linewidth=0.5, fill=False, zorder=2, color='red')
            ax.add_patch(e1)
    
        
    plt.ylim([maxy,0])
    plt.xlim([0,maxx])

    f.add_subplot(2, 2, 4)
    
    imgplot = plt.imshow(img)
    #plt.show()

    plt.savefig(image_id+'.png', bbox_inches='tight', pad_inches=0.01)

    cursor.close()
    db.close()


#fs= genfromtxt('done_with_markings.csv', delimiter=',', dtype=np.str)
#print fs[0]
#counter=len(fs)
#index=random.sample(xrange(counter), 2000)
index=[0]
fs=['APF0000our']

print index
for i in index:
    print fs[i] 
    # start of the main 
    #image_id='APF0000006'
    image_id=fs[i]
    print image_id 
    print fs[i]

    p4_overplot(image_id)
