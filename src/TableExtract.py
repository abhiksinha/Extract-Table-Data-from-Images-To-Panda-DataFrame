## This code extract Table from images and converts to Pandas DataFrame.


### Library

import cv2
import numpy as np
import pytesseract 
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline








print('started')

### Function which converts images to text. Using Tesseract OCR

def imagetotext2(img):

    
    #img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    ### Applying more filters to the image, you can comment this out if not getting better Results.  
    #img= cv2.medianBlur(img, 3)
                          
    ## path where the tesseract module is installed 
    pytesseract.pytesseract.tesseract_cmd =r'Tesseract-OCR\tesseract.exe'
    tessdata_dir_config = r'--tessdata-dir "Tesseract-OCR\tessdata"'
    
    
    ## Converts the image to text 
    ## Using Pre-Trained DataSet. You can use other data set or train your own.  
    result = pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config) 
    
    return result







### Function which detect the Contors(Rows,coloumns, indivisual cell) in the image

            
def Table_Find(img):
    
    ## Thresholding the image to a binary image
    binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)

    ## Inverting the image 
    binary = ~binary
    cv2.imwrite('inverted.png',binary)

    ## 1/40 of total width
    length = np.array(img).shape[1]//40
    
    ## Horizontal Kernel for horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    
    ## Detecting Horizontal lines in an Image.
    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines)

    
    
    ## Vertical Kernel to detect verticle lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

    #Detecting vertical lines in an Image
    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
    cv2.imwrite("vertical_lines.jpg",vertical_lines)



   
    ## Blending Horizontal and Vertical lines. 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    blended = cv2.addWeighted( src1=horizontal_lines, alpha=0.5,src2=vertical_lines ,beta=0.5, gamma=0)
    
    ##Eroding and thesholding the image. 
    blended = cv2.erode(~blended, kernel, iterations=2)
    thresh, blended = cv2.threshold(blended,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("blended.jpg", blended)
     
    ## Detect contours for following box detection 
    contours, hierarchy = cv2.findContours(blended, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours,hierarchy

########## End of Function################



#### Main Code Now  ##############

### Reading image on GrayScale

img=cv2.imread('table.png',0)

##Finding Contors
con,hie=Table_Find(img)

##Variable which will store the bounding boxes of all contors
box=[]

##Creating Bounding boxes for all contors and maintaing a list.
##Note here 0,1 contors are skipped as they are external contors. You can also filter it out with width,height Bounding Box. 
for i in range(len(con)):
    if i>1:
        box.append(cv2.boundingRect(con[i]))
        
##Reversing the box,as the contors are detected in reverse order i.e from bottom right cell to left thn up.        
box=box[::-1]

##Now we will find total number of rows and column by a trick.
##Assuming that cells are distributed in rows and cols linearly and there is no sub-cells.
##Unpacking the first bounding box and then check after how many boxes the x axis value is repeated. It will become number of cols
## No. of Rows= total length / no of Column

x,y,w,h=box[0]
for i in range(1,len(box)):
    xt,yt,wt,ht=box[i]
    if x==xt:
        break
col=i
row=len(box)//i

##use for plotting 
#fig, axs = plt.subplots(row,col,figsize=(10,10))

##list of texts returns by each cell
data=[]

### for all the cells finding the text or data using imagetotext2 function

for i in range(0,len(box)):
    x,y,w,h=box[i]
    ## Cropping the cell from the original image
    img_crop=img[y:y+h,x:x+w]
    ## Resizing so that tesseract recognize properly
    img_crop = cv2.resize(img_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  
    
    res=imagetotext2(img_crop)
    
    ## Retrying with offset if result is null 
    if res in('','\n',' \n\x0c','\x0c'):
       offset=h//3
       img_crop=img[y-offset:y+h,x:x+w]
       img_crop = cv2.resize(img_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) 
       res=imagetotext2(img_crop)
    
    ##Stripping the new lines
    res=res.rstrip()  
    ##plotting each cell
    #axs[i//col,i%col].imshow(img_crop,cmap='gray')
    ##appending result text to data list
    data.append(res)


print('done')

###Converting the data list to a pandas dataframe


np_array = np.array(data)
##Reshaping in 2d array
np_array = np.reshape(np_array, (row, col))
##Taking first row as heading and converting to dataframe finally.
cols=np_array[0,:]
dataframe = pd.DataFrame(np_array[1:,:],columns=cols)
dataframe.head()

#### Done :)
