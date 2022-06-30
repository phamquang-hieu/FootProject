import streamlit as st
from main import calc 
import cv2
from skimage.io import imread
from PIL import Image
import numpy as np
import pandas as pd

def load_image(image_file):
    img = Image.open(image_file)
    return img

def parse_row(row):
    if "-" in row:
        low, high = np.array(row.split("-"), dtype=float)
        return (low, high)
    return [float(row)]


if __name__ == '__main__':
    st.title("Foot Measurement!")
    st.subheader("User instructions")
    with open("./data/instructions.txt") as f:
        for line in f:
            st.markdown(line)
    
    st.subheader("An example:")
    example_img = imread("./data/images/23801.jpeg")
    st.image(example_img, width=example_img.shape[0]//10)
    
    st.subheader("Upload your image here")
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    with open("data/size_converter.csv") as f:
        cvt_table = pd.read_csv(f)
    if image_file is not None:
       
        file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
        
        img = load_image(image_file)
        # print(img.size[])
        
        st.image(img, width=img.size[1]//10)
        selection = st.selectbox("Which dimension of your foot do you want to choose for shoes size recomendation?", ("length", "width"))
        color_system = st.selectbox("Select the option that you find your foot and the paper are best separated out from the background \n (You may have to run a multiple time, sorry for this inconvinient)", ('RGB', 'HSV', 'LAB', 'YCrCb'))
        if st.button("RUN"):
            st.text("RUNNING, PLEASE WAIT")
            try:
                results, stages = calc(image_file, color_sys=color_system)
            except Exception:
                st.text("UNABLE TO RECOGNIZE YOUR FOOT, PLEASE TRY AGAIN!")
                
            cols = st.columns(len(stages))
    
            for i, stage in enumerate(stages):
                with cols[i]:
                    st.image(stage, width=stage.shape[1]//5)
            
            # base on selection -> coversion table
            measure_ranges = cvt_table[selection]
            for idx, r in enumerate(measure_ranges):
                r = parse_row(r)
                if (idx == 0 and results[selection] < r[0]) or (idx == len(measure_ranges) and result > r[-1]):
                    size = cvt_table['size'][idx]
                    break
                    
                if len(r)==2 and r[0] < results[selection] < r[1]:
                    size = cvt_table['size'][idx]
                    break
                elif results[selection] == r[0]:
                    size = cvt_table['size'][idx]
                    break
            
            st.text(f"your foot's {selection} is: {results[selection]} corresponds to size number: {size}")
            st.text(results)
            
            
            
    