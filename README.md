# Wm-with-opencv
build with opencv 3.1.0
the watermark algorithms :
 1, scambler the watermark image with pseudo-random Pemutation
 2, binary the scambler watermark
 3, calculate the CPB : Coefficients Per Blocks
 4, divide origin image to 8*8 blocks, DCT, quantum and zigzag to choise 22 coeffs in middle range
 5, Watermark 
    if (WM == 0)
      C1 > C2;
    else
      C1<C2
  C1,C2 is two subquense coeffs in middle range  
  6, Dequantum and IDCT
  Finish
