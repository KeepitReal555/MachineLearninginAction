import faiz
import numpy as np


# dataMat, labelMat = faiz.loadDataSet("few.csv")
# print dataMat
# print labelMat

# weights = faiz.gradAscent(dataMat, labelMat)
# weights = faiz.stocGradAscent1(np.array(dataMat), labelMat)
# print weights

# errorRate = faiz.bankTest()

# fileName = "abc.csv"
# index = 199364
# with open(fileName, 'w') as f:
#     f.write('user_id,label\n')
#     for num in range(20):
#         tempstring = str(int( index )+num) + ',' + str(0) + "\n"
#         f.write(tempstring)
# f.close()

faiz.banktestFinal()