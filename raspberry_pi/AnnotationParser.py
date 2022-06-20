# To make python 2 and python 3 compatible code
from __future__ import absolute_import

class AnnotationParser:
    def getAnnotations(self, predictions):
        try:
            listRectangles = []
            listColors = []
            for prediction in predictions:
                print(prediction["boudingbox"])
                for rectList, color in zip(prediction['boundingbox'], prediction['color']):
                    topLeftX = rectList[0]
                    topLeftY = rectList[1]
                    bottomRightX = rectList[2]
                    bottomRightY = rectList[3]
                    listRectangles.append(
                    [topLeftX, topLeftY, bottomRightX, bottomRightY])
                    listColors.append(color)
            return listRectangles, listColors
        except Exception as e:
            print(e)
