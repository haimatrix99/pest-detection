# To make python 2 and python 3 compatible code
from __future__ import absolute_import

class AnnotationParser:
    def getAnnotations(self, predictions):
        try:
            listRectangles = []
            listColors = []
            for prediction in predictions:
                listRectangles.append(prediction['boundingbox'])
                listColors.append(prediction['color'])
            return listRectangles, listColors
        except Exception as e:
            print(e)
