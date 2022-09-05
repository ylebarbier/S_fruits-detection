import sys
import os
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from shapely import wkt

import yolov5.detect

from cytomine.models import Annotation
from cytomine import CytomineJob
from cytomine.models.image import ImageInstance, ImageInstanceCollection


mapping = {0: 549437565, 1: 549437575, 2: 549437583,3: 549437593, 4: 549437601, 5: 549437615, 6: 549437623 ,7: 549437631 ,8: 549437641, 9: 549437649, 10: 549437657,
           11: 549437667, 12: 549437675, 13: 549437683, 14: 549437693, 15: 549437701, 16: 549437709, 17: 549437717, 18: 549437727, 19: 549437735, 20: 549437743,
           21: 549437751, 22: 549437761, 23: 549437769, 24: 549437777, 25: 549437785, 26: 549437793, 27: 549437803, 28: 549437811, 29: 549437919, 30: 549437338,
           31: 549437827}

def main(argv):
     with CytomineJob.from_cli(argv) as cj:
        params = cj.parameters
      
        images = ImageInstanceCollection().fetch_with_filter("project", params.cytomine_id_project)

        if params.images != 'all':
            ids = [int(id) for id in params.images.split(',')]
            images = {image.id: image for image in images}
            images = [images[id] for id in ids]
        
        for image in images:

            image.dump()


            imw = image.width
            imh = image.height

            yolov5.detect.run(weights='/fruits_detection/yolo.pt', source='{}.jpg'.format(image.id), save_txt=True, project='/fruits_detection', exist_ok=True)

            if not os.path.exists("/fruits_detection/exp/labels/{}.txt".format(image.id)):
                continue

            df = pd.read_csv("/fruits_detection/exp/labels/{}.txt".format(image.id), header=None, sep=" ", names=['label', 'x', 'y', 'w', 'h'])

            for i in range(len(df)):
                x = df.loc[i]['x']
                y = df.loc[i]['y']
                w = df.loc[i]['w']
                h = df.loc[i]['h']
                label = df.loc[i]['label']

                x *= imw
                y *= imh
                w *= imw
                h *= imh

                y = imh - y

                x1 = x - w/2
                x2 = x + w/2
                y1 = y - h/2
                y2 = y + h/2

                geometry = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1), (x1,y1)])
                
                annotation = Annotation()
                annotation.project = params.cytomine_id_project
                annotation.image = image.id
                annotation.location = wkt.dumps(geometry)
                annotation.term = mapping[label]
                annotation.save()

if __name__ == "__main__":
    main(sys.argv[1:])