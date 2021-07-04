import time
from tabulate import tabulate
import pandas as pd

class TimeKeeper:
    def __init__(self):
        self.pretty_df = pd.DataFrame(columns=['Image','BLEU','Total_Time','Comm_Time'])

        self.E_START_CLIENT_PROCESSING = 'E_START_CLIENT_PROCESSING'
        self.E_STOP_CLIENT_PROCESSING = 'E_STOP_CLIENT_PROCESSING'
        self.E_START_COMMUNICATION = 'E_START_COMMUNICATION'
        self.E_STOP_COMMUNICATION = 'E_STOP_COMMUNICATION'

        self.I_BUFFER_SIZE = 'I_BUFFER_SIZE'
        self.I_BLEU = 'I_BLEU'
        self.I_REAL_CAPTION = 'I_REAL_CAPTION'
        self.I_PRED_CAPTION = 'I_PRED_CAPTION'
        self.I_CLIENT_PROCESSING_TIME = 'I_CLIENT_PROCESSING_TIME'
        self.I_COMMUNICATION_TIME = 'I_COMMUNICATION_TIME'

        self.records = {}

    def startRecord(self, image):
        self.records[image] = {}

    def logTime(self, image, event):
        self.records[image][event] = time.perf_counter()

    def logInfo(self, image, info_key, info):
        self.records[image][info_key] = info

    def finishRecord(self, image):
        # self.records[image] = {}
        self.records[image][self.I_CLIENT_PROCESSING_TIME] = self.records[image][self.E_STOP_CLIENT_PROCESSING] - \
            self.records[image][self.E_START_CLIENT_PROCESSING]
        self.records[image][self.I_COMMUNICATION_TIME] = self.records[image][self.E_STOP_COMMUNICATION] - \
            self.records[image][self.E_START_COMMUNICATION]
        record = self.records[image]
        pretty_record = {}
        pretty_record['Image'] = image.rsplit('/', 1)[-1]
        pretty_record['BLEU'] = "{:.2f}".format(self.records[image][self.I_BLEU])
        pretty_record['Total_Time'] = "{:.2f}".format(self.records[image][self.I_BLEU])
        pretty_record['Comm_Time'] = "{:.2f}".format(self.records[image][self.I_BLEU])
        self.pretty_df = self.pretty_df.append(pretty_record,ignore_index=True)
        pass

    def printAll(self):
        print(tabulate(self.pretty_df, headers='keys', tablefmt='psql'))

    def summary(self):
        df = pd.DataFrame(self.records)
        df_t = df.T
        print("Average BLEU : %.2f" % (df_t[self.I_BLEU].mean()))
        print("Average client processing : %.2f s" % (df_t[self.I_CLIENT_PROCESSING_TIME].mean()))
        print("Average communication time : %.2f s" % (df_t[self.I_COMMUNICATION_TIME].mean()))

        pass
