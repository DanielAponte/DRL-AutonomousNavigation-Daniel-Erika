import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np         #array library

class fileMngmnt():
    def exportData(self, fileName):
        rewardData = ['Reward']  
        timeData = ['Time dif']
        actsData = ['Total Acts'] 
        qTableData = ['W table Acts'] 

        with open(fileName) as temp_f:
            datafile = temp_f.readlines()

        for line in datafile:
            if 'Episode_reward' in line:
                data = line.split('Episode_reward ')[1].replace('\n', "")
                rewardData.append(float(data))
            elif 'Time dif' in line: 
                data = line.split('Time dif: ')[1].replace('\n', "")
                timeData.append(float(data))
            elif 'Total acts' in line: 
                data = line.split('Total acts: ')[1].split(' ')[0].replace('\n', "")
                actsData.append(float(data))       
                data = line.split('Q-Table: ')[1].split(' ')[0].replace('\n', "")
                qTableData.append(float(data))         

        processData = [rewardData, timeData, actsData, qTableData]

        return processData

    def exportDataFromLog(self, fileName):
        rewardData = []
        actsData = [] 
        errorActs = []
        epsilonData = []

        with open(fileName) as temp_f:
            datafile = temp_f.readlines()

        for line in datafile:
            if 'Reward_hist' in line:
                rewardData = line.split('Reward_hist[')[1].replace('\n', "").replace(']', "").split(", ")
            elif 'Epsilon_hist' in line:
                epsilonData = line.split('Epsilon_hist[')[1].replace('\n', "").replace(']', "").split(", ")
            elif 'Total_acts_hist' in line:
                actsData = line.split('Total_acts_hist[')[1].replace('\n', "").replace(']', "").split(", ")
            elif 'Total_error_acts_hist' in line:
                errorActs = line.split('Total_error_acts_hist[')[1].replace('\n', "").replace(']', "").split(", ")

        processData = [rewardData, epsilonData, actsData, errorActs]
        for data in processData:
            index = 0
            for num in data:
                data[index] = float(num)
                index += 1

        return processData


    def writeXlsxFile(self, data, fileName):
        workbook = xlsxwriter.Workbook(fileName + '.xlsx')
        worksheet = workbook.add_worksheet()
        row = 0
        for col, d in enumerate(data):
            worksheet.write_column(row, col, d)
        workbook.close()

    def getRewardData(self, fileName):
        rewardData = []    
        with open(fileName) as temp_f:
            datafile = temp_f.readlines()
        for line in datafile:
            if 'Episode_reward' in line:
                data = line.split('Episode_reward ')[1].replace('\n', "")
                rewardData.append(float(data))
        return rewardData

    def getTimeData(self, fileName):
        timeData = []
        with open(fileName) as temp_f:
            datafile = temp_f.readlines()
        for line in datafile:
            if 'Time dif' in line: 
                data = line.split('Time dif: ')[1].replace('\n', "")
                timeData.append(float(data))            

        return timeData

    def getActsData(self, fileName):
        actsData = []
        with open(fileName) as temp_f:
            datafile = temp_f.readlines()
        for line in datafile:
            if 'Total acts' in line: 
                data = line.split('Total acts: ')[1].split(' ')[0].replace('\n', "")
                actsData.append(float(data))            

        return actsData

    def getQTableData(self, fileName):
        qTableData = []
        with open(fileName) as temp_f:
            datafile = temp_f.readlines()
        for line in datafile:
            if 'Total acts' in line: 
                data = line.split('Q-Table: ')[1].split(' ')[0].replace('\n', "")
                qTableData.append(float(data))            

        return qTableData

    def printData(self, data, label):
        plt.plot(list(range(len(data))), data)
        plt.ylabel(label)
        plt.show()

fileM = fileMngmnt()
data = fileM.exportDataFromLog('logs_info2022-10-05.log')
fileM.writeXlsxFile(data, 'data_train')

# rewardData = fileM.getRewardData('logs_info2407.log')
# timeData = fileM.getTimeData('logs_info2407.log')
# print(timeData)
# actsData = fileM.getActsData('logs_info2407.log')
# qtableData = fileM.getQTableData('logs_info2407.log')
# fileM.printData(rewardData, 'Reward')
# fileM.printData(timeData, 'Time')
# fileM.printData(actsData, 'Acts')
# fileM.printData(qtableData, 'QTable')
# print('Bye')


