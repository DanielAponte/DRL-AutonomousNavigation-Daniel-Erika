import matplotlib.pyplot as plt
import xlsxwriter

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
data = fileM.exportData('logs_info2407.log')
fileM.writeXlsxFile(data, 'data')

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


