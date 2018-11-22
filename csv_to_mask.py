import io
import base64
import xlrd
import pandas as pd
import io
f = io.open('/home/panchenko/Documents/Panchenko_Sergey_report_10_2017.xls')


l = [line for line in io.open('/home/panchenko/Documents/Panchenko_Sergey_report_10_2017.xls', encoding="UTF-8")]


print l[6][:l[6].find("9")+5]