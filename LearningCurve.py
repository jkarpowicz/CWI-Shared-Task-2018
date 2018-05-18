#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:44:06 2018

"""
import pandas
import matplotlib.pyplot as plt


Spanish = pandas.read_excel('Spanish.xls', sheet_name='Sheet1')
English = pandas.read_excel('English.xls', sheet_name='English')



fig = plt.figure()
ax = plt.subplot(111)
ax.plot(English.PER_DATA, English.BaseDev, 'b-', label='Baseline Dev')
ax.plot(English.PER_DATA, English.ImpDev, 'r-', label='Improved model Dev')
ax.plot(English.PER_DATA, English.BaseTest, 'y-', label='Baseline Test')
ax.plot(English.PER_DATA, English.ImpTest, 'g-', label='Improved model Test')
plt.xlabel('Part of data used')
plt.ylabel('F1-score')
plt.title('Learning curve - English')
ax.legend(loc='best')
plt.show()




fig = plt.figure()
ax = plt.subplot(111)
ax.plot(Spanish.PER_DATA, Spanish.BaseDev, 'b-', label='Baseline Dev')
ax.plot(Spanish.PER_DATA, Spanish.ImpDev, 'r-', label='Improved model Dev')
ax.plot(Spanish.PER_DATA, Spanish.BaseTest, 'y-', label='Baseline Test')
ax.plot(Spanish.PER_DATA, Spanish.ImpTest, 'g-', label='Improved model Test')
plt.xlabel('Part of data used')
plt.ylabel('F1-score')
plt.ylim(0.72,0.82)
plt.title('Learning curve - Spanish')
ax.legend(loc='best')
plt.show()
